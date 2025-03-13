"""
Adversarial patch training
"""
import shutil
import time
import warnings

import pandas as pd
import psutil
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_timesteps
from torch import optim
from torchvision import transforms as T
from tqdm import tqdm

import patch_config
import post_util
import wandb
from load_data import *
from patch_embed import init_ldm, prompt_embedding, gen_patch
from post_util import plot_boxes
from tools import map_cal
from utils.general import xywh2xyxy

warnings.filterwarnings("ignore")

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

# Expand to show
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)


class PatchTrainer:
    def __init__(self, mode):
        self.epoch_length = 0
        if isinstance(mode, patch_config.BaseConfig):
            self.config = mode
        else:
            self.config = patch_config.patch_configs[mode]()
        self.model = self.config.model.cuda()
        self.prob_extractor = self.config.prob_extractor.cuda()

        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer(scale=self.config.scale,
                                                  minangle=self.config.minangle,
                                                  maxangle=self.config.maxangle,
                                                  min_brightness=self.config.min_brightness,
                                                  max_brightness=self.config.max_brightness,
                                                  offsetx=self.config.offsetx,
                                                  offsety=self.config.offsety,
                                                  min_contrast=self.config.min_contrast,
                                                  max_contrast=self.config.max_contrast,
                                                  noise_factor=self.config.noise_factor).cuda()
        self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.total_variation = TotalVariation().cuda()

        self.train_loader = torch.utils.data.DataLoader(
            InriaDataset(self.config.img_dir, self.config.lab_dir, self.config.img_size,
                         cls_ids=self.config.cls_id, pad=True),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=16,
            collate_fn=collate_fn,
        )
        self.val_loader = torch.utils.data.DataLoader(
            InriaDataset(self.config.val_img_dir, self.config.val_lab_dir, self.config.img_size,
                         cls_ids=self.config.cls_id, pad=False),
            batch_size=int(self.config.batch_size * 1.5),
            shuffle=True,
            num_workers=16,
            collate_fn=collate_fn)
        self.process = psutil.Process()

    def train(self, start_epoch=0):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """
        cfg = self.config

        n_epochs = 500
        epoch_length = len(self.train_loader)
        time_str = time.strftime("%Y%m%d-%H%M%S")
        # Generate stating point
        pipeline = init_ldm(device=cfg.device, dtype=cfg.dtype)
        timesteps = torch.linspace(cfg.start_time_step, cfg.end_time_step,
                                   cfg.num_inference_steps, dtype=torch.int)
        # if latents is None:
        latents = torch.randn((1, pipeline.unet.config.in_channels, 64, 64),
                              generator=cfg.generator,
                              device=pipeline.unet.device,
                              dtype=cfg.dtype)
        with torch.no_grad():
            latents *= pipeline.scheduler.init_noise_sigma
            prompt_embeds = prompt_embedding(pipeline, cfg.prompt, cfg.negative_prompt,
                                             cfg.do_classifier_free_guidance, cfg.device)
            timesteps1, _ = retrieve_timesteps(pipeline.scheduler, cfg.init_num_inference_steps, cfg.device)
            latents, init_patch = gen_patch(pipeline, latents, prompt_embeds,
                                            timesteps1,
                                            cfg.init_num_inference_steps,
                                            cfg.do_classifier_free_guidance,
                                            cfg.guidance_scale, cfg.generator, cfg.device)
            latents = latents.to(pipeline.unet.device)
            noise = torch.randn_like(latents)
            latents = pipeline.scheduler.add_noise(latents, noise, timesteps[:1])
            wandb.log({
                "Patches": wandb.Image(init_patch, caption="patch_init"),
                "Timesteps": timesteps,
            })
        latents.requires_grad_(True)
        optimizer = optim.Adam([latents], lr=cfg.start_learning_rate, amsgrad=True)
        # optimizer = optim.SGD([latents], lr=cfg.start_learning_rate)
        scheduler = self.config.scheduler_factory(optimizer)

        wandb.watch_called = False  # Re-run the model without restarting the runtime, unnecessary after our next release
        wandb.watch(self.model, log="all")

        et0 = time.time()

        for epoch in range(start_epoch, n_epochs):
            torch.cuda.empty_cache()

            ep_det_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            bt0 = time.time()
            desc = 'Running epoch %d, loss %f'
            pbar = tqdm(enumerate(self.train_loader),
                        desc=desc % (epoch, 0),
                        total=epoch_length)
            for i_batch, (img_batch, lab_batch, idx_batch, _) in pbar:
                img_batch = img_batch.cuda()
                lab_batch = lab_batch.cuda()
                idx_batch = idx_batch.cuda()
                _, adv_patch = gen_patch(pipeline, latents, prompt_embeds, timesteps, len(timesteps),
                                         cfg.do_classifier_free_guidance,
                                         cfg.guidance_scale, cfg.generator, cfg.device)
                adv_batch_t, _, _ = self.patch_transformer(adv_patch, lab_batch, cfg.img_size,
                                                           rand_loc=cfg.rand_loc,
                                                           by_rectangle=cfg.by_rect)
                p_img_batch = self.patch_applier(img_batch, adv_batch_t, idx_batch)
                p_img_batch = F.interpolate(p_img_batch, cfg.imgsz)

                output = self.model(p_img_batch)

                extracted_prob = self.prob_extractor(output)
                nps = self.nps_calculator(adv_patch)
                tv = self.total_variation(adv_patch)

                nps_loss = nps * 0.01
                tv_loss = tv * 2.5
                det_loss = torch.mean(extracted_prob)
                # loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())
                loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())

                ep_det_loss += det_loss.item()
                ep_nps_loss += nps_loss.item()
                ep_tv_loss += tv_loss.item()
                ep_loss += loss.item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_description(desc % (epoch, loss))

                # if i_batch % (epoch_length // 2) == 0:
                #     torch.cuda.empty_cache()
                #     with torch.no_grad():
                #         iteration = epoch_length * epoch + i_batch
                #         boxes = preds2boxes(cfg, output)
                #         train_imgs = post_util.grid_images(
                #             [plot_boxes(p_img_batch[i], boxes[i], class_names=cfg.class_names) for i in
                #              range(len(p_img_batch))])
                #         map50, img_preds, patch_img_preds = self.val(adv_patch)
                #         wandb.log({
                #             "val/Patches": wandb.Image(adv_patch, caption="patch{}".format(iteration)),
                #             "train/Img": wandb.Image(train_imgs),
                #             "val/tv_loss": ep_tv_loss / (i_batch + 1),
                #             "val/nps_loss": ep_nps_loss / (i_batch + 1),
                #             "val/det_loss": ep_det_loss / (i_batch + 1),
                #             "val/total_loss": loss,
                #             'val/map': map50,
                #             'val/img_pred': wandb.Image(img_preds),
                #             'val/patch_img_pred': wandb.Image(patch_img_preds),
                #             "val/step": step
                #         })
                #         step += 1
            et1 = time.time()
            ep_det_loss = ep_det_loss / epoch_length
            ep_nps_loss = ep_nps_loss / epoch_length
            ep_tv_loss = ep_tv_loss / epoch_length
            ep_loss = ep_loss / epoch_length

            scheduler.step(ep_loss)

            et0 = time.time()

            torch.cuda.empty_cache()
            with torch.no_grad():
                boxes = preds2boxes(cfg, output)
                train_imgs = post_util.grid_images(
                    [plot_boxes(p_img_batch[i], boxes[i], class_names=cfg.class_names) for i in
                     range(len(p_img_batch))])
                map50, img_preds, patch_img_preds = self.val(adv_patch)
                wandb.log({
                    "train/patch_img_pred": wandb.Image(train_imgs),
                    "train/Patches": wandb.Image(adv_patch),
                    "train/tv_loss": ep_tv_loss,
                    "train/nps_loss": ep_nps_loss,
                    "train/det_loss": ep_det_loss,
                    "train/total_loss": ep_loss,
                    "train/time": et1 - et0,
                    'train/step': epoch,
                    'val/map': map50,
                    'val/img_pred': wandb.Image(img_preds),
                    'val/patch_img_pred': wandb.Image(patch_img_preds),
                    "val/step": epoch,
                })
            # wandb.log({
            #     "train/Patches": wandb.Image(adv_patch),
            #     "train/tv_loss": ep_tv_loss,
            #     "train/nps_loss": ep_nps_loss,
            #     "train/det_loss": ep_det_loss,
            #     "train/total_loss": ep_loss,
            #     "train/time": et1 - et0,
            #     'train/step': epoch
            # })

    def val(self, adv_patch, cal_map=True):
        cfg = self.config
        gt_path = os.path.join('temp', 'gt')
        dr_path = os.path.join('temp', 'dr')
        if os.path.exists(gt_path):
            shutil.rmtree(gt_path)
        if os.path.exists(dr_path):
            shutil.rmtree(dr_path)
        os.makedirs(dr_path)
        os.makedirs(gt_path)

        bn = len(self.val_loader)
        bidx = np.random.randint(0, bn)
        cnt = 0

        for bi, (img_batch, lab_batch, idx_batch, paths) in tqdm(enumerate(self.val_loader),
                                                                 desc="Validation: ",
                                                                 total=len(self.val_loader)):
            img_batch = img_batch.cuda()
            lab_batch = lab_batch.cuda()
            idx_batch = idx_batch.cuda()

            with torch.no_grad():
                output = self.model(F.interpolate(img_batch, cfg.imgsz))
                output = preds2boxes(cfg, output)
                adv_batch_t, _, _ = self.patch_transformer(adv_patch, lab_batch, cfg.img_size,
                                                           do_blur=False,
                                                           do_rotate=False,
                                                           rand_loc=False,
                                                           do_aug=False,
                                                           by_rectangle=cfg.by_rect)
                p_img_batch = self.patch_applier(img_batch.clone(), adv_batch_t, idx_batch)
                patch_output = self.model(F.interpolate(p_img_batch, cfg.imgsz))
                patch_output = preds2boxes(cfg, patch_output)

            # if bi == bidx:
            if True:
                img_preds = post_util.grid_images(
                    [plot_boxes(img_batch[i], output[i], class_names=cfg.class_names) for i in
                     range(len(img_batch))])
                patch_img_preds = post_util.grid_images(
                    [plot_boxes(p_img_batch[i], patch_output[i], class_names=cfg.class_names) for i in
                     range(len(p_img_batch))])
                # save_imgs
                img_preds.save(f'pred/{cnt}.jpg')
                patch_img_preds.save(f'pred/patch_{cnt}.jpg')
                cnt += 1
            if not cal_map:
                continue
            for i, (gt_pred, dr_pred, img_path) in enumerate(zip(output, patch_output, paths[0])):
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                gt_box = gt_pred[:, :4]
                dr_box = dr_pred[:, :4]
                dr_confs = dr_pred[:, 4]
                gt_box = gt_box.cpu().numpy()
                dr_box = dr_box.cpu().numpy()
                gt_box = xywh2xyxy(gt_box)
                dr_box = xywh2xyxy(dr_box)
                gt_box = np.round(gt_box * cfg.img_size)
                dr_box = np.round(dr_box * cfg.img_size)
                with open(os.path.join(gt_path, f'{img_name}.txt'), 'w') as f:
                    for i, box in enumerate(gt_box):
                        coords = ' '.join(map(str, box))
                        f.write(f'{int(gt_pred[i, -1].item())} {coords}\n')
                    f.close()
                with open(os.path.join(dr_path, f'{img_name}.txt'), 'w') as f:
                    for i, (box, dr_conf) in enumerate(zip(dr_box, dr_confs)):
                        coords = ' '.join(map(str, box))
                        f.write(f'{int(dr_pred[i, -1].item())} {dr_conf} {coords}' + '\n')
                    f.close()
        map50 = map_cal.count(path_ground_truth=gt_path,
                              path_detection_results=dr_path) if cal_map else None
        return map50, img_preds, patch_img_preds

    def read_image(self, path):
        patch_img = Image.open(path).convert('RGB')
        transforms = T.Compose([T.Resize((self.config.patch_size, self.config.patch_size)), T.ToTensor()])
        return transforms(patch_img).cuda()


def main():
    # trainer = PatchTrainer(sys.argv[1])
    # for p in ['yolov3_dota']:
    # for s in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    # trainer = PatchTrainer(p)
    cfg = patch_config.yolov3()
    # cfg.batch_size = 1
    trainer = PatchTrainer(cfg)
    wandb.init(project="Adversarial-attack", config=dict(
        name=trainer.config.patch_name,
        batch_size=trainer.config.batch_size,
        scale=trainer.config.scale,
        lab_dir=trainer.config.lab_dir,
        seed=trainer.config.seed,
        lr=trainer.config.start_learning_rate,
        cls_id=trainer.config.cls_id,
        patch_size=trainer.config.patch_size,
        img_size=trainer.config.img_size,
        imgsz=trainer.config.imgsz,
        conf_thres=trainer.config.img_size,
        iou_thres=trainer.config.img_size,
        max_det=trainer.config.max_det,
        prompt=trainer.config.prompt[0],
        negative_prompt=trainer.config.negative_prompt[0],
        guidance_scale=trainer.config.guidance_scale,
        init_num_inference_steps=trainer.config.init_num_inference_steps,
        num_inference_steps=trainer.config.num_inference_steps,
        start_time_step=trainer.config.start_time_step,
        end_time_step=trainer.config.end_time_step,
        do_classifier_free_guidance=trainer.config.do_classifier_free_guidance,
    ))
    trainer.train()
    wandb.finish()


if __name__ == '__main__':
    main()
