import torch
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler


def init_ldm(device, dtype):
    # load sd
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained('runwayml/stable-diffusion-v1-5',
                                                              torch_dtype=dtype,
                                                              cache_dir='..',
                                                              requires_safety_checker=False,
                                                              local_files_only=True,
                                                              use_safetensors=True).to('cuda:1')
    pipeline.text_encoder = pipeline.text_encoder.to(device)
    pipeline.vae = pipeline.vae.to(device)
    pipeline.unet.eval()
    pipeline.vae.eval()
    pipeline.text_encoder.eval()
    # change scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    return pipeline


def prompt_embedding(pipeline, prompt, negative_prompt, do_classifier_free_guidance, device):
    prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
        prompt,
        device,
        1,
        do_classifier_free_guidance,
        negative_prompt,
    )
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    return prompt_embeds.detach()


def gen_patch(pipeline, latents, prompt_embeds, timesteps, num_inference_steps, do_classifier_free_guidance,
              guidance_scale, generator, device, decode=True):
    unet_device = pipeline.unet.device
    latents = latents.to(unet_device)
    prompt_embeds = prompt_embeds.to(unet_device)
    timesteps = timesteps.to(unet_device)
    pipeline.scheduler.num_inference_steps = num_inference_steps
    pipeline.scheduler.timesteps = timesteps
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, 0.)
    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)
        # predict the noise residual
        noise_pred = pipeline.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]
        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        # compute the previous noisy sample x_t -> x_t-1
        latents = pipeline.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
    latents = latents.to(device)
    if decode:
        patch = \
            pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
        patch = pipeline.image_processor.denormalize(patch)
        return latents, patch[0]
    else:
        return latents, None
