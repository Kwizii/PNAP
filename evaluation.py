import pandas as pd
import torch.cuda

from train_patch import PatchTrainer


def main():
    # 选择测试补丁用于攻击
    patches = ["patches/PNAP/v3tiny-mix-67.png"]
    # patches = []
    # for p1 in os.listdir('patches'):
    #     p1 = os.path.join('patches', p1)
    #     if os.path.isdir(p1):
    #         for p2 in os.listdir(p1):
    #             patches.append(os.path.join(p1, p2))
    #     else:
    #         patches.append(p1)
    # patches = sorted(list(filter(lambda x: x.endswith(('.png', '.jpg')), patches)))
    data = {
        'patch': patches,
    }
    # 根据配置选择攻击数据集
    # for m in patch_config.patch_configs:
    for m in ['yolov3tiny', 'yolov3tiny-mpii', 'yolov3tiny-mix']:
        map50s = []
        trainer = PatchTrainer(m)
        with torch.no_grad():
            for p in patches:
                # 计算map
                patch = trainer.read_image(p)
                map50, _, _ = trainer.val(patch)
                map50s.append(map50)
                print(f'{m}-{p}: {map50}')
        data[m] = map50s
        del trainer
        torch.cuda.empty_cache()
    df = pd.DataFrame(data)
    df.to_csv(f'result.csv', index=False)


if __name__ == '__main__':
    main()
