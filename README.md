This repository is an adaptation of [AP-PA](https://github.com/JiaweiLian/AP-PA).

## PNAP-YOLO: An Improved Prompts-Based Naturalistic Adversarial Patch Model for Object Detectors

## Naturalistic Physical Adversarial Patch for Object Detectors

### Abstract

Detectors have been extensively utilized in various scenarios such as autonomous driving and video surveillance.
Nonetheless, recent studies have revealed that these detectors are vulnerable to adversarial attacks, particularly
adversarial patch attacks. Adversarial patches are specifically crafted to disrupt deep learning models by disturbing
image regions, thereby misleading the deep learning models when added to into normal images. Traditional adversarial
patches often lack semantics, posing challenges in maintaining concealment in physical world scenarios. To tackle this
issue, this paper proposes a Prompt-based Natural Adversarial Patch (PNAP) generation method, which creates patches
controllable by textual descriptions to ensure flexibility in application. This approach leverages the latest
text-to-image generation model—Latent Diffusion Model (LDM) to produce adversarial patches. We optimize the attack
performance of the patches by updating the latent variables of LDM through a combined loss function. Experimental
results indicate that our method can generate more natural, semantically rich adversarial patches, achieving effective
attacks on various detectors.

## Installation

### Clone the code and build the environment

Clone the code:

```bash
git clone https://github.com/Kwizii/PNAP
cd PNAP
```

Import the conda environment

```bash
conda env create -f pnap.yaml
conda activate pnap
```

### Dataset

Download the INRIA dataset
from [Naturalistic-Adversarial-Patch](https://github.com/aiiu-lab/Naturalistic-Adversarial-Patch).

The original INRIA dataset comes from [INRIA](http://pascal.inrialpes.fr/data/human/).

Check the dataset position:

```
Naturalistic-Adversarial-Patch                           
 └─── dataset
        └───── inria
                └───── Test
                        └───── ...
                └───── Train
                        └───── pos
                                └───── yolo-labels_yolov4tiny
                                └───── *.png
                                └───── ...
 
```

### Pretrained weights

You can download the necessary weights by running the following command:

- YOLOv5 and YOLOv8: Automatic downloading when running code.
- YOLOv4、YOLOv4tiny、YOLOv3、YOLOv3tiny and YOlOv2: Downloading weights
  from [Naturalistic-Adversarial-Patch](https://github.com/aiiu-lab/Naturalistic-Adversarial-Patch).

## How to Run

You should choose a right config in `patch_config.py` when running code.

We use WandB to visualize the training and validation process.

### Test an adversarial patch:

```bash
python evaluation.py
```

### Train an adversarial patch:

```bash
python train_patch.py
```

## Credits

- YOLOv2 and adversarial patch codes are based on: [adversarial-yolo](https://gitlab.com/EAVISE/adversarial-yolo)
- YOLOv3 code and weights are based on: [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)
- YOLOv4 code and weights are based on: [pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4)
- YOLOv5 code and weights are based on: [Ultralytics-YOLOv5](https://github.com/ultralytics/yolov5)
- YOLOv8 code and weights are based on: [Ultralytics-YOLOv8](https://github.com/ultralytics/ultralytics)
