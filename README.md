# A Multi-Level Training Supervision Framework for Lightweight Facial Expression Recognition

**Official PyTorch implementation of POSTER V2-ResNet18 with DR-CAC.**

## Requirements

Please install the required packages:

```bash
pip install -r requirements.txt
```

## Datasets

The original datasets are not included in this repository. Please download them from their official or public sources and follow the corresponding license or access conditions.

* RAF-DB: http://www.whdeng.cn/RAF/model1.html
* CAER-S: https://caer-dataset.github.io/
* FANE: https://www.kaggle.com/datasets/furcifer/fane-facial-expressions-and-emotion-dataset

## Pretrained and Teacher Checkpoints

Before training, please place the required pretrained backbone weights and POSTER++ teacher checkpoints under:

```text
models/pretrain/
  raf-db-model_best.pth
  caer-s-model_best.pth
  fane-model_best.pth
  ir50.pth
  mobilefacenet_model_best.pth.tar
  resnet18_msceleb.pth
```
These files include pretrained backbone weights and the POSTER++ teacher checkpoints used as knowledge sources in this study. For FANE, a POSTER++ teacher checkpoint was trained under the adopted dataset setting because the original POSTER++ work did not provide a checkpoint for this dataset.

Due to license and file-size considerations, the pretrained backbone weights and teacher checkpoints listed above are not redistributed directly in this repository. Users should prepare the required files and place them in the specified directory before training.


## Training

Please run the training scripts from the project root directory.

The dataset-specific training entry points are:

- RAF-DB: `train_distill.py`
- CAER-S: `train_distill-cears.py`
- FANE: `train_distill_9.py`

The training settings used for the reported results, including the number of epochs, batch size, learning rate, distillation temperature, loss coefficients, and focal-loss parameter, are described in the manuscript and implemented in the corresponding training scripts.

Before training, please ensure that the datasets and required pretrained/teacher checkpoints are organized according to the directory structures described above.

## Results

| Dataset | Top-1 Accuracy (%) | Parameters | FLOPs |
| ------- | -----------------: | ---------: | ----: |
| RAF-DB  | 90.97 | 20.89M | 3.82G |
| CAER-S  | 92.16 | 20.89M | 3.82G |
| FANE    | 73.79 | 20.89M | 3.82G |

## Released Model Checkpoints

The trained POSTER V2-ResNet18 checkpoints corresponding to the results reported in the manuscript are available from:

[Google Drive checkpoint folder](https://drive.google.com/drive/folders/17AhweJCFLquS3k7MaTEyw6AoKj5BQPA6?usp=sharing)

## Citation
Citation information will be updated upon publication.

## License
This repository is released under the MIT License. See [LICENSE](LICENSE) for details.
Datasets, pretrained weights, and other external resources are subject to their respective original licenses and terms of use.
