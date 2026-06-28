# POSTER V2-ResNet18 with NA-MSAC

This repository provides the PyTorch implementation of the manuscript:

**Efficient Facial Expression Recognition via Multi-Level Knowledge Transfer and Noise-Aware Multi-Scale Attention Consistency**

The proposed method trains a lightweight **POSTER V2-ResNet18** model for facial expression recognition (FER). The training framework combines logits distillation, feature representation alignment, and Noise-Aware Multi-Scale Attention Consistency (NA-MSAC). During inference, only POSTER V2-ResNet18 is used.

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

Please organize the datasets as follows:

```text
data_preprocessing/
  raf-db-divide-7folders/
    train/
    valid/
  CAER-S-divide-7folders/
    train/
    valid/
  FANE-divide-9folders/
    train/
    valid/
```

The folder `valid/` is used as the evaluation split in the training scripts.

## Required Checkpoints

Please place the required checkpoints under:

```text
models/pretrain/
  raf-db-model_best.pth
  caer-s-model_best.pth
  fane-model_best.pth
  ir50.pth
  mobilefacenet_model_best.pth.tar
  resnet18_msceleb.pth
```
These checkpoints include publicly available pretrained weights and teacher checkpoints used in this study. The FANE teacher checkpoint was trained by the authors using the POSTER++ architecture because the original POSTER++ paper did not provide a FANE checkpoint.

Due to license and file-size considerations, these checkpoint files are not redistributed in this repository. Users should prepare the required files and place them in the specified directory before training.


## Training

Please run all commands from the project root directory.

### RAF-DB

```bash
python train_distill.py \
  --data ./data_preprocessing/raf-db-divide-7folders \
  --epochs 200 \
  --batch-size 64 \
  --lr 1.0e-4 \
  --alpha 0.6 \
  --temperature 4.0 \
  --lambda_feature 0.7 \
  --lambda_na_msac 1.0 \
  --focal_gamma 2.0
```

### CAER-S

```bash
python train_distill-cears.py \
  --data ./data_preprocessing/CAER-S-divide-7folders \
  --epochs 250 \
  --batch-size 96 \
  --lr 1.5e-4 \
  --alpha 0.3 \
  --temperature 4.0 \
  --lambda_feature 0.7 \
  --lambda_na_msac 1.0 \
  --focal_gamma 1.0
```

### FANE

```bash
python train_distill_9.py \
  --data ./data_preprocessing/FANE-divide-9folders \
  --epochs 200 \
  --batch-size 128 \
  --lr 2.0e-4 \
  --alpha 0.6 \
  --temperature 4.0 \
  --lambda_feature 0.7 \
  --lambda_na_msac 1.0 \
  --focal_gamma 2.5
```

## Results

| Dataset | Accuracy (%) | Parameters | FLOPs |
| ------- | -----------: | ---------: | ----: |
| RAF-DB  |        91.17 |     20.89M | 3.82G |
| CAER-S  |        92.36 |     20.89M | 3.82G |
| FANE    |        73.67 |     20.89M | 3.82G |

## Checkpoints and Logs

Training checkpoints are saved in:

```text
checkpoints/
```

Training logs are saved in:

```text
log/
log_caers/
log_Fane/
```

The final trained checkpoints corresponding to the reported best results are available from Google Drive:

[Google Drive checkpoint folder](https://drive.google.com/drive/folders/17AhweJCFLquS3k7MaTEyw6AoKj5BQPA6?usp=sharing)

## Code and Data Availability

This repository provides the source code, model definitions, and training scripts used in the manuscript.

The original datasets are not redistributed due to license restrictions. Users should download the datasets from their official or public sources and organize them according to the directory structure described above.

The required pretrained weights and teacher checkpoints are not included in this repository due to license and file-size considerations. The final trained checkpoints corresponding to the reported best results are provided through the Google Drive checkpoint folder listed above.

## Citation

If you use this code, please cite:

Wang, D. (2026). POSTER V2-ResNet18 with NA-MSAC (v1.0.1). Zenodo. https://doi.org/10.5281/zenodo.20985138

## Acknowledgements

We thank the open-source community for providing useful resources for facial expression recognition research.

## License

This repository is released under the MIT License. See [LICENSE](LICENSE) for details.
