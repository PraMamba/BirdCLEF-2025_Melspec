This codebase contains the melspec models part of the [4th place solution](https://www.kaggle.com/competitions/birdclef-2024/discussion/511845) in the BirdCLEF 2024 competition.

These codes are a fork of honglihang's BirdCLEF2023 2nd place solution. The original code can be found [here](https://github.com/LIHANG-HONG/birdclef2023-2nd-place-solution).

You can find the code for our other model, the Raw-signal reshape model, in [this](https://github.com/tamotamo17/BirdCLEF2024-4th-place-solution) repository.

## Hardware
* Ubuntu 20.04 LTS
* GPU: 1x NVIDIA RTX 4090 24GB
* CPU: 13th Gen Intel(R) Core(TM) i9-13900KF, 24vCPU
* Memory: 64GB
* CUDA: 12.1

## Requirements
We use [Rye](https://rye.astral.sh/) for the Python environment setup. After installing Rye, up the environment with

```
rye sync
```

## Data Preparation

### Pretraining
Download the [Birdclef2021](https://www.kaggle.com/c/birdclef-2021/data) dataset, [Birdclef2022](https://www.kaggle.com/c/birdclef-2022/data) dataset, and [Birdclef2023](https://www.kaggle.com/competitions/birdclef-2023/data) dataset from kaggle and extract the contents to the `./inputs/pretrain/` folder.

### Training
Download the [Birdclef2024](https://www.kaggle.com/competitions/birdclef-2024/data) dataset and extract the contents to the `./inputs/birdclef-2024` folder.

### Background Noise
Download the audios from [here](https://www.kaggle.com/datasets/honglihang/background-noise) and put all the audios to `./inputs/background_noise`.

## Training

Open `notebooks/train.ipynb` and set `model_name` and `stage` in the `Configs` cell as follows:

#### 2021-2nd CNN Model (seresnext26ts)
```python
model_name = "cnn_v1"
stage = "train_bce"
```

#### 2021-2nd CNN Model (rexnet_150)
```python
model_name = "cnn_v3_rexnet"
stage = "train_bce"
```

#### Simple CNN Model (inception_next_nano)
```python
model_name = "simple_cnn_v1"
stage = "train_bce"
```

Then, execute the notebook. After training, the last checkpoint (model weights) will be saved to the folder `./outputs/MODEL_NAME/pytorch/STAGE`.

## Convert Model

### OpenVINO

Run

```sh
python3 convert.py --model_name MODEL_NAME --stage train_bce --openvino
```

The converted model will be saved in the `./outputs/MODEL_NAME/openvino` folder.

### Quantization(Option)

Open `notebooks/quantize.ipynb` and set the `model_name` and `stage` to be the same as the training. Then, run the notebook.

The quantized model will be placed in folder `./outputs/MODEL_NAME/openvino/quantization`.

## Inference
Inference is published in a kaggle kernel [here](https://www.kaggle.com/code/ajobseeker/b24-final?scriptVersionId=182393504).
