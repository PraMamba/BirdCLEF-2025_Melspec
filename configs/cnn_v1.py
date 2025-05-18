import copy
from configs.common import common_cfg
from modules.augmentations import (
    CustomCompose,
    CustomOneOf,
    NoiseInjection,
    GaussianNoise,
    PinkNoise,
    AddGaussianNoise,
    AddGaussianSNR,
    GaussianNoiseSNR,
    PinkNoiseSNR,
)
from audiomentations import Compose as amCompose
from audiomentations import OneOf as amOneOf
from audiomentations import AddBackgroundNoise, Gain, GainTransition, TimeStretch
import numpy as np

cfg = copy.deepcopy(common_cfg)

cfg.model_type = "cnn"
cfg.model_name = "efficientnet_b0"

cfg.secondary_label = 0.9
cfg.secondary_label_weight = 0.5


cfg.batch_size = 64
cfg.PRECISION = "bf16"
cfg.seed = {
    "pretrain_ce": 20231121,
    "pretrain_bce": 20230503,
    "train_ce": 202111210524,
    "train_bce": 20231911,
    "finetune": 20230523,
    "split": 0
}
cfg.DURATION_TRAIN = 20
cfg.DURATION_FINETUNE = 30
cfg.freeze = False
cfg.mixup = False
cfg.mixup2 = True
cfg.mixup_prob = 0.3
cfg.mixup_double = 1.0
cfg.mixup2_prob = 1.0
cfg.mix_beta = 5
cfg.mix_beta2 = 1
cfg.in_chans = 1
cfg.epochs = {
    #"pretrain_ce": 70,
    "pretrain_bce": 12,
    "train_ce": 32,
    "train_bce": 32,
    #"finetune": 10,
}
cfg.lr = {
    #"pretrain_ce": 3e-4,
    "pretrain_bce": 1e-3,
    "train_ce": 3e-4,
    "train_bce": 1e-3,
    #"finetune": 6e-4,
}
cfg.scheduler = "cosine"

cfg.model_ckpt = {
    #"pretrain_ce": None,
    "pretrain_bce": None,
    "train_ce": None,
    "train_bce": None,
    #"finetune": "outputs/cnn_v1/pytorch/train_bce/last.ckpt",
}

cfg.output_path = {
    #"pretrain_ce": "/data2/Mamba/Project/Kaggle/BirdCLEF-2025/cnn_v1/pytorch/pretrain_ce",
    "pretrain_bce": "/data2/Mamba/Project/Kaggle/BirdCLEF-2025/cnn_v1/pytorch/pretrain_bce",
    "train_ce": "/data2/Mamba/Project/Kaggle/BirdCLEF-2025/cnn_v7/pytorch/train_ce",
    "train_bce": "/data2/Mamba/Project/Kaggle/BirdCLEF-2025/cnn_v6/pytorch/train_bce",
    "finetune": "/data2/Mamba/Project/Kaggle/BirdCLEF-2025/cnn_v1/pytorch/finetune",
    "quantization": "/data2/Mamba/Project/Kaggle/BirdCLEF-2025/cnn_v1/openvino/quantization",
}

cfg.final_model_path = "/data2/Mamba/Project/Kaggle/BirdCLEF-2025/cnn_v7/pytorch/train_ce/epoch=30_step=11098_val_roc_auc=0.905_val_cmap_pad=0.745_val_ap=0.798.ckpt"
cfg.onnx_path = "/data2/Mamba/Project/Kaggle/BirdCLEF-2025/cnn_v7/onnx"
cfg.openvino_path = "/data2/Mamba/Project/Kaggle/BirdCLEF-2025/cnn_v7/openvino"

cfg.loss = {
    #"pretrain_ce": "ce",
    "pretrain_bce": "bce",
    "train_ce": "ce",
    "train_bce": "bce",
    #"finetune": "bce",
}

cfg.img_size = 256
cfg.n_mels = 128
cfg.n_fft = 2048
cfg.f_min = 0
cfg.f_max = 16000

cfg.valid_part = int(cfg.valid_duration / cfg.infer_duration)
cfg.hop_length = cfg.infer_duration * cfg.SR // (cfg.img_size - 1)

cfg.normal = 255

cfg.am_audio_transforms = amCompose([
    AddBackgroundNoise(cfg.birdclef2021_nocall + cfg.birdclef2020_nocall + cfg.freefield + cfg.warblrb + cfg.birdvox + cfg.rainforest + cfg.environment, min_snr_db=3.0, max_snr_db=30.0, p=0.5),
    Gain(min_gain_db=-12, max_gain_db=12, p=0.2),
])


cfg.np_audio_transforms = CustomCompose([
  CustomOneOf([
    NoiseInjection(p=0.5, max_noise_level=0.04),
    GaussianNoiseSNR(p=0.5),
    PinkNoiseSNR(p=0.5)
  ]),
])

cfg.input_shape = (120,cfg.in_chans,cfg.n_mels,cfg.img_size)
cfg.input_names = [ "x" ]
cfg.output_names = [ "y" ]
cfg.opset_version = 17

# quantization config
cfg.quant_batch_size = 32
cfg.quant_subset_size = 600
cfg.quant_fast_bias_correction = False
cfg.quant_ignore_layer_names = ['/head/Gemm/WithoutBiases', '/global_pool/Pow', '/global_pool/GlobalAveragePool', '/global_pool/Pow_1', '/global_pool/Clip']
cfg.quant_ovn_model_path = '/data2/Mamba/Project/Kaggle/BirdCLEF-2025/cnn_v1/openvino/cnn_v1.xml'

basic_cfg = cfg
