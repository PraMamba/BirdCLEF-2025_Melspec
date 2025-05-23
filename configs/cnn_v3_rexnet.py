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
from audiomentations import AddBackgroundNoise, Gain, GainTransition, TimeStretch, Shift, Normalize
import numpy as np

cfg = copy.deepcopy(common_cfg)

cfg.model_type = "cnn"
cfg.model_name = "rexnet_150"

cfg.secondary_label = 0.9
cfg.secondary_label_weight = 0.5
cfg.use_2024_additional_cleaned = True
cfg.class_exponent_weight = -0.5
cfg.test_size = 0.1


cfg.batch_size = 96
cfg.PRECISION = 16
cfg.seed = {
    "pretrain_ce": 111,
    "pretrain_bce": 132,
    "train_ce": 332,
    "train_bce": 991,
    "finetune": 2,
}
cfg.DURATION_TRAIN = 15
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
    "pretrain_bce": 30,
    "train_ce": 15,
    "train_bce": 20,
    #"finetune": 10,
}
cfg.lr = {
    #"pretrain_ce": 3e-4,
    "pretrain_bce": 1e-3,
    "train_ce": 3e-4,
    "train_bce": 1e-3,
    #"finetune": 6e-4,
}
cfg.scheduler = "linear"

cfg.model_ckpt = {
    #"pretrain_ce": None,
    "pretrain_bce": None,
    "train_ce": None,
    "train_bce": "outputs/cnn_v3_rexnet/pytorch/pretrain_bce/last.ckpt",
    #"finetune": "outputs/cnn_v1/pytorch/train_bce/last.ckpt",
}

cfg.output_path = {
    #"pretrain_ce": "outputs/cnn_v1/pytorch/pretrain_ce",
    "pretrain_bce": "outputs/cnn_v3_rexnet/pytorch/pretrain_bce",
    "train_ce": "outputs/cnn_v3_rexnet/pytorch/train_ce",
    "train_bce": "outputs/cnn_v3_rexnet/pytorch/train_bce",
    "finetune": "outputs/cnn_v3_rexnet/pytorch/finetune",
    "quantization": "outputs/cnn_v3_rexnet/openvino/quantization",
}

cfg.final_model_path = "outputs/cnn_v3_rexnet/pytorch/train_bce/last-v2.ckpt"
cfg.onnx_path = "outputs/cnn_v3_rexnet/onnx"
cfg.openvino_path = "outputs/cnn_v3_rexnet/openvino"

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

cfg.normal = 80

cfg.am_audio_transforms = amCompose([
    AddBackgroundNoise(cfg.birdclef2021_nocall + cfg.birdclef2020_nocall + cfg.freefield + cfg.warblrb + cfg.birdvox + cfg.rainforest + cfg.environment, min_snr_in_db=3.0,max_snr_in_db=30.0,p=0.5),
    Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.2),

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
cfg.quant_fast_bias_correction = True
cfg.quant_ignore_layer_names = ['/head/Gemm/WithoutBiases', '/global_pool/Pow', '/global_pool/AveragePool', '/global_pool/Pow_1']
cfg.quant_ovn_model_path = 'outputs/cnn_v3_rexnet/openvino/cnn_v3_rexnet.xml'

basic_cfg = cfg
