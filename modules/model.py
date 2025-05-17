import torch
import torch.nn.functional as F
import torch.nn as nn
import timm
from torch.distributions import Beta
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import sklearn
import timm
import torchaudio
import os
from torch_audiomentations import Compose, PitchShift, Shift, OneOf, AddColoredNoise
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
)
import sklearn.metrics
from torch.cuda.amp import autocast
from transformers import get_linear_schedule_with_warmup
from modules.sumix import sumix, sumup
from modules.head import MultiHeadAttentionClassifier
from modules.metrics import padded_cmap, map_score, roc_score
from modules.make_delta import image_delta


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        if not isinstance(smoothing, torch.Tensor):
            self.smoothing = nn.Parameter(
                torch.tensor(smoothing), requires_grad=False)
        else:
            self.smoothing = nn.Parameter(
                smoothing, requires_grad=False)
        assert 0 <= self.smoothing.min() and self.smoothing.max() < 1
    
    @staticmethod
    def _smooth(targets:torch.Tensor, smoothing:torch.Tensor):
        with torch.no_grad():
            if smoothing.shape != targets.shape:
                _smoothing = smoothing.expand_as(targets)
            else:
                _smoothing = smoothing
            return targets * (1.0 - _smoothing) + 0.5 * _smoothing

    def forward(self, inputs, targets):
        targets = self._smooth(targets, self.smoothing)
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1. - pt)**self.gamma * bce_loss
        return focal_loss

    def __repr__(self):
        return f'FocalLoss(smoothing={self.smoothing})'


class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        """Mixup coefficient generator."""
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size):
        """Get mixup random coefficients.
        Args:
          batch_size: int
        Returns:
          mixup_lambdas: (batch_size,)
        """
        mixup_lambdas = []
        for n in range(0, batch_size, 2):
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1.0 - lam)

        return torch.from_numpy(np.array(mixup_lambdas, dtype=np.float32))


def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear",
    ).squeeze(1)

    return output


def gem(x: torch.Tensor, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (8, 4)).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"
        )


class AttBlockV2(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)


class Mixup(nn.Module):
    def __init__(self, mix_beta, mixup_prob, mixup_double):
        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.mixup_prob = mixup_prob
        self.mixup_double = mixup_double

    def forward(self, X, Y, weight=None):
        p = torch.rand((1,))[0]
        if p < self.mixup_prob:
            bs = X.shape[0]
            n_dims = len(X.shape)
            perm = torch.randperm(bs)

            p1 = torch.rand((1,))[0]
            if p1 < self.mixup_double:
                X = X + X[perm]
                Y = Y + Y[perm]
                Y = torch.clamp(Y, 0, 1)

                if weight is None:
                    return X, Y
                else:
                    weight = 0.5 * weight + 0.5 * weight[perm]
                    return X, Y, weight
            else:
                perm2 = torch.randperm(bs)
                X = X + X[perm] + X[perm2]
                Y = Y + Y[perm] + Y[perm2]
                Y = torch.clamp(Y, 0, 1)

                if weight is None:
                    return X, Y
                else:
                    weight = (
                        1 / 3 * weight + 1 / 3 * weight[perm] + 1 / 3 * weight[perm2]
                    )
                    return X, Y, weight
        else:
            if weight is None:
                return X, Y
            else:
                return X, Y, weight


class Mixup2(nn.Module):
    def __init__(self, mix_beta, mixup2_prob):
        super(Mixup2, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.mixup2_prob = mixup2_prob

    def forward(self, X, Y, weight=None):
        p = torch.rand((1,))[0]
        if p < self.mixup2_prob:
            bs = X.shape[0]
            n_dims = len(X.shape)
            perm = torch.randperm(bs)
            coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)

            if n_dims == 2:
                X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]
            elif n_dims == 3:
                X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
            else:
                X = (
                    coeffs.view(-1, 1, 1, 1) * X
                    + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]
                )
            Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]
            # Y = Y + Y[perm]
            # Y = torch.clamp(Y, 0, 1)

            if weight is None:
                return X, Y
            else:
                weight = coeffs.view(-1) * weight + (1 - coeffs.view(-1)) * weight[perm]
                return X, Y, weight
        else:
            if weight is None:
                return X, Y
            else:
                return X, Y, weight


class BirdClefModelBase(pl.LightningModule):
    def __init__(self, cfg, stage):
        super().__init__()
        self.num_classes = len(cfg.bird_cols)
        self.birds = [bird for bird in cfg.bird_cols]
        self.stage = stage
        self.cfg = cfg
        self.loss = cfg.loss[stage]
        self.lr = cfg.lr[stage]
        self.epochs = cfg.epochs[stage]
        self.in_chans = cfg.in_chans
        self.training_step_outputs = []
        self.validation_step_outputs = []

        if self.loss == "ce":
            self.loss_function = nn.CrossEntropyLoss(
                label_smoothing=self.cfg.label_smoothing, reduction="none"
            )
        elif self.loss == "bce":
            self.loss_function = nn.BCEWithLogitsLoss(reduction="none")
        elif self.loss == "focal":
            self.loss_function = FocalLoss()
        else:
            raise NotImplementedError
        self.mixup = Mixup(
            mix_beta=self.cfg.mix_beta,
            mixup_prob=self.cfg.mixup_prob,
            mixup_double=self.cfg.mixup_double,
        )
        self.mixup2 = Mixup2(
            mix_beta=self.cfg.mix_beta2, mixup2_prob=self.cfg.mixup2_prob
        )
        self.ema = None

        self.audio_transforms = Compose(
            [
                # AddColoredNoise(p=0.5),
                PitchShift(
                    min_transpose_semitones=-4,
                    max_transpose_semitones=4,
                    sample_rate=self.cfg.SR,
                    p=0.4,
                ),
                Shift(min_shift=-0.5, max_shift=0.5, p=0.4),
            ]
        )

        self.time_mask_transform = torchaudio.transforms.TimeMasking(
            time_mask_param=60, iid_masks=True, p=0.5
        )
        self.freq_mask_transform = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=24, iid_masks=True
        )

        # window = torch.hann_window(window_length = 2048, device = self.cfg.DEVICE)
        self.melspec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.cfg.SR,
            hop_length=self.cfg.hop_length,
            n_mels=self.cfg.n_mels,
            f_min=self.cfg.f_min,
            f_max=self.cfg.f_max,
            n_fft=self.cfg.n_fft,
            center=True,
            pad_mode="constant",
            norm="slaney",
            onesided=True,
            mel_scale="slaney",
        )

        if self.cfg.DEVICE.type == "cuda":
            self.melspec_transform = self.melspec_transform.cuda()
        else:
            self.melspec_transform = self.melspec_transform.cpu()

        self.db_transform = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=80
        )

    def lower_upper_freq(self, images):
        # images = images - images.min(1,keepdim=True).values.min(2,keepdim=True).values.min(3,keepdim=True).values
        r = torch.randint(self.cfg.n_mels // 2, self.cfg.n_mels, size=(1,))[0].item()
        x = (torch.rand(size=(1,))[0] / 2).item()
        pink_noise = (
            torch.from_numpy(
                np.array(
                    [
                        np.concatenate(
                            (
                                1 - np.arange(r) * x / r,
                                np.zeros(self.cfg.n_mels - r) - x + 1,
                            )
                        )
                    ]
                ).T
            )
            .float()
            .to(self.cfg.DEVICE)
        )
        images = images * pink_noise
        # images_max = images.max(1,keepdim=True).values.max(2,keepdim=True).values.max(3,keepdim=True).values
        # images = images/images_max
        return images

    @torch.cuda.amp.autocast(enabled=False)
    def transform_to_spec(self, audio):
        if self.training:
            audio = self.audio_transforms(audio, sample_rate=self.cfg.SR)

        # normalize
        # max_amplitude = torch.abs(audio).max()
        # audio = audio/max_amplitude

        spec = self.melspec_transform(audio)
        spec = self.db_transform(spec)
        if self.cfg.normal == 80:
            spec = (spec + 80) / 80
        elif self.cfg.normal == 255:
            spec = spec / 255
        else:
            raise NotImplementedError
        # spec = qtransform(audio)
        # spec = spec.unsqueeze(1)
        if self.training:
            spec = self.time_mask_transform(spec)
            if torch.rand(size=(1,))[0] < 0.5:
                spec = self.freq_mask_transform(spec)
            if torch.rand(size=(1,))[0] < 0.5:
                spec = self.lower_upper_freq(spec)
        return spec

    def set_ema(self, ema):
        self.ema = ema

    def on_before_zero_grad(self, *args, **kwargs):
        if self.ema is not None:
            if (self.global_step + 1) % 10 == 0:
                self.ema.update(self)

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)     

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    def configure_optimizers(self):
        model_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=self.cfg.weight_decay,
        )
        if self.cfg.scheduler == "cosine":
            interval = "epoch"

            lr_scheduler = CosineAnnealingWarmRestarts(
                model_optimizer, T_0=self.epochs, T_mult=1, eta_min=1e-6, last_epoch=-1
            )
        elif self.cfg.scheduler == "linear":
            total_steps = self.trainer.estimated_stepping_batches
            interval = "step"
            print(f"{total_steps=}")
            lr_scheduler = get_linear_schedule_with_warmup(
                model_optimizer,
                num_warmup_steps= int(0.1 * total_steps),
                num_training_steps=total_steps
            )
        else:
            raise NotImplementedError

        return {
            "optimizer": model_optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": interval,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    def freeze(self):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        self.freeze()
        y_pred, target, loss = self(batch)

        # self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_step=True)

        self.training_step_outputs.append({"loss": loss})
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        if self.ema is not None:
            y_pred, target, val_loss = self.ema.module(batch)
        else:
            y_pred, target, val_loss = self(batch)
        # print(y_pred)
        # self.log("val_loss", val_loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.validation_step_outputs.append({"val_loss": val_loss, "logits": y_pred, "targets": target})

        return val_loss

    def train_dataloader(self):
        return self._train_dataloader

    def validation_dataloader(self):
        return self._validation_dataloader

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        current_lr = self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[0]["lr"]
        if len(outputs):
            avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
            output_val = (
                torch.cat([x["logits"] for x in outputs], dim=0)
                .sigmoid()
                .cpu().float()
                .detach()
                .numpy()
            )
            target_val = (
                torch.cat([x["targets"] for x in outputs], dim=0).cpu().detach().numpy()
            )

            # print(output_val.shape)
            val_df = pd.DataFrame(target_val, columns=self.birds)
            pred_df = pd.DataFrame(output_val, columns=self.birds)

            if self.current_epoch > -1:
                avg_score = roc_score(val_df, pred_df)
                avg_score2 = padded_cmap(val_df, pred_df, padding_factor=3)
                avg_score3 = sklearn.metrics.label_ranking_average_precision_score(
                    target_val, output_val
                )
                self.log(
                    "val_loss",
                    avg_loss,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    prog_bar=True,
                )
                self.log(
                    "val_roc_auc",
                    avg_score,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    prog_bar=True,
                )
                self.log(
                    "val_cmap_pad",
                    avg_score2,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    prog_bar=True,
                )
                self.log(
                    "val_ap",
                    avg_score3,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    prog_bar=True,
                )
                self.log(
                    "lr",
                    current_lr,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    prog_bar=True,
                )

                print(f"epoch {self.current_epoch} validation loss {avg_loss}")
                print(f"epoch {self.current_epoch} validation ROCAUC score {avg_score}")
                print(f"epoch {self.current_epoch} validation C-MAP score pad 3 {avg_score2}")
                print(f"epoch {self.current_epoch} validation AP score {avg_score3}")
            else:
                self.log(
                    "val_loss",
                    avg_loss,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    prog_bar=True,
                )
                avg_score = 0
                print(f"epoch {self.current_epoch} validation loss {avg_loss}")
        else:
            avg_loss = 0
            avg_score = 0

        self.validation_step_outputs.clear()
        return {"val_loss": avg_loss, "val_cmap": avg_score}

    def on_train_epoch_end(self):
        outputs = self.training_step_outputs


class BirdClefTrainModelSED(BirdClefModelBase):
    def __init__(self, cfg, stage):
        super().__init__(cfg, stage)

        self.bn0 = nn.BatchNorm2d(cfg.n_mels)

        base_model = timm.create_model(
            cfg.model_name,
            pretrained=True,
            in_chans=self.in_chans,
            drop_path_rate=0.2,
            drop_rate=0.5,
        )
        # base_model.conv_stem.stride = (1,1)
        layers = list(base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)

        if "efficientnet" in self.cfg.model_name:
            in_features = base_model.classifier.in_features
        elif "eca" in self.cfg.model_name:
            in_features = base_model.head.fc.in_features
        elif "res" in self.cfg.model_name:
            in_features = base_model.fc.in_features
        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(in_features, self.num_classes, activation="sigmoid")

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_bn(self.bn0)

    def freeze(self):
        if self.stage == "finetune":
            self.encoder.eval()
            self.fc1.eval()
            self.bn0.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.fc1.parameters():
                param.requires_grad = False
            for param in self.bn0.parameters():
                param.requires_grad = False
        return

    def extract_feature(self,x):
        x = x.permute((0, 1, 3, 2))
        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        # if self.training:
        #    x = self.spec_augmenter(x)

        x = x.transpose(2, 3)
        # (batch_size, channels, freq, frames)
        x = self.encoder(x)

        # (batch_size, channels, frames)
        x = torch.mean(x, dim=2)

        # channel smoothing
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        return x, frames_num

    def forward(self, batch):
        x = batch[0]
        y = batch[1]
        weight = batch[2]
        if not self.training:
            bs, channel, parts = x.shape[0], x.shape[1], x.shape[2]
            x = x.reshape((bs * parts, channel, -1))

        if self.training:
            if self.cfg.mixup:
                x, y, weight = self.mixup(x, y, weight)
        #with autocast(enabled=False):
        x = self.transform_to_spec(x)
        if self.in_chans == 3:
            x = image_delta(x)

        if self.training:
            if self.cfg.mixup2:
                x, y, weight = self.mixup2(x, y, weight)

        x, frames_num = self.extract_feature(x)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        interpolate_ratio = frames_num // segmentwise_output.size(1)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output, interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)
        output_dict = {
            "framewise_output": framewise_output,
            "segmentwise_output": segmentwise_output,
            "logit": logit,
            "framewise_logit": framewise_logit,
            "clipwise_output": clipwise_output,
        }
        if not self.training:
            clipwise_output = clipwise_output.reshape((bs, parts, -1)).max(dim=1).values
            seg_num = segmentwise_logit.shape[1]
            fram_num = framewise_logit.shape[1]
            segmentwise_logit = (
                segmentwise_logit.reshape((bs, parts, seg_num, -1)).max(dim=1).values
            )
            framewise_logit = (
                framewise_logit.reshape((bs, parts, fram_num, -1)).max(dim=1).values
            )

        loss = 0.5 * self.loss_function(
            torch.logit(clipwise_output), y
        ) + 0.5 * self.loss_function(segmentwise_logit.max(1)[0], y)
        # loss = 0.5*self.loss_function(torch.logit(clipwise_output), y) + 0.5*self.loss_function(framewise_logit.max(1)[0], y)
        if self.loss == "ce":
            loss = (loss * weight) / weight.sum()
        elif self.loss == "bce":
            loss = loss.sum(dim=1) * weight
        else:
            raise NotImplementedError
        loss = loss.sum()

        return torch.logit(clipwise_output), y, loss

class BirdClefInferModelSED(BirdClefTrainModelSED):
    def forward(self,x,tta_delta=2):
        x,_ = self.extract_feature(x)
        time_att = torch.tanh(self.att_block.att(x))
        feat_time = x.size(-1)
        start = (
            feat_time / 2 - feat_time * (self.cfg.infer_duration / self.cfg.DURATION) / 2
        )
        end = start + feat_time * (self.cfg.infer_duration / self.cfg.DURATION)
        start = int(start)
        end = int(end)
        pred = self.attention_infer(start,end,x,time_att)

        start_minus = start-tta_delta
        end_minus=end-tta_delta
        pred_minus = self.attention_infer(start_minus,end_minus,x,time_att)

        start_plus = start+tta_delta
        end_plus=end+tta_delta
        pred_plus = self.attention_infer(start_plus,end_plus,x,time_att)

        pred = 0.5*pred + 0.25*pred_minus + 0.25*pred_plus
        return pred

    def attention_infer(self,start,end,x,time_att):
        feat = x[:, :, start:end]
        #att = torch.softmax(time_att[:, :, start:end], dim=-1)
        #             print(feat_time, start, end)
        #             print(att_a.sum(), att.sum(), time_att.shape)
        framewise_pred = torch.sigmoid(self.att_block.cla(feat))
        framewise_pred_max = framewise_pred.max(dim=2)[0]
        #clipwise_output = torch.sum(framewise_pred * att, dim=-1)
        #logits = torch.sum(
        #    self.att_block.cla(feat) * att,
        #    dim=-1,
        #)

        #return clipwise_output
        return framewise_pred_max

class BirdClefTrainModelCNN(BirdClefModelBase):
    def __init__(self, cfg, stage):
        super().__init__(cfg, stage)

        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=True,
            in_chans=self.in_chans,
            num_classes=0,
            global_pool="",
            drop_path_rate=0.2,
            drop_rate=0.5,
        )

        if "efficientnet" in self.cfg.model_name:
            backbone_out = self.backbone.num_features
        elif "eca" in self.cfg.model_name:
            backbone_out = self.backbone.num_features
        elif "res" in self.cfg.model_name:
            backbone_out = self.backbone.num_features
        else:
            backbone_out = self.backbone.num_features
        self.global_pool = GeM()

        self.head = nn.Linear(backbone_out, self.num_classes)

        self.big_dropout = nn.Dropout(p=0.5)

    def freeze(self):
        if self.stage == "finetune":
            self.backbone.eval()
            self.global_pool.eval()
            for param in self.global_pool.parameters():
                param.requires_grad = False
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, batch):
        x = batch[0].squeeze(1)
        y = batch[1]
        weight = batch[2]
        if self.training:
            self.factor = self.cfg.train_part
        else:
            self.factor = self.cfg.valid_part

        if not self.training:
            bs = x.shape[0]
            x = x.reshape((bs, -1))
        else:
            if self.cfg.mixup:
                x, y, weight = self.mixup(x, y, weight)
        bs, time = x.shape
        x = x.reshape(bs * self.factor, time // self.factor)
        #with autocast(enabled=False):
        x = self.transform_to_spec(x.unsqueeze(1))
        if self.in_chans == 3:
            x = image_delta(x)

        x = x.permute(0, 1, 3, 2)
        # x = x.permute(0, 2, 1)
        # x = x[:, None, :, :]

        if self.training:
            b, c, t, f = x.shape
            x = x.permute(0, 2, 1, 3)
            x = x.reshape(b // self.factor, self.factor * t, c, f)

            if self.cfg.mixup2:
                x, y, weight = self.mixup2(x, y, weight)
            # if self.cfg.mixup2:
            #    x, y, weight = self.mixup(x, y, weight)

            x = x.reshape(b, t, c, f)
            x = x.permute(0, 2, 1, 3)

        x = self.backbone(x)

        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(b // self.factor, self.factor * t, c, f)
        x = x.permute(0, 2, 1, 3)
        x = self.global_pool(x)
        x = x[:, :, 0, 0]
        if self.training:
            logits = torch.mean(
                torch.stack([self.head(self.big_dropout(x)) for _ in range(5)], dim=0),
                dim=0,
            )
        else:
            logits = self.head(x)

        loss = self.loss_function(logits, y)
        if self.loss == "ce":
            loss = (loss * weight) / weight.sum()
        elif self.loss == "bce" or self.loss == "focal":
            loss = loss.sum(dim=1) * weight
        else:
            raise NotImplementedError
        loss = loss.sum()

        return logits, y, loss

class BirdClefInferModelCNN(BirdClefTrainModelCNN):
    def forward(self,x):
        x = x.permute(0, 1, 3, 2)
        x = self.backbone(x)
        x = self.global_pool(x)
        x = x[:, :, 0, 0]
        logits = self.head(x)
        return logits


class BirdSimpleCNN(BirdClefModelBase):
    def __init__(self, cfg, stage):
        super().__init__(cfg, stage)

        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            in_chans=self.in_chans,
            num_classes=0,
            global_pool="",
            drop_path_rate=0.2,
            drop_rate=0.5,
        )

        backbone_out = self.backbone.num_features
        # self.head = nn.LazyLinear(self.num_classes)
        self.head = MultiHeadAttentionClassifier(backbone_out, num_classes=self.num_classes)

        self.big_dropout = nn.Dropout(p=0.5)

    def freeze(self):
        if self.stage == "finetune":
            self.backbone.eval()
            self.global_pool.eval()
            for param in self.global_pool.parameters():
                param.requires_grad = False
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, batch):
        x = batch[0].squeeze(1)
        y = batch[1]
        weight = batch[2]

        if not self.training:
            x = x[:, 0]
            bs = x.shape[0]
            x = x.reshape((bs, -1))
        else:
            if torch.rand(1) < 0.5:
                x, y, weight = sumup(x, y, weight)
            if torch.rand(1) < 0.5:
                x, y, weight = self.mixup(x, y, weight)
            if torch.rand(1) < 0.15:
                x, y, weight = self.mixup(x, y, weight)

        bs, time = x.shape

        x = self.transform_to_spec(x.unsqueeze(1))
        x = self.backbone(x)
        # x = F.adaptive_avg_pool2d(x,1)[:,:,0,0]

        logits = self.head(x)

        # calc loss
        loss = self.loss_function(logits, y)
        if self.loss == "ce":
            loss = (loss * weight) / weight.sum()
        elif self.loss == "bce" or self.loss == "focal":
            loss = loss.sum(dim=1) * weight
        else:
            raise NotImplementedError
        loss = loss.mean()

        return logits, y, loss

class BirdSimpleInferCNN(BirdSimpleCNN):
    def forward(self,x):
        x = self.backbone(x)
        # x = F.adaptive_avg_pool2d(x, 1)[:,:,0,0]
        logits = self.head(x)
        return logits


class ReshapeModel(BirdClefModelBase):
    def __init__(self, cfg, stage, reshape_factor=125, drop_rate=0.1, drop_path_rate=0.1,):
        super().__init__(cfg, stage)
        self.backbone = timm.create_model(
            cfg.model_name,
            in_chans=self.in_chans,
            pretrained=True,
            drop_rate=drop_rate,
            drop_path_rate = drop_path_rate,
            num_classes=0,
            global_pool="",
        )
        backbone_out = self.backbone.num_features
        self.reshape_factor = reshape_factor

        self.fc = nn.Linear(backbone_out, self.num_classes)
        self.downsample = torchaudio.transforms.Resample(
            cfg.SR,
            16000,
            lowpass_filter_width=16,
            rolloff=0.85,
            resampling_method="sinc_interp_kaiser",
            beta=8.555504641634386,
        )

    def freeze(self):
        pass

    def reshape_waveform(self, x):
        x = self.downsample(x)

        bs, d = x.shape
        x = x.view(bs, d//self.reshape_factor, self.reshape_factor)
        x = torch.unsqueeze(x, dim=1)
        x = torch.cat([x, x, x], dim=1)
        return x

    def forward(self, batch):
        x = batch[0] # bs t
        y = batch[1]
        weight = batch[2]

        if not self.training:
            x = x[:,0,0]
        else:
            x = x[:,0]
            if torch.rand(1) < 0.2:
                x, y, weight = sumup(x, y, weight)
    
        x = self.reshape_waveform(x)
        x = self.backbone(x) # (bs, ch, feat_size, feat_size)

        # pooling
        x = F.adaptive_avg_pool2d(x, 1)[:,:,0,0]

        logits = self.fc(x) # (bs, num_classes)

        loss = self.loss_function(logits, y)
        loss = loss.sum(dim=1) * weight
        loss = loss.mean()

        return logits, y, loss

class ReshapeInferModel(ReshapeModel):
    def forward(self,x):
        x = self.reshape_waveform(x)
        x = self.backbone(x) # (bs, ch, feat_size, feat_size)

        # pooling
        x = F.adaptive_avg_pool2d(x, 1)[:,:,0,0]

        logits = self.fc(x) # (bs, num_classes)
        return logits


def load_model(cfg,stage, train=True):
    if train:
        model_ckpt = cfg.model_ckpt[stage]
    else:
        model_ckpt = cfg.final_model_path

    if model_ckpt is not None:
        state_dict = torch.load(model_ckpt, map_location=cfg.DEVICE)['state_dict']
        print('loading model from checkpoint')
    else:
        state_dict = None

    # drop ema because did not used in the competition
    if state_dict is not None:
        keys = list(state_dict.keys())
        for k in keys:
            if 'ema' in k:
                state_dict.pop(k)

    if cfg.model_type=='sed':
        if train:
            model = BirdClefTrainModelSED(cfg, stage)
            if state_dict is not None:
                # pretrain to train
                state_dict.pop('att_block.att.weight')
                state_dict.pop('att_block.att.bias')
                state_dict.pop('att_block.cla.weight')
                state_dict.pop('att_block.cla.bias')
                model.load_state_dict(state_dict,strict=False)
        else:
            model = BirdClefInferModelSED(cfg, stage)
            model.load_state_dict(state_dict,strict=False)

    elif cfg.model_type=='cnn':
        if train:
            model = BirdClefTrainModelCNN(cfg, stage)
            if state_dict is not None:
                # pretrain to train
                if stage == 'train_ce':
                    state_dict.pop('head.weight')
                    state_dict.pop('head.bias')
                    model.load_state_dict(state_dict,strict=False)
                else:
                    state_dict.pop('head.weight')
                    state_dict.pop('head.bias')
                    state_dict.pop('melspec_transform.spectrogram.window')
                    state_dict.pop('melspec_transform.mel_scale.fb')
                    model.load_state_dict(state_dict,strict=False)
        else:
            model = BirdClefInferModelCNN(cfg, stage)
            model.load_state_dict(state_dict,strict=False)
    elif cfg.model_type=='simple_cnn':
        if train:
            model = BirdSimpleCNN(cfg, stage)
            if state_dict is not None:
                state_dict.pop('head.weight', None)
                state_dict.pop('head.bias', None)
                state_dict.pop('head.classifier.1.weight', None)
                state_dict.pop('head.classifier.1.bias', None)
                state_dict.pop('head.query', None)
                
                model.load_state_dict(state_dict,strict=False)
        else:
            model = BirdSimpleInferCNN(cfg, stage)
            model.load_state_dict(state_dict,strict=False)
    elif cfg.model_type=='reshape':
        if train:
            model = ReshapeModel(cfg, stage)
            if state_dict is not None:
                state_dict.pop('head.weight')
                state_dict.pop('head.bias')
        else:
            model = ReshapeInferModel(cfg, stage)
            model.load_state_dict(state_dict,strict=False)
    else:
        raise NotImplementedError

    if not train:
        model.eval()
    return model


