import librosa as lb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from modules.utils import crop_or_pad
import torchaudio as ta

import pickle
with open('/home/guangp/BirdCLEF-2025_Melspec/notebooks/train_voice_data.pkl', 'rb') as f:
    VOICE_DATA = pickle.load(f)

class BirdTrainDataset(Dataset):

    # 初始化方法，传入多个参数控制数据读取行为
    def __init__(self, df, df_labels, cfg, res_type="kaiser_fast", resample=True, train=True, pseudo=None, transforms=None):
        self.cfg = cfg  # 配置对象，包含训练参数、音频参数等
        self.df = df  # 数据文件信息的 DataFrame，包含路径、标签等
        self.df_labels = df_labels  # 标签 DataFrame，存储 one-hot 或多标签数组
        self.sr = cfg.SR  # 目标采样率，例如 32000
        self.n_mels = cfg.n_mels  # 用于提取 Mel spectrogram 的滤波器个数
        self.fmin = cfg.f_min  # Mel 滤波器最小频率
        self.fmax = cfg.f_max  # Mel 滤波器最大频率

        self.train = train  # 指示当前是训练模式还是验证/测试模式
        self.duration = cfg.DURATION  # 每段音频的目标持续时间（单位：秒）

        self.audio_length = self.duration * self.sr  # 每段音频应包含的采样点数

        self.res_type = res_type  # 重采样方法，默认使用 librosa 的 "kaiser_fast"
        self.resample = resample  # 是否启用重采样功能

        self.pseudo = pseudo  # 伪标签（用于半监督学习），可以为 None 或 DataFrame

        self.transforms = transforms  # 数据增强处理，如加入噪声、时间拉伸等

    # 返回数据集中样本数量
    def __len__(self):
        return len(self.df)

    # 数据加载函数，负责读取、截取、增强并返回音频与标签
    def load_data(self, filepath, target, row):
        filename = row['filename']  # 获取文件名
        # 组合主标签与次标签，并过滤掉不在目标类别列表中的标签
        labels = [bird for bird in list(set([row[self.cfg.primary_label_col]] + row[self.cfg.secondary_labels_col])) if bird in self.cfg.bird_cols]
        # 只保留合法的次标签
        secondary_labels = [bird for bird in row[self.cfg.secondary_labels_col] if bird in self.cfg.bird_cols]
        duration = row['duration_sec']  # 获取音频实际长度（秒）
        presence = row['presence_type']  # 标签出现类型，常见值如 'foreground', 'background'

        # 默认情况下只取一段音频（无混合）
        self_mixup_part = 1
        # 如果音频是背景或存在辅助标签，就设置混合倍数（用于 self-mixup）
        if (presence != 'foreground') or (len(secondary_labels) > 0):
            self_mixup_part = int(self.cfg.background_duration_thre / self.duration)

        work_duration = self.duration * self_mixup_part  # 总处理音频时长
        work_audio_length = work_duration * self.sr  # 总处理音频采样点数

        # 最大偏移时间，确保随机裁剪不越界
        max_offset = np.max([0, duration - work_duration])
        
        
        # ✅ 根据 .pkl 的路径字段进行人声片段过滤
        full_path = row['path']
        voice_segments = VOICE_DATA.get(full_path, [])
        if isinstance(voice_segments, list) and len(voice_segments) > 0:
            human_voice_duration = sum([max(0.0, seg['end'] - seg['start']) for seg in voice_segments])
            if human_voice_duration > work_duration * 0.5:
                return None, None  # 超过 50%，跳过此样本

        # 仅用于计算推理片段数量（推理用不到这里）
        parts = int(duration // self.cfg.infer_duration) if duration % self.cfg.infer_duration == 0 else int(duration // self.cfg.infer_duration + 1)
        ends = [(p + 1) * self.cfg.infer_duration for p in range(parts)]

        # === 训练模式 ===
        if self.train:
            # 从音频文件中随机偏移位置读取 audio 片段
            offset = int(torch.rand((1,)).item() * max_offset * self.sr)
            audio_sample, orig_sr = ta.load(filepath, frame_offset=offset, num_frames=work_audio_length)  # 使用 torchaudio 加载
            audio_sample = audio_sample.mean(0).numpy()  # 转换为单通道（均值）

            # 如果原始采样率不同，则使用 librosa 重采样
            if (self.resample) and (orig_sr != self.sr):
                audio_sample = lb.resample(audio_sample, orig_sr, self.sr, res_type=self.res_type)

            # 如果音频长度不足，进行补齐
            if len(audio_sample) < work_audio_length:
                audio_sample = crop_or_pad(audio_sample, length=work_audio_length, is_train=self.train)

            audio_sample = audio_sample[:work_audio_length]  # 保证长度

            # 将长音频分割为多个段并加和（用于 self-mixup 模拟）
            audio_sample = audio_sample.reshape((self_mixup_part, -1))
            audio_sample = np.sum(audio_sample, axis=0)  # 多段求和模拟混合背景音

            # 应用可选的变换（如时域增强）
            if self.transforms is not None:
                audio_sample = self.transforms(audio_sample)

            # 最终检查长度是否一致，不一致则补齐
            if len(audio_sample) != self.audio_length:
                audio_sample = crop_or_pad(audio_sample, length=self.audio_length, is_train=self.train)

        # === 验证/测试模式 ===
        else:
            # 从头开始读取固定长度的音频
            audio, orig_sr = ta.load(filepath, frame_offset=0, num_frames=self.cfg.valid_duration * self.sr)
            audio = audio.mean(0).numpy()  # 转为单通道

            # 如有必要进行重采样
            if self.resample and orig_sr != self.sr:
                audio = lb.resample(audio, orig_sr, self.sr, res_type=self.res_type)

            # 将长音频按 audio_length 切割成多个片段
            audio_parts = int(np.ceil(len(audio) / self.audio_length))
            audio_sample = [audio[i * self.audio_length:(i + 1) * self.audio_length] for i in range(audio_parts)]

            # 最后一段不足则补齐
            if len(audio_sample[-1]) < self.audio_length:
                audio_sample[-1] = crop_or_pad(audio_sample[-1], length=self.audio_length, is_train=self.train)

            # 只保留前 valid_len 段，或者补 0 保持统一长度
            valid_len = int(self.cfg.valid_duration / self.duration)
            if len(audio_sample) > valid_len:
                audio_sample = audio_sample[0:valid_len]
            elif len(audio_sample) < valid_len:
                diff = valid_len - len(audio_sample)
                padding = [np.zeros(shape=(self.audio_length,))] * diff
                audio_sample += padding

            # 将 list 转换为 numpy 数组
            audio_sample = np.stack(audio_sample)

        # 增加一个维度并转换为 tensor，shape: (1, time) 或 (1, N, time)
        audio_sample = torch.tensor(audio_sample[np.newaxis]).float()

        # 获取目标标签向量，并二值化（验证阶段用）
        target = target.values
        if not self.train:
            target[target > 0] = 1

        return audio_sample, target

    # 获取一个样本
    def __getitem__(self, idx):
        row = self.df.iloc[idx]  # 获取音频信息行
        target = self.df_labels.iloc[idx]  # 获取标签信息行
        weight = row["weight"]  # 获取样本权重（用于加权损失等）

        # 加载音频样本及其标签
        audio, target = self.load_data(row["path"], target, row)
        
        # ==== ✅ 新增：如果数据被过滤（返回 None），则选下一个样本 ====
        if audio is None or target is None:
            return self.__getitem__((idx + 1) % len(self.df))  # 尝试下一个样本，循环不越界

        # 转换标签为 float tensor
        target = torch.tensor(target).float()

        # 返回三元组：(音频, 标签, 权重)
        return audio, target, weight


def get_train_dataloader(df_train, df_valid, df_labels_train, df_labels_valid, cfg, pseudo=None, transforms=None, num_workers=12):
  ds_train = BirdTrainDataset(
      df_train,
      df_labels_train,
      cfg,
      train=True,
      pseudo=pseudo,
      transforms = transforms,
  )
  ds_val = BirdTrainDataset(
      df_valid,
      df_labels_valid,
      cfg,
      train=False,
      pseudo=None,
      transforms=None,
  )
  sampler = RandomSampler(ds_train)

  dl_train = DataLoader(ds_train, batch_size=cfg.batch_size , sampler=sampler, num_workers=num_workers, pin_memory=True)
  dl_val = DataLoader(ds_val, batch_size=cfg.test_batch_size, num_workers=num_workers, pin_memory=True)
  return dl_train, dl_val, ds_train, ds_val
