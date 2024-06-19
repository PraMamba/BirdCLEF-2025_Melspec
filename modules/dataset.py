import librosa as lb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from modules.utils import crop_or_pad
import torchaudio as ta

class BirdTrainDataset(Dataset):

    def __init__(self, df, df_labels, cfg, res_type="kaiser_fast",resample=True, train = True, pseudo=None, transforms=None):
        self.cfg =cfg
        self.df = df
        self.df_labels = df_labels
        self.sr = cfg.SR
        self.n_mels = cfg.n_mels
        self.fmin = cfg.f_min
        self.fmax = cfg.f_max

        self.train = train
        self.duration = cfg.DURATION

        self.audio_length = self.duration*self.sr

        self.res_type = res_type
        self.resample = resample

        # self.df["weight"] = np.clip(self.df["rating"] / self.df["rating"].max(), 0.1, 1.0)
        self.pseudo = pseudo

        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def load_data(self, filepath,target,row):
        filename = row['filename']
        labels = [bird for bird in list(set([row[self.cfg.primary_label_col]] + row[self.cfg.secondary_labels_col])) if bird in self.cfg.bird_cols]
        secondary_labels = [bird for bird in row[self.cfg.secondary_labels_col] if bird in self.cfg.bird_cols]
        duration = row['duration_sec']
        presence = row['presence_type']

        # self mixup
        self_mixup_part = 1
        if (presence!='foreground') | (len(secondary_labels)>0):
          self_mixup_part = int(self.cfg.background_duration_thre/self.duration)
        work_duration = self.duration * self_mixup_part
        work_audio_length = work_duration*self.sr

        max_offset = np.max([0,duration-work_duration])
        parts = int(duration//self.cfg.infer_duration) if duration%self.cfg.infer_duration==0 else int(duration//self.cfg.infer_duration + 1)
        ends = [(p+1)*self.cfg.infer_duration for p in range(parts)]

        if self.train:
            offset = torch.rand((1,)).numpy()[0] * max_offset * self.sr
            audio_sample, orig_sr = ta.load(filepath, frame_offset=offset, num_frames=work_audio_length)
            audio_sample = audio_sample.mean(0).numpy()
            # audio_sample, orig_sr = lb.load(filepath, sr=None, mono=True,offset=offset, duration=work_duration)
            if (self.resample)&(orig_sr != self.sr):
                audio_sample = lb.resample(audio_sample, orig_sr, self.sr, res_type=self.res_type)

            if len(audio_sample) < work_audio_length:
                audio_sample = crop_or_pad(audio_sample, length=work_audio_length,is_train=self.train)

            audio_sample = audio_sample[:work_audio_length]
            audio_sample = audio_sample.reshape((self_mixup_part,-1))
            audio_sample = np.sum(audio_sample,axis=0)

            if self.transforms is not None:
              audio_sample = self.transforms(audio_sample)

            if len(audio_sample) != self.audio_length:
                audio_sample = crop_or_pad(audio_sample, length=self.audio_length,is_train=self.train)


        else:
            audio, orig_sr = ta.load(filepath, frame_offset=0, num_frames=self.cfg.valid_duration*self.sr)
            audio = audio.mean(0).numpy()
            # audio, orig_sr = lb.load(filepath, sr=None, mono=True,offset=0,duration=self.cfg.valid_duration)
            if self.resample and orig_sr != self.sr:
                audio = lb.resample(audio, orig_sr, self.sr, res_type=self.res_type)

            audio_parts = int(np.ceil(len(audio)/self.audio_length))
            audio_sample = [audio[i*self.audio_length:(i+1)*self.audio_length] for i in range(audio_parts)]

            if len(audio_sample[-1])<self.audio_length:
              audio_sample[-1] = crop_or_pad(audio_sample[-1],length=self.audio_length,is_train=self.train)

            valid_len = int(self.cfg.valid_duration/self.duration)
            if len(audio_sample)> valid_len:
              audio_sample = audio_sample[0:valid_len]
            elif len(audio_sample)<valid_len:
              diff = valid_len-len(audio_sample)
              padding = [np.zeros(shape=(self.audio_length,))] * diff
              audio_sample += padding

            audio_sample = np.stack(audio_sample)

        audio_sample = torch.tensor(audio_sample[np.newaxis]).float()

        target = target.values
        if not self.train:
          target[target>0] = 1
        return audio_sample,target

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        target = self.df_labels.iloc[idx]

        weight = row["weight"]
        audio, target = self.load_data(row["path"],target,row)
        target = torch.tensor(target).float()
        return audio, target , weight

def get_train_dataloader(df_train, df_valid, df_labels_train, df_labels_valid, cfg,pseudo=None,transforms=None, num_workers=12):
  ds_train = BirdTrainDataset(
      df_train,
      df_labels_train,
      cfg,
      train = True,
      pseudo = pseudo,
      transforms = transforms,
  )
  ds_val = BirdTrainDataset(
      df_valid,
      df_labels_valid,
      cfg,
      train = False,
      pseudo = None,
      transforms=None,
  )
  sampler = RandomSampler(ds_train)

  dl_train = DataLoader(ds_train, batch_size=cfg.batch_size , sampler=sampler, num_workers = num_workers, pin_memory=True)
  dl_val = DataLoader(ds_val, batch_size=cfg.test_batch_size, num_workers = num_workers, pin_memory=True)
  return dl_train, dl_val, ds_train, ds_val
