import pandas as pd
import numpy as np
from ast import literal_eval
import os
from sklearn.model_selection import train_test_split

def prepare_cfg(cfg,stage):
    if stage in ["pretrain_ce","pretrain_bce"]:
        cfg.bird_cols = cfg.bird_cols_pretrain
    elif stage in ["train_ce","train_bce","finetune"]:
        cfg.bird_cols = cfg.bird_cols_train
    else:
        raise NotImplementedError

    if stage == 'finetune':
        cfg.DURATION = cfg.DURATION_FINETUNE
        cfg.freeze = True
    elif stage in ["pretrain_ce","pretrain_bce","train_ce","train_bce"]:
        cfg.DURATION = cfg.DURATION_TRAIN
    else:
        raise NotImplementedError

    cfg.test_batch_size = int(
        np.max([int(cfg.batch_size / (int(cfg.valid_duration) / cfg.DURATION)), 2])
    )
    cfg.train_part = int(cfg.DURATION / cfg.infer_duration)
    return cfg


def preprocess(cfg, stage):
    def transforms(audio):
        audio = cfg.np_audio_transforms(audio)
        audio = cfg.am_audio_transforms(audio,sample_rate=cfg.SR)
        return audio.copy()

    # load dataframe
    df = pd.read_pickle(cfg.train_data)
    df['secondary_labels'] = df['secondary_labels'].apply(lambda x: literal_eval(x))
    df["path"] = cfg.train_dir + "/" + df["filename"]

    # drop duplicates
    # df_dup = get_duplicated_files()
    # df = df[~df["filename"].isin(df_dup["file2"])]


    # ensure all the train data is available
    if not df['path'].apply(lambda x:os.path.exists(x)).all():
        print('===========================================================')
        print('warning: missing audio files in cfg.train_dir')
        print('warning: only audios available will be used for training')
        print('===========================================================')
    df = df[df['path'].apply(lambda x:os.path.exists(x))].reset_index(drop=True)

    labels = np.zeros(shape=(len(df),len(cfg.bird_cols)))
    df_labels = pd.DataFrame(labels,columns=cfg.bird_cols)
    class_sample_count = {col:0 for col in cfg.bird_cols}
    include_in_train = []
    presence_type = []
    for i,(primary_label, secondary_labels) in enumerate(zip(df[cfg.primary_label_col].values,df[cfg.secondary_labels_col].values)):
        include = False
        presence = 'background' if primary_label!='soundscape' else 'soundscape'
        if primary_label in cfg.bird_cols:
            include = True
            presence = 'foreground'
            df_labels.loc[i,primary_label] = 1
            class_sample_count[primary_label] += 1
        for secondary_label in secondary_labels:
            if secondary_label in cfg.bird_cols:
                include = True
                df_labels.loc[i,secondary_label] = cfg.secondary_label
                class_sample_count[secondary_label] += cfg.secondary_label_weight
        presence_type.append(presence)
        include_in_train.append(include)

    df['presence_type'] = presence_type
    df = df[include_in_train].reset_index(drop=True)
    df_labels = df_labels[include_in_train].reset_index(drop=True)

    df_labels[((df['duration']<=cfg.background_duration_thre)&(df['presence_type']!='foreground'))|(df['presence_type']=='foreground')].reset_index(drop=True)
    df = df[((df['duration']<=cfg.background_duration_thre)&(df['presence_type']!='foreground'))|(df['presence_type']=='foreground')].reset_index(drop=True)

    # calculate class sampling weight
    all_primary_labels = df["primary_label"]
    sample_weights = (
        all_primary_labels.value_counts() / 
        all_primary_labels.value_counts().sum()
    )  ** (cfg.class_exponent_weight)
    sample_weights = sample_weights / sample_weights.mean()
    df["weight"] = sample_weights[df["primary_label"].values].values

    # train-valid split
    if "split" in cfg.seed:
        seed = cfg.seed["split"]
    else:
        seed = cfg.seed[stage]
    df_train, df_valid, df_labels_train, df_labels_valid = train_test_split(df, df_labels, test_size=cfg.test_size, shuffle=True, random_state=seed)
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    df_labels_train = df_labels_train.reset_index(drop=True)
    df_labels_valid = df_labels_valid.reset_index(drop=True)

    return df_train, df_valid, df_labels_train, df_labels_valid, transforms
