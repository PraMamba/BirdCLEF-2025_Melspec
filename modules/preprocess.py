import pandas as pd
import numpy as np
from ast import literal_eval
import os
from sklearn.model_selection import train_test_split

def prepare_cfg(cfg, stage):
    # 根据阶段选择要使用的标签列集合
    if stage in ["pretrain_ce", "pretrain_bce"]:
        cfg.bird_cols = cfg.bird_cols_pretrain  # 预训练阶段使用的鸟类类别
    elif stage in ["train_ce", "train_bce", "finetune"]:
        cfg.bird_cols = cfg.bird_cols_train  # 训练/微调阶段使用的鸟类类别
    else:
        raise NotImplementedError  # 非法阶段，抛出错误

    # 配置不同阶段的音频片段时长和是否冻结模型参数
    if stage == 'finetune':
        cfg.DURATION = cfg.DURATION_FINETUNE  # 微调阶段的音频片段时长
        cfg.freeze = True  # 冻结部分模型参数
    elif stage in ["pretrain_ce", "pretrain_bce", "train_ce", "train_bce"]:
        cfg.DURATION = cfg.DURATION_TRAIN  # 预训练或正式训练阶段的时长
    else:
        raise NotImplementedError  # 其他情况不支持

    # 推理时的 batch size 根据推理音频片段数调整，保证不会太小
    cfg.test_batch_size = int(
        np.max([int(cfg.batch_size / (int(cfg.valid_duration) / cfg.DURATION)), 2])
    )

    # 一段完整音频可分成几个推理片段
    cfg.train_part = int(cfg.DURATION / cfg.infer_duration)

    return cfg


def preprocess(cfg, stage):
    # 定义一个 transform 函数，用于对原始波形做预处理（如增强）
    def transforms(audio):
        audio = cfg.np_audio_transforms(audio)  # NumPy 格式的增强（如噪声、扰动）
        audio = cfg.am_audio_transforms(audio, sample_rate=cfg.SR)  # 声学增强（如 time stretch）
        return audio.copy()

    # 加载原始数据（通常是一个 Pandas Pickle 文件）
    df = pd.read_pickle(cfg.train_data)

    # 将 secondary_labels（文本）转换为列表（字符串 -> list）
    df['secondary_labels'] = df['secondary_labels'].apply(lambda x: literal_eval(x))

    # 拼接文件路径（目录 + 文件名）
    df["path"] = cfg.train_dir + "/" + df["filename"]

    # 如果路径中某些文件不存在，提示警告
    if not df['path'].apply(lambda x: os.path.exists(x)).all():
        print('===========================================================')
        print('warning: missing audio files in cfg.train_dir')
        print('warning: only audios available will be used for training')
        print('===========================================================')

    # 过滤掉音频文件不存在的样本
    df = df[df['path'].apply(lambda x: os.path.exists(x))].reset_index(drop=True)

    # 初始化标签矩阵，shape = [样本数, 类别数]，全为 0
    labels = np.zeros(shape=(len(df), len(cfg.bird_cols)))
    df_labels = pd.DataFrame(labels, columns=cfg.bird_cols)

    # 初始化类别采样计数器，用于类别不平衡处理
    class_sample_count = {col: 0 for col in cfg.bird_cols}

    # 标记哪些样本应被用于训练
    include_in_train = []
    presence_type = []  # 标记样本属于前景、背景还是 soundscape

    # 遍历每个样本的主标签与辅助标签
    for i, (primary_label, secondary_labels) in enumerate(zip(df[cfg.primary_label_col].values, df[cfg.secondary_labels_col].values)):
        include = False
        # 如果主标签是 'soundscape'，则标记为 soundscape，否则默认为 background
        presence = 'background' if primary_label != 'soundscape' else 'soundscape'

        # 如果主标签在类别列表中，打标签为前景
        if primary_label in cfg.bird_cols:
            include = True
            presence = 'foreground'
            df_labels.loc[i, primary_label] = 1  # 主标签位置设为 1
            class_sample_count[primary_label] += 1  # 该类别计数 +1

        # 辅助标签处理（弱标签），赋值较小权重
        for secondary_label in secondary_labels:
            if secondary_label in cfg.bird_cols:
                include = True
                df_labels.loc[i, secondary_label] = cfg.secondary_label  # 弱标签标记
                class_sample_count[secondary_label] += cfg.secondary_label_weight

        presence_type.append(presence)  # 保存该样本的 presence 类型
        include_in_train.append(include)  # 标记该样本是否用于训练

    # 为每个样本打上 presence_type（foreground/background/soundscape）
    df['presence_type'] = presence_type

    # 只保留标签有效的训练样本（包含主标签或辅助标签）
    df = df[include_in_train].reset_index(drop=True)
    df_labels = df_labels[include_in_train].reset_index(drop=True)

    # 再次筛选：仅保留满足条件的短背景音或前景音样本（避免背景过长）
    df_labels = df_labels[((df['duration'] <= cfg.background_duration_thre) & (df['presence_type'] != 'foreground')) | (df['presence_type'] == 'foreground')].reset_index(drop=True)
    df = df[((df['duration'] <= cfg.background_duration_thre) & (df['presence_type'] != 'foreground')) | (df['presence_type'] == 'foreground')].reset_index(drop=True)

    # ========================
    # 类别不平衡权重计算
    # ========================
    all_primary_labels = df["primary_label"]  # 提取主标签
    sample_weights = (
        all_primary_labels.value_counts() /
        all_primary_labels.value_counts().sum()
    ) ** (cfg.class_exponent_weight)  # 加权指数可调节分布

    # 对权重进行归一化，使平均为 1
    sample_weights = sample_weights / sample_weights.mean()

    # 根据每个样本的主标签，赋值 sample weight
    df["weight"] = sample_weights[df["primary_label"].values].values

    # ========================
    # 划分训练集和验证集
    # ========================
    if "split" in cfg.seed:
        seed = cfg.seed["split"]  # 使用固定划分种子
    else:
        seed = cfg.seed[stage]  # 否则使用当前阶段的种子

    # 使用 sklearn 的 train_test_split 进行划分
    df_train, df_valid, df_labels_train, df_labels_valid = train_test_split(
        df, df_labels,
        test_size=cfg.test_size,
        shuffle=True,
        random_state=seed
    )

    # 重置索引，确保数据清晰
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    df_labels_train = df_labels_train.reset_index(drop=True)
    df_labels_valid = df_labels_valid.reset_index(drop=True)

    return df_train, df_valid, df_labels_train, df_labels_valid, transforms
