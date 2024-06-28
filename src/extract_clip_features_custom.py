import os
from pathlib import Path

import pandas as pd
from PIL import Image

import clip
import numpy as np
import torch
from tqdm import tqdm
from util.custom_utils import *

#%% Model & Data params
device = "mps" if torch.cuda.is_available() else "cpu"
model_name = 'ViT-B/32'

moments_img_path = '/Volumes/swagbox/moments_images/Moments_in_Time_Raw'
embd_output_dir = '/Users/alonz/PycharmProjects/clip2brain/features/CLIP/vitB_32/full_mats'
train_video_names_path = '/Users/alonz/PycharmProjects/pSTS_DB/psts_db/datasets/momments/fmri/annotations/train.csv'
test_video_names_path = '/Users/alonz/PycharmProjects/pSTS_DB/psts_db/datasets/momments/fmri/annotations/test.csv'
#%% Load clip model & data

# model, preprocess = clip.load(model_name, device=device)
imgs_path_dict = build_img_path_dict(moments_img_path)
all_categories = list(set([val['category'] for val in imgs_path_dict.values()]))

train_names = pd.read_csv(train_video_names_path)['video_name'].apply(lambda x: x.split('.')[0]).values
test_names = pd.read_csv(test_video_names_path)['video_name'].apply(lambda x: x.split('.')[0]).values

#%% Extract features
# all_features = extract_last_layer_feature(model, preprocess, imgs_path_dict, modality='vision', device=device)
#%% Save features
train_feature_keys = [key for key, val in imgs_path_dict.items() if key in train_names]
test_feature_keys = [key for key, val in imgs_path_dict.items() if key in test_names]

train_features = {key: np.nan for key in train_names}
test_features = {key: np.nan for key in test_names}
train_features.update({key: val.reshape(-1) for key, val in all_features.items() if key in train_feature_keys})
test_features.update({key: val.reshape(-1) for key, val in all_features.items() if key in test_feature_keys})

np.save(f'{embd_output_dir}/train_feature_matrix.npy', train_features)
np.save(f'{embd_output_dir}/test_feature_matrix.npy', test_features)

#%%
test = np.load('/Users/alonz/PycharmProjects/clip2brain/features/VJEPA/attentinve_pooler_out/full_mats/test_feature_matrix.npy', allow_pickle=True).item()