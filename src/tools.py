import os
from glob import glob
import numpy as np
import pandas as pd
from pathlib import Path

import torch


def train_test_split(feature_matrix, train_videos, test_videos):
    train_feature_matrix = {}
    test_feature_matrix = {}
    feature_dim = len(list(feature_matrix.values())[0])
    for video in train_videos:
        if video in feature_matrix:
            train_feature_matrix[video] = feature_matrix[video]
        else:
            train_feature_matrix[video] = np.array([np.nan] * feature_dim)

    for video in test_videos:
        if video in feature_matrix:
            test_feature_matrix[video] = feature_matrix[video]
        else:
            test_feature_matrix[video] = np.array([np.nan] * feature_dim)

    return train_feature_matrix, test_feature_matrix

def load_and_combine_feature_embeds(path):
    feature_files = glob(f'{path}/*.npy')
    feature_matrix = {}
    for file in feature_files:
        video_name = file.split('/')[-1].split('.')[0]
        feature_matrix[video_name] = np.load(file)
        print(f'Loaded {video_name} with shape {feature_matrix[video_name].shape}')

def strip_filetypes_conv_to_npy(feature_matrix):
    new_feature_matrix = {}
    for video_name, feature in feature_matrix.items():
        video_name = video_name.split('.')[0]
        if not isinstance(feature, np.ndarray):
            new_feature_matrix[video_name] = feature.numpy()
    return new_feature_matrix


#%% Load all feature embeddings into single matrix and split to train / test
# This script works on separate embeddings saved for each video

# path = '/Users/alonz/PycharmProjects/clip2brain/features/VJEPA/attentinve_pooler_out'
# path = '/Users/alonz/PycharmProjects/clip2brain/features/CLIP/original/final_clip_embd.npy'
embeddings_path = "/Users/alonz/PycharmProjects/Clip_FineTune/idans_embeddings/Audio/*"

train_csv_path = '/Users/alonz/Downloads/osfstorage-archive/annotations/train.csv'
test_csv_path = '/Users/alonz/Downloads/osfstorage-archive/annotations/test.csv'
train_csv = pd.read_csv(train_csv_path)
test_csv = pd.read_csv(test_csv_path)

train_videos = train_csv['video_name'].apply(lambda x: x.split('.')[0]).values
test_videos = test_csv['video_name'].apply(lambda x: x.split('.')[0]).values

all_train_embedings = {}
all_test_embedings = {}
for path in glob(embeddings_path):
    if 'full_mats' in path:
        continue

    model_name = Path(path).stem

    if os.path.isdir(path):
        feature_matrix = load_and_combine_feature_embeds(path)
    elif path.endswith('.npy'):
        feature_matrix = np.load(path, allow_pickle=True).item()
        path = Path(path).parent
    elif path.endswith('.pt'):
        feature_matrix = torch.load(path)
        path = Path(path).parent
    else:
        print(f'Path:{path} Unsupported file type')
        continue

    # strip filetypes and convert to numpy
    feature_matrix = strip_filetypes_conv_to_npy(feature_matrix)

    # Split into train & test if video feature exist, if not set to nan
    train_feature_matrix, test_feature_matrix = train_test_split(feature_matrix, train_videos, test_videos)

    # Add the train and test matrixes to the full matrix
    all_train_embedings = {video_name: np.r_[all_train_embedings.get(video_name, []), train_feature_matrix[video_name]]
                           for video_name in train_feature_matrix}
    all_test_embedings = {video_name: np.r_[all_test_embedings.get(video_name, []), test_feature_matrix[video_name]]
                          for video_name in test_feature_matrix}

    # Now lets save the train and test feature matrices
    results_dir = Path(path) / 'full_mats'
    (results_dir).mkdir(parents=True, exist_ok=True)
    np.save(f'{results_dir}/{model_name}_train_feature_matrix.npy', train_feature_matrix)
    np.save(f'{results_dir}/{model_name}_test_feature_matrix.npy', test_feature_matrix)
    print('Saved train and test feature matrices at:', results_dir)

    # save the full matrices of all layers
    np.save(f'{results_dir}/train_feature_all_layers_matrix.npy', all_train_embedings)
    np.save(f'{results_dir}/test_feature_all_layers_matrix.npy', all_test_embedings)

#%% Load the train and test feature matrices
# train_feature_matrix = np.load(f'{results_dir}/train_feature_matrix.npy', allow_pickle=True).item()
# df = pd.DataFrame(train_feature_matrix).T
#
# #%% Evaluate results
# path = "/Users/alonz/PycharmProjects/clip2brain/src/output/bootstrap/subj1/rsq_dist_mreserve_combined_whole_brain.npy"
# rsq_dist = np.load(path)