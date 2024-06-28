import os
from pathlib import Path

import clip
import numpy as np
import pandas as pd
import torch
from PIL.Image import Image
from tqdm import tqdm


def extract_last_layer_feature(model, preprocess, data, modality="vision", device='mps'):
    print("Number of Images/Captions: {}".format(len(data)))

    if modality == "vision":
        all_features = {}

        for key, val in tqdm(data.items()):
            image = preprocess(Image.open(val['path'])).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)

            all_features[key] = image_features.cpu().data.numpy()


        return all_features
    elif modality == "text":  # this is subject specific
        # extract text feature of image titles
        all_text_features = {}
        for key, val in tqdm(data.items()):
            with torch.no_grad():
                embs = []
                for caption in [key]:
                    text = clip.tokenize(caption).to(device)
                    embs.append(model.encode_text(text).cpu().data.numpy())

                mean_emb = np.mean(np.array(embs), axis=0).squeeze()

                all_text_features[key] = mean_emb

        print(len(all_text_features))
        return all_text_features

def build_img_path_dict(moments_img_path):
    img_path_dict = {}
    for root, dirs, files in os.walk(moments_img_path):
        for file in files:
            if file.endswith('.png'):
                if file.startswith('.'):
                    continue
                img_path_dict[Path(file).stem] = {}
                img_path_dict[Path(file).stem]['path'] = os.path.join(root, file)
                img_path_dict[Path(file).stem]['category'] = Path(img_path_dict[Path(file).stem]['path']).parent.stem

    return img_path_dict

def separate_train_test_feature_results(features_results_path=None, specific_key=None):
    if features_results_path is None:
        features_results_path = "/Users/alonz/PycharmProjects/clip2brain/features/CLIP/others/laion2b_clip.npy"

    features = np.load(features_results_path, allow_pickle=True).item()
    if specific_key is not None:
        features = {key: val[specific_key] for key, val in features.items()}

        for key, feature in features.items():
            # if the features are in a list, cat all list variables and mean over them
            if isinstance(feature, list):
                feature = torch.cat([val.unsqueeze(0) for val in feature], dim=0)
                feature = torch.mean(feature, dim=0)
                features[key] = feature

    feature_output_dir = Path(features_results_path).parent

    train_video_names_path = '/Users/alonz/PycharmProjects/pSTS_DB/psts_db/datasets/momments/fmri/annotations/train.csv'
    test_video_names_path = '/Users/alonz/PycharmProjects/pSTS_DB/psts_db/datasets/momments/fmri/annotations/test.csv'

    train_names = pd.read_csv(train_video_names_path)['video_name'].apply(lambda x: x.split('.')[0]).values
    test_names = pd.read_csv(test_video_names_path)['video_name'].apply(lambda x: x.split('.')[0]).values

    train_features = {key: np.nan for key in train_names}
    test_features = {key: np.nan for key in test_names}

    train_features.update({key: val for key, val in features.items() if key in train_names})
    test_features.update({key: val for key, val in features.items() if key in test_names})

    np.save(f'{feature_output_dir}/train_feature_matrix.npy', train_features)
    np.save(f'{feature_output_dir}/test_feature_matrix.npy', test_features)
    print(f"Train features saved at {feature_output_dir}/train_feature_matrix.npy")



results_path = '/Users/alonz/PycharmProjects/clip2brain/features/HHI/ClipCapHHI/full_w_prefix/momments_embd_results.npy'
separate_train_test_feature_results(results_path, specific_key='last_hidden_states')
# Available keys from HHI model: 'visual_proj_embedding', 'video_embedding', 'embedding'