import argparse
import copy
from pathlib import Path

import pandas as pd
import numpy as np
from torch._C import Value
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict

# from sklearn.decomposition import PCA

import torch
from torchvision import transforms
import torch.nn as nn

from PIL import Image
import blip_models
import open_clip
from util.custom_utils import build_img_path_dict

device = "cuda" if torch.cuda.is_available() else "mps"
print(device)


def extract_last_layer_feature(model, dataset="YFCC"):
    # all_images_paths = ["%s/%s.jpg" % (stimuli_dir, id) for id in img_ds]
    all_images_paths = [val["path"] for val in img_ds.values()]
    print("Number of Images: {}".format(len(all_images_paths)))

    if dataset == "YFCC":
        preprocess = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                lambda x: x.convert("RGB"),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        ckpt_path = "models/%s_large_25ep.pt" % model
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = OrderedDict()
        for k, v in ckpt["state_dict"].items():
            state_dict[k.replace("module.", "")] = v

        old_args = ckpt["args"]
        print("=> creating model: {}".format(old_args.model))
        model = getattr(blip_models, old_args.model)(
            rand_embed=False,
            ssl_mlp_dim=old_args.ssl_mlp_dim,
            ssl_emb_dim=old_args.ssl_emb_dim,
        )

        model.load_state_dict(state_dict, strict=True)

    elif "IC" in dataset:
        import clip

        model, _ = clip.load("RN50", device=device)
        # preprocess = transforms.Compose(
        #     [
        #         transforms.RandomResizedCrop(224),
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        #         )
        #     ]
        # )
        # adjust param according to paper
        preprocess = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.481, 0.458, 0.408], std=[0.269, 0.261, 0.276]
                ),
            ]
        )

        ckpt_path = "models/%s.pt" % dataset
        print("Loading checkpoint from: " + ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = OrderedDict()
        for k, v in ckpt["state_dict"].items():
            state_dict[k.replace("module.", "")] = v

        model.load_state_dict(state_dict, strict=True)

    elif dataset == "laion400m":
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion400m_e31"
        )

    elif dataset == "laion2b":
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_e16"
        )
        # model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34_b79k')

    model.to(device)
    model.eval()

    all_features = {}

    for key, val in tqdm(img_ds.items()):
        p = val["path"]
        image = preprocess(Image.open(p)).unsqueeze(0).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image)
            # print(image_features.shape)

        all_features[key] = image_features.cpu().data.numpy()
    all_features = np.array(all_features)
    Path(f"{feature_output_dir}").mkdir(parents=True, exist_ok=True)
    np.save("%s/%s_%s.npy" % (feature_output_dir, dataset, args.model), all_features)
    return all_features


def extract_visual_transformer_feature(model_name, dataset):
    import torchextractor as tx
    import clip
    from util.coco_utils import load_captions

    ckpt_path = "models/%s_large_25ep.pt" % model_name
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = OrderedDict()
    for k, v in ckpt["state_dict"].items():
        state_dict[k.replace("module.", "")] = v

    old_args = ckpt["args"]
    print("=> creating model: {}".format(old_args.model))
    model = getattr(blip_models, old_args.model)(
        rand_embed=False,
        ssl_mlp_dim=old_args.ssl_mlp_dim,
        ssl_emb_dim=old_args.ssl_emb_dim,
    )

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    preprocess = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            lambda x: x.convert("RGB"),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    layer = "visual.blocks.11.drop_path2"  # by default n-1 layers
    LOI_transformer_vision = [layer]
    model = tx.Extractor(model, LOI_transformer_vision)
    feature_dd = {}
    for key, val in tqdm(img_ds.items()):
        with torch.no_grad():
            image_path = val["path"]
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            captions = [val['category']]
            text = clip.tokenize(captions).to(device)

            _, features = model(image, text)
            # print(features[layer].shape)

            feature_dd[key] = features[layer].squeeze().cpu().data.numpy().flatten()

    # feature_list = np.array(feature_list)
    feature_values = np.array(list(feature_dd.values()))
    # Path(f"{feature_output_dir}").mkdir(parents=True, exist_ok=True)
    from sklearn.decomposition import PCA

    print("Running PCA...")
    pca = PCA(n_components=64, whiten=True, svd_solver="full")

    fp = pca.fit_transform(feature_values)
    print(fp.shape)
    feature_dict = {k: v for k, v in zip(feature_dd.keys(), fp)}
    Path(f"{feature_output_dir}").mkdir(parents=True, exist_ok=True)
    np.save("%s/YFCC_%s_layer_n-1.npy" % (feature_output_dir, model_name), feature_dict)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", default=0, type=int)
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="features",
    )
    parser.add_argument(
        "--project_output_dir",
        type=str,
        default="output",
    )
    parser.add_argument("--dataset", type=str, default="YFCC")

    parser.add_argument("--model", type=str, default="clip")
    parser.add_argument("--layer", type=str, default="final")

    args = parser.parse_args()

    import configparser

    config = configparser.ConfigParser()
    config.read("../config.cfg")
    stimuli_dir = config["DATA"]["StimuliDir"]

    print(args)

    if args.layer == "final":
        extract = extract_last_layer_feature
        layer_tag = ""
    else:
        extract = extract_visual_transformer_feature
        layer_tag = "_layer_n-1"

    # if args.subj == 0:
    # for s in range(8):
        # print("Extracting subj%01d" % (s + 1))
    feature_output_dir = "%s/" % (args.feature_dir)
    try:
        np.load(
            "%s/%s_%s%s.npy"
            % (feature_output_dir, args.dataset, args.model, layer_tag)
        )
    except FileNotFoundError:
        # img_ds = np.load(
        #     "%s/coco_ID_of_repeats_subj%02d.npy"
        #     % (args.project_output_dir, (s + 1))
        # )
        img_ds = build_img_path_dict(stimuli_dir)
        extract(args.model, args.dataset)

    # else:
    #     feature_output_dir = "%s/subj%01d" % (args.feature_dir, args.subj)
    #     try:
    #         np.load(
    #             "%s/%s_%s%s.npy"
    #             % (feature_output_dir, args.dataset, args.model, layer_tag)
    #         )
    #     except FileNotFoundError:
    #         img_ds = np.load(
    #             "%s/coco_ID_of_repeats_subj%02d.npy"
    #             % (args.project_output_dir, args.subj)
    #         )
    #
    #         extract(args.model, args.dataset)
