import argparse
import numpy as np
import pandas as pd
import scipy
from pathlib import Path
from glob import glob
from scipy.stats import zscore
from encodingmodel.encoding_model import fit_encoding_model, bootstrap_test
from sklearn.decomposition import PCA
from featureprep.feature_prep import (
    get_preloaded_features,
    extract_feature_with_image_order,
)
from util.data_util import load_subset_trials

def run(
    fm,
    br,
    full_data,
    brain_roi_name,
    model_name,
    test,
    fix_testing,
    cv,
    saving_dir,
):
    if test:
        print("Running Bootstrap Test")
        bootstrap_test(
            fm,
            br,
            model_name=model_name,
            subj=args.subj,
            saving_dir=saving_dir,
        )

    else:
        print("Fitting Encoding Models")
        fit_encoding_model(
            fm,
            br,
            full_data=full_data,
            brain_roi_name=brain_roi_name,
            model_name=model_name,
            subj=args.subj,
            fix_testing=fix_testing,
            cv=cv,
            saving=True,
            saving_dir=saving_dir,
        )

def load_embedding_features(feature_mat_path):
    if feature_mat_path is None:
        return None
    feature_mat_unordered = np.load(feature_mat_path, allow_pickle=True).item()
    if list(feature_mat_unordered.values())[0].ndim > 1:
        feature_mat_unordered.update({k: v.reshape(-1) for k, v in feature_mat_unordered.items() if not np.isnan(v).any()})
    feature_mat_unordered = pd.DataFrame(feature_mat_unordered).T.values
    return feature_mat_unordered


def scale(train_X, test_X):
    # first remove activations that have 0 variance
    variance = np.nanstd(train_X, axis=0).squeeze()
    train_X = train_X[:, np.invert(np.isclose(variance, 0.))]
    test_X = test_X[:, np.invert(np.isclose(variance, 0.))]
    print(f'{np.sum(np.isclose(variance, 0.))} channels had a variance of zero')

    # now scale the data
    mean = np.nanmean(train_X, axis=0).squeeze()
    variance = np.nanstd(train_X, axis=0).squeeze()
    train_X = (train_X - mean) / variance
    test_X = (test_X - mean) / variance
    return train_X, test_X

def project_to_pca(data, nb_componentes=5, pca=None):
    if pca is None:
        pca = PCA(n_components=nb_componentes, svd_solver="full")
        pca.fit(data)
    PCs = pca.components_
    data = pca.transform(data)
    # from sklearn.preprocessing import Normalizer, StandardScaler
    # # Create a normalizer object
    # scaler = StandardScaler()  # l2 norm results in a norm of 1
    # # Fit and transform the data
    # X_scaled = scaler.fit_transform(data)
    # pca = PCA(n_components=nb_componentes, svd_solver="full")
    # pca.fit(X_scaled)
    return data, pca

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Please specify features to model from and parameters of the encoding model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="convnet",
        nargs="+",
        help="input the names of the features.",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="input name of the layer. e.g. input_layer1",
    )
    parser.add_argument("--test", action="store_true", help="Run bootstrap testing.")
    parser.add_argument(
        "--subj",
        type=str,
        default='01',
        help="Specify which subject to build model on. Currently it supports subject 1, 2, 5, 7",
    )
    parser.add_argument(
        "--fix_testing",
        action="store_true",
        help="Use fixed sampling for training and testing (for model performance comparison purpose)",
    )
    parser.add_argument(
        "--cv", action="store_true", default=False, help="run cross-validation."
    )
    parser.add_argument(
        "--get_features_only",
        action="store_true",
        default=False,
        help="only generate and save the feature matrix but not running the encoding models (for preloaded features)",
    )
    parser.add_argument(
        "--feature_pca",
        action="store_true",
        default=False,
        help="choose whether to apply pca on embedding data",
    )
    parser.add_argument(
        "--pca_nb",
        default=20,
        type=int,
        help="number of components for PCA",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Specify the path to the intermediate output directory that contains voxels masks etc",
    )

    parser.add_argument(
        "--saving_dir",
        type=str,
        default="output",
        help="saving dir for the encoding model results. Default is the same as the output but could be somewhere else.",
    )

    parser.add_argument(
        "--features_dir",
        type=str,
        default="features",
        help="Specify the path to the features directory",
    )
    parser.add_argument(
        "--feature_matrix",
        type=str,
        default=None,
        help="Specify the path to the feature matrix (should be a numpy array)",
    )
    parser.add_argument(
        "--test_feature_matrix",
        type=str,
        default=None,
        help="Specify the path to the feature matrix (should be a numpy array)",
    )
    parser.add_argument(
        "--feature_order",
        type=str,
        default=None,
        help="Specify the path to the ordering of the feature matrix (should be a numpy array)",
    )
    parser.add_argument(
        "--model_name_to_save",
        type=str,
        default=None,
        help="Specify a name to save the performance with",
    )
    parser.add_argument(
        "--subset_data",
        type=str,
        default=None,
        help="specify a category to subset training and testing data",
    )

    parser.add_argument(
        "--zscore",
        action="store_true",
        default=False,
        help="specify whether to apply zscoring to the brain data",
    )

    args = parser.parse_args()
    print(args)
    args.stage = 'train' if not args.test else 'test'
    args.test_feature_matrix = args.test_feature_matrix if args.test_feature_matrix is not None else\
        args.feature_matrix.replace("train", "test")

    subj_array = ['01', '02', '03', '04']
    for subj in subj_array:
        args.subj = subj
        all_brain_files = glob(f"/Users/alonz/PycharmProjects/pSTS_DB/psts_db/fmri/voxel_roi/{args.stage}/sub-{args.subj}/*.pkl")

        for brain_path in all_brain_files:
            print("Loading brain data from: " + Path(brain_path).name)
            brain_roi_name = brain_path.split("T1w_roi-")[-1].split('_roi')[0]
            test_brain_path = brain_path.replace("train", "test")

            # Load brain data
            br_data = np.load(brain_path, allow_pickle=True)
            if isinstance(br_data, dict):
                br_data = pd.DataFrame(br_data).T.values
            test_br_data = np.load(test_brain_path, allow_pickle=True)
            if isinstance(test_br_data, dict):
                test_br_data = pd.DataFrame(test_br_data).T.values
            # deal with voxels that are zeros in runs and therefore cause nan values in zscoring
            # only happens in some subjects (e.g. subj5)
            bad_cols = (np.sum(br_data, axis=0) == 0)
            br_data = br_data[:, ~bad_cols]
            test_br_data = test_br_data[:, ~bad_cols]
            # test_br_data = test_br_data[:, ~(np.sum(test_br_data, axis=0) == 0)]


            if args.zscore:
                print("Zscoring brain data...")
                br_data = zscore(br_data)
                test_br_data = zscore(test_br_data)
                print("NaNs? Finite?:")
                print(np.any(np.isnan(br_data)))
                print(np.all(np.isfinite(br_data)))
            # try:
            #     non_zero_mask = np.load(
            #         "%s/voxels_masks/subj%d/nonzero_voxels_subj%02d.npy"
            #         % (args.output_dir, args.subj, args.subj)
            #     )
            #     print("Masking zero voxels...")
            #     br_data = br_data[:, non_zero_mask]
            # except FileNotFoundError:
            #     pass

            # dead with trials that are nan because subjects has never seen the images
            trial_mask = np.sum(np.isnan(br_data), axis=1) <= 0
            br_data = br_data[trial_mask, :]

            test_trial_mask = np.sum(np.isnan(test_br_data), axis=1) <= 0
            test_br_data = test_br_data[test_trial_mask, :]

            print("NaNs? Finite?:")
            print(np.any(np.isnan(br_data)))
            print(np.all(np.isfinite(br_data)))
            print("Brain response size is: " + str(br_data.shape))

            try:
                stimulus_list = np.load(
                    "%s/coco_ID_of_repeats_subj%s.npy" % (args.output_dir, args.subj)
                )
            except FileNotFoundError:
                stimulus_list = pd.read_csv(
                    f'/Users/alonz/PycharmProjects/pSTS_DB/psts_db/datasets/momments/fmri/annotations/{args.stage}.csv').values[:,0]

            # Load feature spaces
            if args.feature_matrix is not None:  # for general design matrix input
                if args.test:
                    args.feature_matrix = args.feature_matrix.replace("train", "test")
                feature_mat_unordered = load_embedding_features(args.feature_matrix)
                test_feature_mat = load_embedding_features(args.test_feature_matrix)
                model_name_to_save = args.model_name_to_save
                # image_order = np.load(args.image_order)
                # feature_mat = extract_feature_with_image_order(
                #     stimulus_list, feature_mat_unordered, image_order)
                feature_mat = feature_mat_unordered
            else:
                if args.layer is not None:
                    model_name_to_save = args.model[0] + "_" + args.layer
                else:
                    model_name_to_save = args.model[0]

                feature_mat = get_preloaded_features(
                    args.subj,
                    stimulus_list,
                    args.model[0],
                    layer=args.layer,
                    features_dir=args.features_dir,
                )

                if len(args.model) > 1:
                    for model in args.model[1:]:
                        more_feature = get_preloaded_features(
                            args.subj, stimulus_list, model, features_dir=args.features_dir
                        )
                        feature_mat = np.hstack((feature_mat, more_feature))

                        model_name_to_save += "_" + model

            trial_mask2 = np.sum(np.isnan(feature_mat), axis=1) <= 0
            feature_mat = feature_mat[(trial_mask & trial_mask2), :]
            br_data = br_data[(trial_mask & trial_mask2), :]


            test_trial_mask2 = np.sum(np.isnan(test_feature_mat), axis=1) <= 0
            test_feature_mat = test_feature_mat[(test_trial_mask & test_trial_mask2), :]
            test_br_data = test_br_data[(test_trial_mask & test_trial_mask2), :]

            if args.feature_pca:
                feature_mat, test_feature_mat = scale(feature_mat, test_feature_mat)
                feature_mat, pca = project_to_pca(feature_mat, nb_componentes=args.pca_nb)
                test_feature_mat, pca = project_to_pca(test_feature_mat, nb_componentes=args.pca_nb, pca=pca)


            full_data = {
                "X_train": feature_mat,
                "Y_train": br_data,
                "X_test": test_feature_mat,
                "Y_test": test_br_data,
            }

            print("=======================")
            print("Running ridge encoding model on :")
            print(model_name_to_save)

            print("Feature size is: " + str(feature_mat.shape))
            print("=======================")

            if not args.get_features_only:
                run(
                    feature_mat,
                    br_data,
                    full_data=full_data,
                    brain_roi_name=brain_roi_name,
                    model_name=model_name_to_save,
                    test=args.test,
                    fix_testing=args.fix_testing,
                    cv=args.cv,
                    saving_dir=args.saving_dir,
                )
