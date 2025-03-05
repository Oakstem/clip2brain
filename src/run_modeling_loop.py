import os
from glob import glob
from pathlib import Path

encodings_dir = '/Users/alonz/PycharmProjects/Clip_FineTune/idans_embeddings/Audio/full_mats/*'

for encoding_file in glob(encodings_dir):
    if not encoding_file.endswith('.npy') or 'test' in encoding_file:
        continue
    print(f"Running voxel regression for: {encoding_file}")
    output_model_name = Path(encoding_file).stem.split('_features')[0]
    command = (f"python run_modeling_simple.py --feature_matrix {encoding_file}"
               f" --model_name_to_save {output_model_name} --fix_testing")
    os.system(command)