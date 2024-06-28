import glob
import pickle
from pathlib import Path

import numpy as np


## R2 score visualization
# first lets read all the R2 scores from the csv files
# then we will plot the R2 scores for each subject
# we will also plot the mean R2 score for each subject
# all plotes will be grouped by the brain region


def get_brain_region(file):
    roi_name = file.split('_')[-2]
    roi_side = file.split('_')[-1].split('.')[0].split('-')[-1]
    full_name = "_".join([roi_name, roi_side])
    return full_name

#%%
results_path = '/Users/alonz/PycharmProjects/clip2brain/src/output/encoding_results'
plots_path = '/Users/alonz/PycharmProjects/clip2brain/src/output/plots'
Path(plots_path).mkdir(parents=True, exist_ok=True)
results_path = Path(results_path)

r2_files = glob.glob(f'{results_path}/**/rsq*.p', recursive=True)
clip_only = False
if clip_only:
    r2_files = [file for file in r2_files if '_clip' in file]
    model_type = ['_'.join(Path(file).name.split('_')[1:5]) for file in r2_files]
else:
    r2_files_nonclip = [file for file in r2_files if '_clip' not in file]
    model_type = [Path(file).name.split('_')[1] for file in r2_files_nonclip]
    clip_r2_files = [file for file in r2_files if '_clip' in file]
    clip_model_type = ['_'.join(Path(file).name.split('_')[1:-2]) for file in clip_r2_files]

    r2_files = r2_files_nonclip + clip_r2_files
    model_type = model_type + clip_model_type
subj_names = [Path(file).parent.name for file in r2_files]
brain_rois = [get_brain_region(file) for file in r2_files]
brain_rois = [f'{roi}' for subj, roi in zip(subj_names, brain_rois)]
r2_scores = [pickle.load(open(file, 'rb')) for file in r2_files]
r2_scores_mean = [np.mean(score) for score in r2_scores]
r2_percentile = [np.percentile(score, 95) for score in r2_scores]

full_data = list(zip(subj_names, brain_rois, model_type, r2_scores, r2_scores_mean, r2_percentile))
#%% turn the list of tuples into a pandas dataframe
import pandas as pd

df = pd.DataFrame(full_data, columns=['subject', 'brain_region', 'model_type', 'r2_scores', 'r2_scores_mean', 'r2_95_percentile'])
# sort by brain region
df = df.sort_values(by='brain_region')
#%% plot the R2 scores for each subject grouped by brain region
import plotly.express as px
import plotly.offline as pyo

unq_models = np.unique(model_type)

for model in unq_models:
    model_df = df[df['model_type'] == model]
    # make the plot x axis be sorted by the brain region naming
    brain_regions = model_df['brain_region'].unique()
    brain_regions = sorted(brain_regions)
    model_df['brain_region'] = pd.Categorical(model_df['brain_region'], categories=brain_regions, ordered=True)
    # for boxplot of r2_score arrays, we need to melt the data
    df2 = model_df.explode('r2_scores')
    # now lets remove inf values
    df2 = df2[(df2['r2_scores'] != np.inf) & (df2['r2_scores'] != -np.inf)]
    # plot
    fig = px.box(df2,x='brain_region', y='r2_scores', color='subject', title=f'{model} R2 scores for each subject grouped by brain region')
    # limit y axis to 0-1
    fig.update_yaxes(range=[-1, 1])
    # pyo.iplot(fig)
    fig.show()
    # save to html
    fig.write_html(f'{plots_path}/r2_{model}_voxel_distribution.html')

#%% box plot of mean R2 scores for each subject grouped by brain region
# First take only the relevant models for plot
# With extra layers of clip
# relevant_models = ['mreserve', 'resnet50', 'clip_expert_pond_layer_5', 'clip_expert_pond_layer_12',
#                    'clip_expert_pond_final_layer',
#                    'clip_original_layer_5', 'clip_original_layer_12', 'clip_original_final_layer',
#                    'vjepa', ]
# Only final layers of clip
relevant_models = ['mreserve', 'resnet50', 'clip_expert_pond_final_layer',
                   'clip_original_final_layer', 'clip_expert_pond_all_layers_PCA20', 'clip_expert_pond_all_layers',
                   'vjepa']
df_filtered = df[df['model_type'].str.contains('|'.join(relevant_models))]
# rename the clip finetuned wandb name to "clip_finetuned"
wandb_name = 'expert_pond'
df_filtered['model_type'] = df_filtered['model_type'].apply(lambda x: x.replace(wandb_name, 'clip_finetuned'))

fig = px.box(df_filtered, x='brain_region', y='r2_scores_mean', color='model_type', title='Different models R2 scores grouped by brain region')
# pyo.iplot(fig)
fig.show()
# save to html
fig.write_html(f'{plots_path}/r2_model_vs_fmri_voxels.html')


#%% lets compare R2 peformance per voxel for each model
unq_models = np.unique(model_type)
model_1_name = unq_models[1]
model_2_name = unq_models[4]
model_1 = df[df['model_type'] == model_1_name]
model_2 = df[df['model_type'] == model_2_name]

model_1 = model_1.explode('r2_scores')
model_2 = model_2.explode('r2_scores')

# lets do all roi plots in one figure with subplots for each roi in matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

rows = 5
cols = len(df['brain_region'].unique())//rows + 1
fig, axs = plt.subplots(rows, cols, figsize=(25, 25))
for i in range(rows):
    for j in range(cols):
        if i*cols + j >= len(df['brain_region'].unique()):
            print(f'breaking at i: {i}, j: {j}')
            break
        roi = df['brain_region'].unique()[i*cols + j]
        print(f'roi: {roi}, i: {i}, j: {j}')
        roi_model_1 = model_1[model_1['brain_region'] == roi]
        roi_model_2 = model_2[model_2['brain_region'] == roi]
        roi_model_1 = roi_model_1.rename(columns={'r2_scores': model_1_name})
        roi_model_2 = roi_model_2.rename(columns={'r2_scores': model_2_name})
        roi_model_1 = roi_model_1.reset_index(drop=True)
        roi_model_2 = roi_model_2.reset_index(drop=True)
        roi_model_1['index'] = roi_model_1.index
        roi_model_2['index'] = roi_model_2.index

        roi_model_1 = roi_model_1[['index', model_1_name]]
        roi_model_2 = roi_model_2[['index', model_2_name]]
        roi_model = pd.concat([roi_model_1, roi_model_2], axis=1)
        roi_model = roi_model.dropna()
        roi_model = roi_model.reset_index(drop=True)
        sns.scatterplot(data=roi_model, x=model_1_name, y=model_2_name, ax=axs[i, j])
        axs[i, j].set_title(roi)
        axs[i, j].set_xlim(-0.1, 1)
        axs[i, j].set_ylim(-0.1, 1)

# plt.show()
# save the figure
plt.savefig(f'{plots_path}/r2_{model_1_name}_vs_{model_2_name}.jpg', bbox_inches='tight', dpi=150)
print(f'saved figure to {plots_path}/r2_{model_1_name}_vs_{model_2_name}.jpg')
#%%
df['brain_region'].unique()