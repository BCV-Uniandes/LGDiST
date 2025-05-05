import subprocess
import torch

# Create comand to build exemplars for the dataset in dataset_config
command_list = ['python', 'imputation_inference.py', '--train=false', '--test=false', '--full_inference=true', '--sample_diffusion_steps=50']

# List of datasets in spared benchmark
all_datasets = ['10xgenomic_human_brain', '10xgenomic_human_breast_cancer', '10xgenomic_mouse_brain_coronal', 
                '10xgenomic_mouse_brain_sagittal_anterior', '10xgenomic_mouse_brain_sagittal_posterior', 
                'abalo_human_squamous_cell_carcinoma', 'erickson_human_prostate_cancer_p1', 'erickson_human_prostate_cancer_p2', 
                'fan_mouse_brain_coronal', 'fan_mouse_olfatory_bulb', 'mirzazadeh_human_colon_p1', 'mirzazadeh_human_colon_p2', 
                'mirzazadeh_human_pediatric_brain_tumor_p1', 'mirzazadeh_human_pediatric_brain_tumor_p2', 
                'mirzazadeh_human_prostate_cancer', 'mirzazadeh_human_small_intestine', 'mirzazadeh_mouse_bone', 
                'mirzazadeh_mouse_brain_p1', 'mirzazadeh_mouse_brain_p2', 'mirzazadeh_mouse_brain', 'parigi_mouse_intestine', 
                'vicari_human_striatium', 'vicari_mouse_brain', 'villacampa_kidney_organoid', 'villacampa_lung_organoid', 'villacampa_mouse_brain']

print(f'Amount of datasets: {len(all_datasets)}')

for dataset in all_datasets:
    print(f'Running dataset: {dataset}')
    command_list.append(f'--dataset={dataset}')
    command_list.append(f'--autoencoder_ckpts_path=/home/dvegaa/ST_diffusion/transformer_models/{dataset}/autoencoder_model.ckpt')  
    command_list.append(f'--dit_ckpts_path=/home/dvegaa/ST_diffusion/Experiments/{dataset}/{dataset}_12_1024_0.0001_noise.pt')
    subprocess.run(command_list)
    command_list.pop()