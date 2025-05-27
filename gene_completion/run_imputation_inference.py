import subprocess

# Create comand to build exemplars for the dataset in dataset_config
command_list = ['python', 'imputation_inference.py', '--train=false', '--test=false', '--full_inference=true', '--sample_diffusion_steps=50']

# List of datasets in spared benchmark
all_datasets = ['erickson_human_prostate_cancer_p2', 
                'mirzazadeh_mouse_brain', 
                'vicari_mouse_brain']

print(f'Amount of datasets: {len(all_datasets)}')

for dataset in all_datasets:
    print(f'Running dataset: {dataset}')
    command_list.append(f'--dataset={dataset}')
    command_list.append(f'--autoencoder_ckpts_path=/home/dvegaa/ST_Diffusion/stDiff_Spared/transformer_models/{dataset}/autoencoder_model.ckpt')  
    command_list.append(f'--dit_ckpts_path=/media/SSD0/pcardenasg2/missing_completion_ckpts/{dataset}_12_1024_0.0001_noise.pt')
    subprocess.run(command_list)
    command_list.pop()