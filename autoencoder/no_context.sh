CUDA_VISIBLE_DEVICES=0 python train_autoencoder_no_context_genes.py --dataset 10xgenomic_mouse_brain_sagittal_posterior --embedding_dim 512 --lr 0.0001 --input_dim 128
CUDA_VISIBLE_DEVICES=0 python train_autoencoder_no_context_genes.py --dataset mirzazadeh_mouse_bone --embedding_dim 512 --lr 0.00001 --input_dim 128
CUDA_VISIBLE_DEVICES=0 python train_autoencoder_no_context_genes.py --dataset abalo_human_squamous_cell_carcinoma --embedding_dim 512 --lr 0.00001 --input_dim 128
CUDA_VISIBLE_DEVICES=0 python train_autoencoder_no_context_genes.py --dataset villacampa_lung_organoid --embedding_dim 512 --lr 0.0001 --input_dim 128
CUDA_VISIBLE_DEVICES=0 python train_autoencoder_no_context_genes.py --dataset vicari_human_striatium --embedding_dim 512 --lr 0.00001 --input_dim 32 --latent_dim 32
CUDA_VISIBLE_DEVICES=0 python train_autoencoder_no_context_genes.py --dataset villacampa_mouse_brain --embedding_dim 512 --lr 0.00001 --input_dim 32 --latent_dim 32