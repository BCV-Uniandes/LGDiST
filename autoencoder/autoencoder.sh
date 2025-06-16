CUDA_VISIBLE_DEVICES=3 python train_autoencoder_no_neighbors.py --dataset 10xgenomic_mouse_brain_sagittal_posterior --embedding_dim 512 --lr 0.0001
CUDA_VISIBLE_DEVICES=3 python train_autoencoder_no_neighbors.py --dataset mirzazadeh_mouse_bone --embedding_dim 512 --lr 0.00001
CUDA_VISIBLE_DEVICES=5 python train_autoencoder_no_neighbors.py --dataset villacampa_lung_organoid --embedding_dim 512 --lr 0.0001
CUDA_VISIBLE_DEVICES=5 python train_autoencoder_no_neighbors.py --dataset vicari_human_striatium --embedding_dim 512 --lr 0.00001
CUDA_VISIBLE_DEVICES=4 python train_autoencoder_no_neighbors.py --dataset abalo_human_squamous_cell_carcinoma --embedding_dim 512 --lr 0.00001
CUDA_VISIBLE_DEVICES=4 python train_autoencoder_no_neighbors.py --dataset villacampa_mouse_brain --embedding_dim 512 --lr 0.00001