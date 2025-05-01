import os
import numpy as np
import warnings
import torch
import anndata as ad
import argparse
import numpy as np
import torch
import squidpy as sq
import anndata as ad
from tqdm import tqdm
import copy

warnings.filterwarnings('ignore')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Auxiliary function to use booleans in parser
str2bool = lambda x: (str(x).lower() == 'true')
str2intlist = lambda x: [int(i) for i in x.split(',')]
str2floatlist = lambda x: [float(i) for i in x.split(',')]
str2h_list = lambda x: [str2intlist(i) for i in x.split('//')[1:]]

def get_main_parser():
    parser = argparse.ArgumentParser(description='Code for Diffusion Imputation Model')
    # Dataset parameters #####################################################################################################################################################################
    parser.add_argument('--dataset',                type=str,         default='villacampa_lung_organoid',           help='Dataset to use.')
    parser.add_argument('--prediction_layer',       type=str,         default='c_d_deltas',                         help='The prediction layer from the dataset to use.')
    parser.add_argument("--normalize_encoder",      type=str,         default="none",                               help='If the normalization is done in range [-1, 1] (-1-1) or is done in range [0, 1] (0-1) or is none')
    # Train parameters #######################################################################################################################################################################
    parser.add_argument('--lr',                     type=float,       default=0.000001,                             help='lr to use')
    parser.add_argument('--num_epochs',             type=int,         default=5000,                                 help='Number of training epochs')
    parser.add_argument('--batch_size',             type=int,         default=128,                                  help='The batch size to train model')
    # Autoencoder parameters #######################################################################################################################################################################
    parser.add_argument('--num_layers',             type=int,         default=4,                                    help='Number of layers in the autoencoder')
    parser.add_argument('--num_heads',              type=int,         default=1,                                    help='Number of heads in the transformer encoder')
    parser.add_argument("--embedding_dim",          type=int,         default=512,                                  help='Embedding dimensions in the eutoencoder')
    parser.add_argument('--input_dim',              type=int,         default=1024,                                 help='Input dimension of the autoencoder')
    parser.add_argument("--latent_dim",             type=int,         default=128,                                  help='Latent dimension in the autoencoder')
    parser.add_argument("--output_dim",             type=int,         default=1024,                                 help='Ouput dimension of the autoencoder')
    # Data masking parameters ################################################################################################################################################################
    parser.add_argument('--num_hops',                       type=int,           default=1,                          help="Amount of graph hops to consider for context during imputation")
    parser.add_argument('--seed',                           type=int,               default=1202,                            help='Seed to control initialization')
    return parser

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True   

def normalize_to_minus_one_to_one(X, X_max, X_min):
    # Apply the normalization formula to -1-1
    X_norm = 2 * (X - X_min) / (X_max - X_min) - 1
    return X_norm

def denormalize_from_minus_one_to_one(X_norm, X_max, X_min):
    # Apply the denormalization formula 
    X_denorm = ((X_norm + 1) / 2) * (X_max - X_min) + X_min
    return X_denorm
        
def get_spatial_neighbors(adata: ad.AnnData, num_neighs: int, hex_geometry: bool) -> dict:
    """
    This function computes a neighbors dictionary for an AnnData object. The neighbors are computed according to topological distances over
    a graph defined by the hex_geometry connectivity. The neighbors dictionary is a dictionary where the keys are the indexes of the observations
    and the values are lists of the indexes of the neighbors of each observation. The neighbors include the observation itself and are found
    inside a n_hops neighborhood of the observation.

    Args:
        adata (ad.AnnData): the AnnData object to process. Importantly it is only from a single slide. Can not be a collection of slides.
        n_hops (int): the size of the neighborhood to take into account to compute the neighbors.
        hex_geometry (bool): whether the graph is hexagonal or not. If True, then the graph is hexagonal. If False, then the graph is a grid. Only
                                used to compute the spatial neighbors and only true for visium datasets.

    Returns:
        dict: The neighbors dictionary. The keys are the indexes of the observations and the values are lists of the indexes of the neighbors of each observation.
    """
    
    # Compute spatial_neighbors
    if hex_geometry:
        sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=num_neighs) # Hexagonal visium case
        #sc.pp.neighbors(adata, n_neighbors=6, knn=True)
    # Get the adjacency matrix (binary matrix of shape spots x spots)
    adj_matrix = adata.obsp['spatial_connectivities']
    
    # Define power matrix
    power_matrix = adj_matrix.copy() #(spots x spots)
    # Define the output matrix
    output_matrix = adj_matrix.copy() #(spots x spots)

    # Zero out the diagonal
    output_matrix.setdiag(0)  #(spots x spots) Apply 0 diagonal to avoid "auto-paths"
    # Threshold the matrix to 0 and 1
    output_matrix = output_matrix.astype(bool).astype(int)

    # Define neighbors dict
    neighbors_dict_index = {}
    # Iterate through the rows of the output matrix
    for i in range(output_matrix.shape[0]):
        # Get the non-zero elements of the row (non zero means a neighbour)
        non_zero_elements = output_matrix[:,i].nonzero()[0]
        # Add the neighbors to the neighbors dicts. NOTE: the first index is the query obs
        #Key: int number (id of each spot) -> Value: list of spots ids
        neighbors_dict_index[i] = [i] + list(non_zero_elements)
    
    # Return the neighbors dict
    return neighbors_dict_index


def build_neighborhood_from_hops(spatial_neighbors, expression_mtx, idx, autoencoder_model, args):
    # Get nn indexes for the n_hop required
    nn_index_list = spatial_neighbors[idx] #Obtain the ids of the spots that are neigbors of idx
    # Normalize data
    
    min_exp = expression_mtx.min()
    max_exp = expression_mtx.max()
    if args.normalize_encoder == "1-1":
        expression_mtx = normalize_to_minus_one_to_one(expression_mtx, max_exp, min_exp)
    #Index the expression matrix (X processed) and obtain the neccesary data
    exp_matrix = expression_mtx[nn_index_list].type('torch.FloatTensor')
    #breakpoint()
    if autoencoder_model != None:
        exp_matrix = autoencoder_model.encoder(exp_matrix.to("cuda"))
    return exp_matrix, max_exp, min_exp #shape (n_neigbors, n_genes)

def get_neigbors_dataset(adata, prediction_layer, num_hops, autoencoder_model, args):
    """
    This function recives the name of a dataset and pred_layer. Returns a list of len = number of spots, each position of the list is an array 
    (n_neigbors + 1, n_genes) that has the information about the neigbors of the corresponding spot.
    """
    all_neighbors_info = {}
    max_min_info = {}
    #Dataset all info
    dataset = adata
    #get dataset splits
    splits = dataset.obs["split"].unique().tolist()
    #get num neighs
    num_neighs = 0
    for hop in range(1, num_hops+1):
        num_neighs += 6*hop
    #iterate over split adata
    for split in splits:
        split_neighbors_info = []
        adata = dataset[dataset.obs["split"]==split]
        #Get dict with all the neigbors info for each spot in the dataset
        spatial_neighbors = get_spatial_neighbors(adata, num_neighs=num_neighs, hex_geometry=True)
        #Expression matrix (already applied post-processing)
        expression_mtx = torch.tensor(adata.layers[prediction_layer]) 
        for idx in tqdm(spatial_neighbors.keys()):
            # Obtainb expression matrix just wiht the neigbors of the corresponding spot
            neigbors_exp_matrix, max_enc, min_enc = build_neighborhood_from_hops(spatial_neighbors, expression_mtx, idx, autoencoder_model, args)
            split_neighbors_info.append(neigbors_exp_matrix)
            
        #append split neighbors info into the complete list
        all_neighbors_info[split] = split_neighbors_info
        max_min_info[split] = [max_enc, min_enc]

    return all_neighbors_info, max_min_info
    
def get_mask_extreme_completion(adata, mask, genes):
    mask_extreme_completion = copy.deepcopy(mask)
    imp_values = adata.layers["mask"] #True en los valores reales y False en los valores imputados
    mask_extreme_completion[imp_values] = 1
    mask_extreme_completion[:,:,1:] = 0
    genes = np.array(genes)[:,np.newaxis]
    mask_extreme_completion = mask_extreme_completion*genes
    for i in range(0, mask_extreme_completion.shape[0]):
        idx_mask = np.where(mask_extreme_completion[0,:,0]==1)[0]
        idx_genes = np.where(genes==1)[0]
        bool_mask = np.isin(idx_mask, idx_genes).all()
        assert bool_mask, "error en mascara"
        
    return mask_extreme_completion


def sort_adatas(adata, adata_128):
    adata.var.reset_index(drop=True, inplace=True)
    adata_128.var.reset_index(drop=True, inplace=True)

    # The previous index (gene_ids) is now a column; create a numeric index
    adata.var.index = range(adata.var.shape[0])
    adata_128.var.index = range(adata_128.var.shape[0])

    adata_sorted = adata.copy()
    adata_128_sorted = adata_128.copy()
    
    # Sort .var (genes) by index
    adata_sorted.var["original_index"] = adata_sorted.var.index
    adata_sorted.var = adata_sorted.var.sort_values(by="gene_ids").reset_index(drop=True)

    adata_128_sorted.var["original_index"] = adata_128_sorted.var.index
    adata_128_sorted.var = adata_128_sorted.var.sort_values(by="gene_ids").reset_index(drop=True)

    #Get indices
    sorted_indices = adata_sorted.var["original_index"].to_numpy()
    sorted_indices = [int(idx) for idx in sorted_indices]
    
    sorted_indices_128 = adata_128_sorted.var["original_index"].to_numpy()
    sorted_indices_128 = [int(idx) for idx in sorted_indices_128]

    # Reorder all layers to match the new gene order
    for layer in adata.layers.keys():
        adata_sorted.layers[layer] = adata.layers[layer][:, sorted_indices]

    for layer in adata_128.layers.keys():
        adata_128_sorted.layers[layer] = adata_128.layers[layer][:, sorted_indices_128]

    return adata_sorted, adata_128_sorted

def add_noise(inputs, noise_factor=0.01):
    """
    Adds Gaussian noise to the inputs for robust training.
    """
    noise = noise_factor * torch.randn_like(inputs)
    return inputs + noise

