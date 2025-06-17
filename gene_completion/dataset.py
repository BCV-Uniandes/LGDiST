from Transformer_simple import Transformer
from torch.utils.data import DataLoader
from tqdm import tqdm
import anndata as ad
from utils import *
import numpy as np
import torch
torch.set_num_threads(1)

parser = get_main_parser()
args = parser.parse_args()

# seed everything
seed = args.seed
seed_everything(seed)

class stLDMDataset(torch.utils.data.Dataset):
    def __init__(self, args, adata, split_name, spared_genes_names, model_autoencoder):
        """
        This is a spatial data class that contains all the information about the dataset. It will call a reader class depending on the type
        of dataset (by now only visium and STNet are supported). The reader class will download the data and read it into an AnnData collection
        object. Then the dataset class will filter, process and plot quality control graphs for the dataset. The processed dataset will be stored
        for rapid access in the future.

        Args:
            adata (ad.AnnData): An anndata object with the data of the entire dataset.
            args (argparse): parser with the values necessary for data processing.
            split_name (str): name of the data split being processed. Useful for identifying which data split the model is being tested on.
            pre_masked (str, optional): specifies if the data incoming has already been masked for testing purposes. 
                    * If True, __getitem__() will return the random mask that was used to mask the original expression 
                    values instead of the median imputation mask, as well as the gt expressions and the masked data.
        """

        self.args = args
        self.pred_layer = args.pred_layer
        self.split_name = split_name
        self.adata = adata
        self.spared_genes_ids = spared_genes_names
        self.model_autoencoder = model_autoencoder
        # Get original expression matrix based on selected prediction layer.
        self.expression_mtx = torch.tensor(self.adata.layers[self.pred_layer])
        # Retrieve the mask from the adata, where "False" corresponds to the SpaCKLE-imputed values or that correspond to non-SpaRED-genes.
        self.great_mask = self.build_general_mask()
      
        # Get adjacency matrix.
        self.adj_mat = None
        self.get_adjacency(self.args.num_neighs)

        # Build and save each spot's neighborhood, and the min and max val of the data split
        self.min_val, self.max_val = np.inf, -np.inf 
        self.neighborhoods = self.build_neighborhoods()

        # Normalize data if needed (data that will be the model's input, i.e. encoded matrices)
        if self.args.normalize_input:
            self.normalize_full_data()
    
    def build_general_mask(self):
        """
        Combines the mask from the adata.layers section to hide the values that were originally 
        artificially completed (i. e. completion through adaptive median or SpaCKLE), with a mask that 
        hides the columns of the genes that are not part of the "genes_to_keep" array. 
        
        This general mask is needed to compute post-decoding metrics.
        """

        # Build bool array, with True in the idxs corresponding to genes present in the SpaRED set
        mask_to_keep = self.adata.var['gene_ids'].isin(self.spared_genes_ids)
        # Get original mask (False in the values that were previously completed using SpaCKLE)
        original_mask = torch.tensor(self.adata.layers["mask"])
        new_mask = original_mask.clone()
        # Set columns corresponding to genes not in 'genes_to_keep' to False
        new_mask[:, ~mask_to_keep] = False
        # Add the modified mask to adata and return it
        self.adata.layers["general_mask"] = new_mask

        return new_mask

    def get_adjacency(self, num_neighs = 6):
        """
        Function description
        """
        # Get num_neighs nearest neighbors for each spot
        sq.gr.spatial_neighbors(self.adata, coord_type='generic', n_neighs=num_neighs)
        self.adj_mat = torch.tensor(self.adata.obsp['spatial_connectivities'].todense())
    
    def build_neighborhoods(self):
        """
        Creates a dictionary of dictionaries, where each element/key corresponds to an individual spot in the
        adata, and each inner-dictionary/value corresponds to its own neighborhood's expression matrix,
        gene-expression-mask, and encoded expression matrix.
        """
        all_neighborhoods = {}
        for idx, spot_name in enumerate(self.adata.obs["unique_id"].unique()):
            # Get gt expression for idx spot and its nn
            spot_exp = self.expression_mtx[idx].unsqueeze(dim=0)
            nn_exp = self.expression_mtx[self.adj_mat[:,idx]==1.]
            exp_matrix = torch.cat((spot_exp, nn_exp), dim=0).type('torch.FloatTensor')

            # Get median imputation mask for idx spot and its nn
            spot_mask = self.great_mask[idx].unsqueeze(dim=0) #size 1xgenes(1024)
            nn_mask = self.great_mask[self.adj_mat[:,idx]==1.] #size 6xgenes(1024)
            great_mask = torch.cat((spot_mask, nn_mask), dim=0)

            all_neighborhoods[str(idx)] = {"spot_id": spot_name, 
                                           "exp_matrix": exp_matrix, 
                                           "exp_mask": great_mask,}

        all_neighs = []
        for idx, spot_name in tqdm(enumerate(self.adata.obs["unique_id"].unique())):
            all_neighs.append(all_neighborhoods[str(idx)]["exp_matrix"])

        data = torch.stack(all_neighs)
        data = torch.tensor(data)
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

        # Encode neighborhoods
        encoded_neighborhoods = []
        self.model_autoencoder.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch = batch[0].to("cuda")
                encoded_exp_matrix = self.model_autoencoder.encoder(batch)
                encoded_neighborhoods.append(encoded_exp_matrix.detach().cpu())
            
        encoded_neighborhoods = torch.cat(encoded_neighborhoods, dim=0)

        # Set min and max values of the data split
        self.min_val = encoded_neighborhoods.min().item()
        self.max_val = encoded_neighborhoods.max().item()    

        # Add encoded layer to dictionaries
        for sample_num in range(encoded_neighborhoods.shape[0]):
            all_neighborhoods[str(sample_num)]["encoded_exp_matrix"] = encoded_neighborhoods[sample_num]

        return all_neighborhoods
    
    def normalize_full_data(self):
        """
        Calls for the normalization function from utils to normalize all neighborhoods/samples
        based on the min and max values of the complete data split.
        """
        for spot_idx in self.neighborhoods.keys():
            encoded_exp_mt = self.neighborhoods[spot_idx]["encoded_exp_matrix"]
            self.neighborhoods[spot_idx]["encoded_exp_matrix"] = data_normalization(encoded_exp_mt, self.min_val, self.max_val)  
    
    def add_noise_to_neighborhoods(self, exp_matrix):
        
        if self.args.noise_fraction == 1:
            exp_matrix[...,0,:] = 0   
            noise_mask = torch.ones(exp_matrix.shape)
            noise_mask[...,0,:] = 0  

        else: 
            # TODO: agregar código para agregar ruido en imputación parcial
            pass    

        return exp_matrix, noise_mask
    
    def __getitem__(self, idx):
        """
        An item returns a dictionary with the following keys and values:
            - 'spot_id': string that corresponds to the id name of the main spot in the current sample.
            - 'exp_matrix': expression matrix with values from adata.layer[pred_layer]. Not encoded, nor normalized.
            - 'exp_mask': bool matrix with the values from adata.layer[pred_layer], as well as False in the columns of genes that aren't part of SpaRED.
            - 'encoded_exp_matrix': same as "exp_matrix" but encoded and then normalized with the min and max values of whole data split.
            - 'condition_matrix': same as "encoded_exp_matrix" but with 0 in the hidden values (the row/column of the main spot).
            - 'condition_mask': bool matrix with False in the values that were changed to 0 for "condition_matrix" (the row/column of the main spot).
        """
        item = self.neighborhoods[str(idx)]
        # Add noise to encoded neighborhood data
        encoded_data = item["encoded_exp_matrix"].clone()
        item["condition_matrix"], item["condition_mask"] = self.add_noise_to_neighborhoods(encoded_data)

        return item

    def __len__(self):
        return len(self.adata)


class SpaREDData():
    def __init__(self, args, autoencoder):
        super().__init__()

        self.args = args
        self.dataset_name = args.dataset
        self.batch_size = args.batch_size
        self.num_neighs = args.num_neighs
        self.prediction_layer = args.pred_layer
        self.autoencoder = autoencoder

        # Load datasets (1024-gene adata, and original SpaRED adata)
        self.load_data()
        # Sort genes in adatas
        self.sort_adatas()
        # Get indexes/location of important genes in 1024-data after sorting
        self.gene_weights = torch.tensor(np.isin(self.full_adata.var['gene_ids'].unique(), self.spared_genes_array))
        # Get average values for 1024-genes adata
        avg_layer = f'{"_".join(self.prediction_layer.split("_")[:-1])}_log1p_avg_exp'
        self.average_vals = torch.tensor(self.full_adata.var[avg_layer]).unsqueeze(0)
        if args.partial:
            # Get tensor with each gene's probability of being masked
            self.mask_prob_tensor = get_mask_prob_tensor(
                masking_method=args.masking_method, 
                adata=self.original_full_adata,
                mask_prob=args.mask_prob,
                scale_factor=args.scale_factor
            )
            # Create synthetic masking and save it in SpaRED/original adata of 128 or 32 genes
            # In the new layer "random_mask", True represents the values that are synthetically masked and that will be predicted
            self.original_full_adata = mask_exp_matrix(
                adata=self.original_full_adata,
                pred_layer=args.pred_layer,
                mask_prob_tensor=self.mask_prob_tensor,
                device="cuda"
            )
        
        # Set split data and create data modules
        self.setup()
        self.train_data = stLDMDataset(self.args, self.spared_train, "train", self.spared_genes_array, self.autoencoder)
        self.val_data = stLDMDataset(self.args, self.spared_val, "val", self.spared_genes_array, self.autoencoder)
        self.test_data = stLDMDataset(self.args, self.spared_test, "test", self.spared_genes_array, self.autoencoder)
        if args.full_inference:
            self.all_data = stLDMDataset(self.args, self.full_adata, "all", self.spared_genes_array, self.autoencoder)

    def load_data(self):
        self.adata_path = f"/media/disk0/pcardenasg/SpaRED/datasets/1024/{self.dataset_name}_1024.h5ad"
        #self.adata_path = f"/media/disk0/pcardenasg/SpaRED/datasets/original/{self.dataset_name}.h5ad"
        self.full_adata = ad.read_h5ad(self.adata_path)
        # Check if test split is available
        self.test_data_available = True if 'test' in self.full_adata.obs['split'].unique() else False
        # Get number of genes in dataset
        self.n_genes = self.full_adata.n_vars

        # Load original SpaRED adata 
        self.original_adata_path = f"/media/disk0/pcardenasg/SpaRED/datasets/original/{self.dataset_name}.h5ad"
        self.original_full_adata = ad.read_h5ad(self.original_adata_path)
        
        # Get array of genes of interest
        self.spared_genes_array = self.original_full_adata.var['gene_ids'].unique()
        # tensor of bool values that indicate which genes to focus on during loss
        self.gene_weights = torch.tensor(np.isin(self.full_adata.var['gene_ids'].unique(), self.spared_genes_array))
        print(f"Genes from full adata that are part of the SpaRED list: {self.gene_weights.sum()}")

    def sort_adatas(self):
        self.full_adata.var.reset_index(drop=True, inplace=True)
        self.original_full_adata.var.reset_index(drop=True, inplace=True)

        adata_1024_sorted = self.full_adata.copy()
        adata_original_sorted = self.original_full_adata.copy()
        
        # Sort genes by index
        adata_1024_sorted.var["original_index"] = adata_1024_sorted.var.index
        adata_1024_sorted.var = adata_1024_sorted.var.sort_values(by="gene_ids").reset_index(drop=True)

        adata_original_sorted.var["original_index"] = adata_original_sorted.var.index
        adata_original_sorted.var = adata_original_sorted.var.sort_values(by="gene_ids").reset_index(drop=True)

        #Get indices
        sorted_indices_1024 = adata_1024_sorted.var["original_index"].to_numpy()
        sorted_indices_1024 = [int(idx) for idx in sorted_indices_1024]
        
        sorted_indices_original = adata_original_sorted.var["original_index"].to_numpy()
        sorted_indices_original = [int(idx) for idx in sorted_indices_original]

        # Reorder all layers to match the new gene order
        for layer in self.full_adata.layers.keys():
            adata_1024_sorted.layers[layer] = self.full_adata.layers[layer][:, sorted_indices_1024]

        for layer in self.original_full_adata.layers.keys():
            adata_original_sorted.layers[layer] = self.original_full_adata.layers[layer][:, sorted_indices_original]

        self.full_adata = adata_1024_sorted
        self.original_full_adata = adata_original_sorted

    def setup(self):
        # Assign train/val/test datasets for use in dataloaders
        self.spared_train = self.full_adata[self.full_adata.obs["split"]=="train"] 
        self.spared_val = self.full_adata[self.full_adata.obs["split"]=="val"]
        self.spared_test = self.full_adata[self.full_adata.obs["split"]=="test"] if self.test_data_available else  self.full_adata[self.full_adata.obs["split"]=="val"]

    def train_dataloader(self):
        # item is a dictionary with keys ['spot_id', 'exp_matrix', 'exp_mask', 'encoded_exp_matrix', 'condition_matrix', 'condition_mask']
        # keys used during train: ['encoded_exp_matrix', 'condition_matrix', 'condition_mask']
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=False) #, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, drop_last=False) #, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, drop_last=False) #, num_workers=self.num_workers)
    
    def all_dataloader(self):
        return DataLoader(self.all_data, batch_size=self.batch_size, shuffle=False, drop_last=False) #, num_workers=self.num_workers)


if __name__ == "__main__":
    
    print(args)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Load trained autoencoder
    print("Using a Transformer-MLP autoencoder for gene preprocessing")
    autoencoder = Transformer(
    input_dim=1024, 
    latent_dim=128, 
    output_dim=1024,
    embedding_dim=256,
    num_layers=2,
    num_heads=2,
    lr=args.lr
    )

    checkpoints = torch.load(args.autoencoder_ckpts_path)
    autoencoder.load_state_dict(checkpoints['state_dict'])
    autoencoder = autoencoder.to(device)

    breakpoint()
    spared_data = SpaREDData(args, autoencoder)
    AA = next(iter(spared_data.train_dataloader()))
    print("finish")

