import os
import anndata as ad
from tqdm import tqdm
import argparse
from spared.datasets import get_dataset
from spared import filtering, denoising, gene_features
from spared.layer_operations import *
from spared.filtering import *
from spared.gene_features import *
from spared.denoising import *

class DatasetProcessor:
    def __init__(self, dataset_name: str, save_path: str):
        self.dataset_name = dataset_name
        self.save_path = save_path

        os.makedirs(save_path, exist_ok=True)

        # Load dataset metadata + template adata
        self.dataset = get_dataset(dataset_name)
        self.base_adata = self.dataset.adata.copy()
        self.param_dict = self.dataset.param_dict.copy()

        # Determine organism
        self.param_dict["organism"] = "mouse" if "mouse" in dataset_name else "human"
        self.param_dict["hex_geometry"] = True

    def preprocess_to_1024(self, raw_path: str, total_genes: int = 1024):
        raw_adata = ad.read_h5ad(raw_path)
        adata = raw_adata.copy()

        adata = get_exp_frac(adata)
        adata = get_glob_exp_frac(adata)

        adata.layers['counts'] = adata.X.toarray()
        adata = tpm_normalization(adata, self.param_dict["organism"], from_layer='counts', to_layer='tpm')
        adata = log1p_transformation(adata, from_layer='tpm', to_layer='log1p')

        adata = denoising.median_cleaner(
            adata, from_layer='log1p', to_layer='d_log1p', 
            n_hops=4, hex_geometry=self.param_dict["hex_geometry"]
        )

        adata = gene_features.compute_moran(adata, hex_geometry=self.param_dict["hex_geometry"], from_layer='d_log1p')
        adata = filtering.filter_by_moran(adata, n_keep=total_genes, from_layer='d_log1p')

        adata = combat_transformation(adata, batch_key=self.param_dict['combat_key'], from_layer='log1p', to_layer='c_log1p')
        adata = combat_transformation(adata, batch_key=self.param_dict['combat_key'], from_layer='d_log1p', to_layer='c_d_log1p')

        adata = get_deltas(adata, from_layer='log1p', to_layer='deltas')
        adata = get_deltas(adata, from_layer='d_log1p', to_layer='d_deltas')
        adata = get_deltas(adata, from_layer='c_log1p', to_layer='c_deltas')
        adata = get_deltas(adata, from_layer='c_d_log1p', to_layer='c_d_deltas')

        adata.layers['mask'] = adata.layers['tpm'] != 0

        adata, _ = denoising.spackle_cleaner(
            adata=adata, dataset=self.dataset_name, 
            from_layer="c_d_log1p", to_layer="c_t_log1p", device="cuda"
        )

        adata = get_deltas(adata, from_layer='c_t_log1p', to_layer='c_t_deltas')

        save_file = f"{self.save_path}/{self.dataset_name}_1024.h5ad"
        adata.write(save_file)

        return adata

    def expand_to_1024(self, adata_small: ad.AnnData, adata_1024: ad.AnnData):
        count = 1024 - adata_small.shape[1]
        genes = adata_small.var["gene_ids"].unique().tolist()

        if adata_1024.var.index.name != "gene_ids":
            adata_1024.var = adata_1024.var.set_index("gene_ids", drop=False)
    
        moran_dict = adata_1024.var["d_log1p_moran"].to_dict()
        
        genes_to_add = []

        for key, _ in moran_dict.items():
            if len(genes_to_add) < count:
                if key not in genes:
                    genes_to_add.append(key)
        
        for gene in tqdm(genes_to_add):
            adata_gene = adata_1024[:, adata_1024.var["gene_ids"] == gene]
            adata_small = ad.concat([adata_small, adata_gene], axis=1, merge="same")

        return adata_small

def main():
    parser = argparse.ArgumentParser(
        description="Process dataset to generate 1024-gene adata and optionally expand a smaller reference adata."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset (as recognized by get_dataset)"
    ) 
    parser.add_argument(
        "--save_path",
        type=str,
        default="./adata_1024",
        help="Directory where processed adatas will be stored"
    )

    args = parser.parse_args()
    processor = DatasetProcessor(args.dataset, args.save_path)

    # Step 1: Generate full 1024 gene adata
    raw_path = os.path.join(processor.dataset.dataset_path, "adata_raw.h5ad")
    adata_1024 = processor.preprocess_to_1024(raw_path)

    # Step 2: Expand the 128-gene baseline to match 1024 genes
    adata_128 = processor.base_adata
    complete_adata = processor.expand_to_1024(adata_128, adata_1024)

    complete_adata.write(f"{args.save_path}/{args.dataset}_1024.h5ad")

if __name__ == "__main__":
    main()