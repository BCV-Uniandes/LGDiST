## Dataset Preprocessing

Before training LGDiST, each dataset must be preprocessed to extract the **Highly Spatially Associated Genes (HSAGs)** and **context genes**. This step reduces the gene expression matrix to **1024 genes**, which are used for model training.

### Requirements
Make sure you have installed the dependencies listed in `requirements.txt`.

### Usage
Run the preprocessing script:

```bash
python preprocessing.py --dataset <dataset_name> --save_path <output_path>
```
where <dataset_name> is the name of the adata you want to process and <output_path> corresponds to the directory where the processed adatas will be stored.
