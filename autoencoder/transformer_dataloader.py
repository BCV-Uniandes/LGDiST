from torch.utils.data import Dataset

class CombinedDataset(Dataset):
    def __init__(self, tensor1, tensor2, tensor3):
        """
        Initializes the CombinedDataset.

        Args:
            tensor1 (torch.Tensor): The first tensor to combine.
            tensor2 (torch.Tensor): The second tensor to combine.
        """
        assert len(tensor1) == len(tensor2), "Tensors must have the same number of samples"
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        self.tensor3 = tensor3

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.tensor1)

    def __getitem__(self, idx):
        """
        Retrieves a pair of tensors by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the corresponding elements from tensor1 and tensor2.
        """
        return self.tensor1[idx], self.tensor2[idx], self.tensor3[idx]
        