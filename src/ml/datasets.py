import torch

class CustomDataset(torch.utils.data.Dataset):
    """
    TODO
    """
    def __init__(self, X, y):
        self.input_ids = torch.tensor(X['input_ids']).to(torch.int64)
        self.attention_masks = torch.tensor(X['attention_mask']).to(torch.int64)
        self.y = torch.tensor(y).to(torch.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.input_ids[i,:], self.attention_masks[i,:], self.y[i]