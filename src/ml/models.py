import torch
from torch import nn

class MarkerModel(nn.Module):
    """
    TODO
    """
    @property
    def device(self):
        for p in self.parameters():
            return p.device

    def __init__(self, pre_trained_model, n_targets, input_size, hidden_layers, p=0.0,
                 train_only_head=True):
        """
        TODO
        """
        super(MarkerModel, self).__init__()

        self.pre_trained_model = pre_trained_model
        if train_only_head:
            for param in self.pre_trained_model.base_model.parameters():
                param.requires_grad = False
        self.FFNN = torch.nn.Sequential()
        input_dim = input_size
        for i, hidden_dim in enumerate(hidden_layers):
            self.FFNN.add_module('linear{}'.format(i), nn.Linear(input_dim, hidden_dim))
            self.FFNN.add_module('relu{}'.format(i), nn.ReLU())
            self.FFNN.add_module('dropout{}'.format(i), nn.Dropout(p=p))
            self.FFNN.add_module('norm{}'.format(i), nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim
        self.FFNN.add_module('classifier', nn.Linear(input_dim, n_targets))
        self.FFNN.add_module('softmax', nn.Softmax(dim=1))

    def forward(self, input_ids, attention_mask):
        """
        TODO
        """
        x = self.pre_trained_model(input_ids=input_ids,
                                   attention_mask=attention_mask)['last_hidden_state'][:,0,:]
        x = self.FFNN(x)
        return x