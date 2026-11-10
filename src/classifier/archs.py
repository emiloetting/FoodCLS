import torch
import torch.functional as f
import torch.nn as nn



class MLP(nn.Module):
    def __init__(self, in_dims:int, dropout_perc: float, classes:int, label_mapping:dict):
        super().__init__()

        self.label_mapping = label_mapping

        self.input_layer = nn.Linear(in_features=in_dims, out_features=2000)
        self.hidden1 = nn.Linear(self.input_layer.out_features, 1000)
        self.hidden2 = nn.Linear(self.hidden1.out_features, 100)
        self.hidden3 = nn.Linear(self.hidden2.out_features, classes)

        self.softmax = nn.Softmax(1)
        self.leaky_r = nn.LeakyReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=dropout_perc)
        self.dropout2 = nn.Dropout(p=dropout_perc)

        self.forward_seq = nn.Sequential(
            self.input_layer,
            self.leaky_r,

            self.hidden1,
            self.dropout1,
            self.leaky_r,

            self.hidden2,
            self.dropout2,
            self.leaky_r,

            self.softmax
        )

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        """Performs forward pass."""
        return self.forward_seq(X)

    def argmax_forward(self, X) -> torch.Tensor:
        preds = self.forward_seq(X)
        return torch.argmax(preds, dim=1)
