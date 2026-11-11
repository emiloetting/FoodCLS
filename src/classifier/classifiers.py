import torch
import torch.functional as f
import torch.nn as nn



class MLP(nn.Module):
    def __init__(self, input_features:int):
        super().__init__()

        self.class_mapping = {
            0: "Obst & Gemüse",
            1: "Fleisch, Fisch & Alternativen",
            2: "Milchprodukte & Eier",
            3: "Brot & Backwaren",
            4: "Nudeln, Reis & Getreide",
            5: "Fertigprodukte",
            6: "Backzutaten",
            7: "Getränke",
            8: "Snacks & Süßigkeiten",
            9: "Öle, Gewürze & Saucen",
            10: "Brotaufstriche",
            11: "Sonstiges"
        }

        self.input_layer = nn.Linear(in_features=input_features, out_features=2000)     # classes 0-11
        self.hidden1 = nn.Linear(self.input_layer.out_features, 1000)
        self.hidden2 = nn.Linear(self.hidden1.out_features, 100)
        self.hidden3 = nn.Linear(self.hidden2.out_features, 12)

        self.softmax = nn.Softmax(1)
        self.leaky_r = nn.LeakyReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=.2)
        self.dropout2 = nn.Dropout(p=.2)

        self.forward_seq = nn.Sequential(
            self.input_layer,
            self.leaky_r,

            self.hidden1,
            self.dropout1,
            self.leaky_r,

            self.hidden2,
            self.dropout2,
            self.leaky_r,

            self.hidden3
        )

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        """Performs forward pass."""
        return self.forward_seq(X)

    def predict(self, X, string_classes:bool=True) -> torch.Tensor:
        if X.ndim == 1:
            X = X.unsqueeze(0)
        preds = self.forward_seq(X)
        res = int(torch.argmax(preds, dim=1).cpu())
        return self.class_mapping[res] if string_classes else res
