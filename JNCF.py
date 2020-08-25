import torch
import torch.nn as nn
import torch.nn.functional as F 


class JNCF(nn.Module):
    def __init__(self, layers):
        super(JNCF, self).__init__()
        MLP_modules = []
        for i, dim in enumerate(layers):
            MLP_modules.append(nn.Linear(dim * 2, dim))
            MLP_modules.append(nn.LeakyReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)
        self.predict_layer = nn.Linear(layers[-1], 1)

        for layer in self.MLP_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, a=1e-2, nonlinearity='leaky_relu')
                layer.bias.data.data.normal_(0.0, 0.001)
        nn.init.xavier_normal_(self.predict_layer.weight)
        self.predict_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, user, item):
        concat = torch.cat((user, item), dim=-1)
        mlp_vector = self.MLP_layers(concat)
        pred = self.predict_layer(mlp_vector)
        return pred.view(-1)