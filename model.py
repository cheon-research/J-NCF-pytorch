import torch
import torch.nn as nn
import torch.nn.functional as F 


class JNCF(nn.Module):
    def __init__(self, n_user, n_item, combination):
        super(JNCF, self).__init__()

        self.combination = combination
        self.DF_user = nn.Sequential(
                        nn.Linear(n_item, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU())

        self.DF_item = nn.Sequential(
                        nn.Linear(n_user, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU())

        if self.combination == 'concat':
            feature_size = 64 + 64
        elif self.combination == 'multi':
            feature_size = 64
        else:
            raise ValueError('combination type should be "concat" or "multi" !')

        self.DI = nn.Sequential(
                    nn.Linear(feature_size, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU())

        self.predict_layer = nn.Linear(64, 1, bias=False)

        for layer in self.DF_user:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                layer.bias.data.normal_(0.0, 0.01)

        for layer in self.DF_item:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                layer.bias.data.normal_(0.0, 0.01)

        for layer in self.DI:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                layer.bias.data.normal_(0.0, 0.01)

        nn.init.normal_(self.predict_layer.weight, 0, 0.01)

    def forward(self, user, item):
        user = F.normalize(user)
        item = F.normalize(item)
        user_feature = self.DF_user(user)
        item_feature = self.DF_item(item)

        if self.combination == 'concat':
            feature_vector = torch.cat((user_feature, item_feature), dim=-1)
        elif self.combination == 'multi':
            feature_vector = user_feature * item_feature

        #pred = torch.sigmoid(self.predict_layer(self.DI(feature_vector)))
        pred = self.predict_layer(self.DI(feature_vector))
        return pred.view(-1)