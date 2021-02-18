import torch
import torch.nn as nn
import torch.nn.functional as F 


class JNCF(nn.Module):
    def __init__(self, DF_layers, DI_layers, n_user, n_item, combination):
        super(JNCF, self).__init__()

        self.embed_dim = DF_layers[-1]
        self.factor_dim = DI_layers[-1]

        self.combination = combination
        if self.combination == 'concat':
            self.embed_dim *= 2
        elif self.combination == 'multi':
            pass
        else:
            raise ValueError('combination type should be "concat" or "multi" !')

        user_MLP = []
        for i in range(len(DF_layers)):
            if i == 0:
                user_MLP.append(nn.Linear(n_item, DF_layers[i]))
                user_MLP.append(nn.ReLU())
            else:
                user_MLP.append(nn.Linear(DF_layers[i-1], DF_layers[i]))
                user_MLP.append(nn.ReLU())
        self.DF_user = nn.Sequential(*user_MLP)

        item_MLP = []
        for i in range(len(DF_layers)):
            if i == 0:
                item_MLP.append(nn.Linear(n_user, DF_layers[i]))
                item_MLP.append(nn.ReLU())
            else:
                item_MLP.append(nn.Linear(DF_layers[i-1], DF_layers[i]))
                item_MLP.append(nn.ReLU())
        self.DF_item = nn.Sequential(*item_MLP)

        DI_MLP = []
        for i in range(len(DI_layers)):
            if i == 0:
                DI_MLP.append(nn.Linear(self.embed_dim, DI_layers[i]))
                DI_MLP.append(nn.ReLU())
            else:
                DI_MLP.append(nn.Linear(DI_layers[i-1], DI_layers[i]))
                DI_MLP.append(nn.ReLU())
        self.DI = nn.Sequential(*DI_MLP)

        self.predict_layer = nn.Linear(self.factor_dim, 1, bias=False)

        self.init_weights()


    def init_weights(self):
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


    def forward(self, user, item_i, item_j):
        user_feature = self.DF_user(user)
        item_i_feature = self.DF_item(item_i)
        item_j_feature = self.DF_item(item_j)

        if self.combination == 'concat':
            i_feature_vector = torch.cat((user_feature, item_i_feature), dim=-1)
            j_feature_vector = torch.cat((user_feature, item_j_feature), dim=-1)
        elif self.combination == 'multi':
            i_feature_vector = user_feature * item_i_feature
            j_feature_vector = user_feature * item_j_feature

        y_i = self.predict_layer(self.DI(i_feature_vector))
        y_j = self.predict_layer(self.DI(j_feature_vector))
        return y_i.view(-1), y_j.view(-1)