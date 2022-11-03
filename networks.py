import torch.nn as nn
import numpy as np


def lin_layer_init(layer):
    nn.init.orthogonal_(layer.weight)
    nn.init.constant_(layer.bias, 0)


class Linear_AE_LC(nn.Module):
    def __init__(self, config):
        super(Linear_AE_LC, self).__init__()

        # Retrieve model configuration
        self.input_size = np.prod(config["input_size"])
        self.layer_size = config["layer_size"]
        self.latent_size = config["latent_size"]

        self.layers_encoder = [nn.Flatten(),
                               nn.Linear(self.input_size, self.layer_size)]
        self.layers_decoder = [nn.Linear(self.layer_size, self.latent_size),
                               nn.Linear(self.latent_size, self.input_size),
                               nn.Unflatten(-1, (1, 28, 28))]
        self.layers = nn.Sequential(*(self.layers_encoder + self.layers_decoder))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                lin_layer_init(m)

    def forward(self, inp):
        return self.layers(inp)

    def set_structure(self, layers_list=None):
        if layers_list is None:
            self.layers = nn.Sequential(*(self.layers_encoder + self.layers_decoder))
        else:
            self.layers = nn.Sequential(*layers_list)

    def add_layers_encoder(self, num_layers, prev_grad_off=True):
        if prev_grad_off:
            for layer in self.layers_encoder:
                for m in layer.parameters():
                    m.requires_grad = False
        new_layers_list = []
        for i in range(num_layers):
            new_layer = nn.Linear(self.layer_size, self.layer_size)
            nn.init.xavier_normal_(new_layer.weight)
            nn.init.constant_(new_layer.bias, 0)
            new_layers_list.append(new_layer)
        self.layers_encoder.extend(new_layers_list)
        self.set_structure()

