import torch.nn as nn
import numpy as np


class Linear_AE_LC(nn.Module):
    def __init__(self, config):
        super(Linear_AE_LC, self).__init__()

        # Retrieve model configuration
        input_size = np.prod(config["input_size"])
        layer_size = config["layer_size"]
        latent_size = config["latent_size"]

        self.layers_encoder = [nn.Flatten(),
                               nn.Linear(input_size, layer_size)]
        self.layers_decoder = [nn.Linear(layer_size, latent_size),
                               nn.Linear(latent_size, input_size),
                               nn.Unflatten(-1, (1, 28, 28))]
        self.layers = nn.Sequential(*(self.layers_encoder + self.layers_decoder))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, inp):
        #self.set_structure()
        return self.layers(inp)

    def set_structure(self):
        self.layers = nn.Sequential(*(self.layers_encoder + self.layers_decoder))
