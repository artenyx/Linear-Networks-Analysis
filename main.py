import torch
import torch.nn as nn

import networks
import exp_config


config1 = exp_config.get_config()
config1["device"] = torch.device("cpu")
model = networks.Linear_AE_LC(config1)
print(model.layers_encoder)
model.layers_encoder += [nn.Linear(200, 200)]
print(model.layers)
test_tensor = torch.randn((1, 1, 28, 28))
print(test_tensor.shape)

test_out = model(test_tensor)
print(test_out.shape)
print(model.layers)
