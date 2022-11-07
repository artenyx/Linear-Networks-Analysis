import torch
import torch.nn as nn


def get_config():
    config = {
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "batch_size": 256,
        "input_size": (1, 28, 28),
        "latent_size": 20,
        "layer_size": 200,
        "num_classes": 10,

        "criterion_class": nn.CrossEntropyLoss,
        "criterion_usl": nn.MSELoss,
        "optimizer_type": torch.optim.Adam,
        "loaders_usl": None,
        "epochs_per_layer_usl_init": 1,
        "epochs_per_layer_usl": 1,
        "epochs_classif": 1,
        "lr_usl": 0.001,
        "lr_classif": 0.01,
        "layers_to_add": 0,
        "print_loss_rate": 1
    }

    return config
