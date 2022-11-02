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
        "optimizer_type": torch.optim.SGD,
        "epochs_usl_init": 1,
        "epochs_usl": 1,
        "epochs_classif": 1,
        "lr_usl": 0.07,
        "lr_classif": 0.01,
        "total_layers": 0,

        "current_epoch": None,
        "current_layer": None
    }

    return config
