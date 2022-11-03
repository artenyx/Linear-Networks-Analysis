import torch
import torch.nn as nn
from datetime import datetime

import networks, exp_config, load_data, train

def main():
    config = exp_config.get_config()
    config['device'] = torch.device("cpu")
    config['loaders_usl'] = load_data.get_mnist(config)
    load_data.make_dir('ExperimentFiles/')
    config['exp_folder_path'] = 'ExperimentFiles/' + 'lw_ae_exp_' + datetime.now().strftime("%m-%d-%Y_%H-%M-%S") + '/'

    model = networks.Linear_AE_LC(config)

    config['layers_to_add'] = 5
    config['epochs_per_layer_usl'] = 50

    train.ae_train_layerwise(model, config)
    return


if __name__ == "__main__":
    main()
