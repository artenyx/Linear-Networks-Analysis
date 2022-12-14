import pandas as pd
import torch
from datetime import datetime
import argparse


import networks, exp_config, load_data, train


def main(args):
    config = exp_config.get_config()
    config['layers_to_add'] = args.add_layers
    config['layers_per_step'] = args.layers_per_step
    config['steps'] = int(args.add_layers / args.layers_per_step)
    config['lr_usl'] = args.lr_usl
    config['lr_le'] = args.lr_le
    config['epochs_per_layer_usl'] = args.epochs_per_step
    config['device'] = torch.device("cpu")

    config['exp_folder_path'] = args.data_root + 'lw_ae_exp_' + datetime.now().strftime("%m-%d-%Y_%H-%M-%S") + '/'
    load_data.make_dir(config['exp_folder_path'])

    config_df = pd.DataFrame(config)
    config_df.to_csv(config['exp_folder_path']+"config")

    config['loaders_usl'] = load_data.get_mnist(config)
    config['loaders_class'] = load_data.get_mnist(config)
    load_data.make_dir(args.data_root)

    model = networks.Linear_AE_LC(config)



    model, data = train.train_ae_layerwise(model, config)
    train.train_classifier(model, config)
    return


if __name__ == "__main__":
    print("Running on cuda." if torch.cuda.is_available() else "Running on cpu.")
    parser = argparse.ArgumentParser("Linear Networks Analysis")
    parser.add_argument("--data_root", type=str, default="ExperimentFiles/")
    parser.add_argument("--add_layers", type=int, required=True, help="Number of layers to add to base model.")
    parser.add_argument("--layers_per_step", type=int, default=1, help="Number of layers to add to per step.")
    parser.add_argument("--epochs_per_step", type=int, required=True, help="Number of epochs to run for each layer added.")
    parser.add_argument("--epochs_classif", type=int, required=True, help="Number of epochs to run classifier for after initialization.")
    parser.add_argument("--lr_usl", type=float, default=0.000001)
    parser.add_argument("--lr_le", type=float, default=0.0001)

    args = parser.parse_args()
    if args.add_layers % args.layers_per_step != 0:
        raise Exception("Added layers must be multiple of layers per step.")

    main(args)
