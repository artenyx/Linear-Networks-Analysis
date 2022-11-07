import torch
from datetime import datetime
import argparse


import networks, exp_config, load_data, train

def main(args):
    config = exp_config.get_config()
    config['device'] = torch.device("cpu")
    config['loaders_usl'] = load_data.get_mnist(config)
    load_data.make_dir(args.data_root)
    config['exp_folder_path'] = args.data_root + 'lw_ae_exp_' + datetime.now().strftime("%m-%d-%Y_%H-%M-%S") + '/'

    model = networks.Linear_AE_LC(config)

    config['layers_to_add'] = args.add_layers
    config['epochs_per_layer_usl'] = args.epochs_per_layer
    config['tqdm'] = args.tqdm

    train.ae_train_layerwise(model, config)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Linear Networks Analysis")
    parser.add_argument("--data_root", type=str, default="ExperimentFiles/")
    parser.add_argument("--add_layers", type=int, required=True, help="Number of layers to add to base model.")
    parser.add_argument("--epochs_per_layer", type=int, required=True, help="Number of epochs to run for each layer added.")
    parser.add_argument("--tqdm", action="store_false", help="Run tqdm on batch loading for every epoch.")
    '''
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "svhn", "cifar100", "MNIST"])
    parser.add_argument("--data_root", type=str, default="../data")
    # optimization
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay_epochs", nargs="+", type=int, default=[160, 180],
                        help="decay learning rate by decay_rate at these epochs")
    parser.add_argument("--decay_rate", type=float, default=.3,
                        help="learning rate decay multiplier")
    parser.add_argument("--clf_only", action="store_true", help="If set, then only train the classifier")
    parser.add_argument("--labels_per_class", type=int, default=-1,
                        help="number of labeled examples per class, if zero then use all labels")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--warmup_iters", type=int, default=-1,
                        help="number of iters to linearly increase learning rate, if -1 then no warmmup")
    # loss weighting
    parser.add_argument("--p_x_weight", type=float, default=1.)
    parser.add_argument("--p_y_given_x_weight", type=float, default=1.)
    parser.add_argument("--p_x_y_weight", type=float, default=0.)
    # regularization
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    '''
    args = parser.parse_args()

    main(args)
