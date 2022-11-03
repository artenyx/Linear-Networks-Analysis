import torch.nn as nn
import pandas as pd

import load_data


def ae_run_epoch(model, config, grad):
    if grad:
        loader = config['loaders_usl'][0]
    else:
        loader = config['loaders_usl'][1]
    optimizer = config['optimizer']
    criterion = config['criterion_usl']()

    loss_epoch = 0
    for img, targ in loader:
        img.to(config['device'])
        out = model(img)
        loss = criterion(img, out)
        if grad:
            loss.backward()
            optimizer.step()
        loss_epoch += loss.item()
        #out = out.reshape((-1, 1, 28, 28))
    loss_epoch /= len(loader)
    return loss_epoch


def ae_train_layerwise(model, config):
    load_data.make_dir(config['exp_folder_path'])
    config['optimizer'] = config['optimizer_type'](model.parameters(), lr=config['lr_usl'])
    epochs = config['epochs_per_layer_usl']
    layers_to_add = config['layers_to_add']

    train_loss, test_loss = [], []
    for i in range(layers_to_add):
        for epoch in range(epochs):
            train_loss.append((i, epoch, ae_run_epoch(model, config, True)))
            test_loss.append((i, epoch, ae_run_epoch(model, config, False)))
        model.add_layers_encoder(1)
        print(model.layers)
    train_loss = pd.DataFrame(train_loss)
    train_loss.to_csv(config['exp_folder_path']+'train_loss.csv')
    test_loss = pd.DataFrame(test_loss)
    test_loss.to_csv(config['exp_folder_path']+'test_loss.csv')
    print("====COMPLETE====")
    return

'''
def train_AE(model, first):
    """ Train a model. """
    config = get_model_configuration()
    loss_function = config.get("loss_function_AE")()
    optimizer = config.get("optimizer")(model.parameters(), lr=config.get("lr_AE"))

    if first:
        epochs = config.get("num_epochs_init")
    else:
        epochs = config.get("num_epochs")
    trainloader, testloader = get_MNIST()
    directory = "test"
    make_dir("test")

    train_loss, test_loss = 0.0, 0.0
    train_loss_list, test_loss_list = [], []

    # Iterate over the number of epochs
    for epoch in range(epochs):
        # Train Epoch Sequence
        current_loss = 0.0
        for inputs, targets in trainloader:
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            inputs = inputs.reshape(-1, inputs.shape[1] * inputs.shape[2] * inputs.shape[3])
            loss = loss_function(inputs, outputs)
            loss.backward()
            optimizer.step()

            current_loss += loss.item()
        train_loss = current_loss / len(trainloader)
        train_loss_list.append(train_loss)
        outputs = outputs.view(outputs.size(0), 1, 28, 28).cpu().data
        inputs = inputs.view(inputs.size(0), 1, 28, 28).cpu().data
        save_image(inputs, './{}/input.png'.format(directory))
        save_image(outputs, './{}/output.png'.format(directory))

        # Test Epoch Sequence
        with torch.no_grad():
            current_loss = 0.0
            for inputs, targets in testloader:
                if torch.cuda.is_available():
                    inputs, targets = inputs.cuda(), targets.cuda()

                outputs = model(inputs)
                inputs = inputs.reshape(-1, inputs.shape[1] * inputs.shape[2] * inputs.shape[3])
                loss = loss_function(inputs, outputs)
                current_loss += loss.item()
            test_loss = current_loss / len(testloader)
            test_loss_list.append(test_loss)
            print("{:03d}/{:03d} Train loss: {:0.6f} || Test loss: {:0.6f}".format(epoch + 1, epochs, train_loss,
                                                                                   test_loss))
    depth = len(model.cpu().layers)
    data = {"Depth": depth, "Epoch": range(epochs), "Train Loss": train_loss_list, "Test Loss": test_loss_list}
    data = pd.DataFrame(data)

    return model, data


def greedy_layerwise_training(model, layers_to_add):
    first = True
    # Iterate over the number of layers to add
    for num_layers in range(layers_to_add + 1):
        # Print which model is trained
        print("=" * 100)
        if num_layers == 0:
            print(f">>> TRAINING THE BASE MODEL:")
            model, data = train_AE(model, first)
        else:
            print(f">>> TRAINING THE MODEL WITH {num_layers} ADDITIONAL LAYERS:")
            first = False
            model = add_layer(model)
            model, data = train_AE(model, first)

    # Process is complete
    print("Training process has finished.")
    return model, data
'''
