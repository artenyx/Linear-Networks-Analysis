import torch
import pandas as pd
import numpy as np
import time

import load_data


def ae_run_epoch(model, config, grad):
    t0 = time.time()
    if grad:
        loader = config['loaders_usl'][0]
    else:
        loader = config['loaders_usl'][1]
    optimizer = config['optimizer']
    criterion = config['criterion_usl']()

    loss_epoch = 0.0
    for i, (img, targ) in enumerate(loader):
        img.to(config['device'])
        out = model(img)
        loss = criterion(img, out)
        if grad:
            loss.backward()
            optimizer.step()
        loss_epoch += loss.item()
        # out = out.reshape((-1, 1, 28, 28))
    loss_epoch /= len(loader)
    t1 = time.time() - t0
    return loss_epoch, t1


def train_ae_layerwise(model, config):
    load_data.make_dir(config['exp_folder_path'])
    config['optimizer'] = config['optimizer_type'](model.parameters(), lr=config['lr_usl'])
    epochs = config['epochs_per_layer_usl']
    layers_to_add = config['layers_to_add']
    layers_per_step = config['layers_per_step']

    data = []
    for i in range(layers_to_add):
        for epoch in range(epochs):
            loss_train, time_train = ae_run_epoch(model, config, True)
            loss_test, time_test = ae_run_epoch(model, config, False)
            data.append((i, epoch, loss_train, time_train, loss_test, time_test))
            print("Layer: {} || Epoch: {}\nTime: {:02f} || {:02f}\nLoss: {:02f} || {:02f}".format(i, epoch, time_train,
                                                                                                  time_test,
                                                                                                  loss_train,
                                                                                                  loss_test))
        model.add_layers_encoder(layers_per_step)
    data = pd.DataFrame(data)
    data = data.set_axis(["Layer", "Epoch", "Train Loss", "Train Time", "Test Loss", "Test Time"], axis=1)
    data.to_csv(config['exp_folder_path'] + 'data_layerwise.csv')
    print("====COMPLETE====")
    return model, data


def classifier_run_epoch(model, config, grad):
    t0 = time.time()
    optimizer = config['optimizer']
    criterion = config['criterion_class']()
    if grad:
        loader = config['loader_class'][0]
    else:
        loader = config['loader_class'][1]

    tot_correct, tot_samples, tot_loss = np.zeros(3)
    for img, target in loader:
        img = img.to(config['device'])
        target = target.to(config['device'])
        optimizer.zero_grad()
        output = model(img)
        index = torch.argmax(output, 1)
        loss = criterion(output, target)
        if grad:
            loss.backward()
            optimizer.step()
        tot_correct += (index == target).float().sum()
        tot_samples += img.shape[0]
        tot_loss += loss.item()
        err = 1 - tot_correct / tot_samples
    avg_loss = tot_loss / len(loader)
    t1 = time.time() - t0
    return err.item(), avg_loss, t1


def train_classifier(model, config):
    if config['device'] is None:
        raise Exception("Device must be configured in exp_config.")
    config['optimizer'] = config['optimizer_type'](model.parameters(), lr=config['lr_le'])
    train_loader, test_loader = config['loaders']['loaders_le']

    data = []
    for epoch in range(config['num_epochs_le']):
        err_train, loss_train, time_train = classifier_run_epoch(model, config, True)
        err_test, loss_test, time_test = classifier_run_epoch(model, config, False)
        data.append((epoch, loss_train, err_train, time_train, loss_test, err_test, time_test))
        if epoch == 0 or (epoch + 1) % config['print_loss_rate'] == 0:
            print("Epoch: {}\nTime: {:02f} || {:02f}\nLoss: {:02f} || {:02f}\nError: {:02f} || {:02f}".format(epoch,
                                                                                                              time_train,
                                                                                                              time_test,
                                                                                                              loss_train,
                                                                                                              loss_test,
                                                                                                              err_train,
                                                                                                              err_test))
    data = pd.DataFrame(data)
    data = data.set_axis(["Epoch", "Train Loss", "Train Error", "Train Time", "Test Loss", "Test Error", "Test Time"], axis=1)
    data.to_csv(config['exp_folder_path'] + 'data_classif.csv')
    return model, data

