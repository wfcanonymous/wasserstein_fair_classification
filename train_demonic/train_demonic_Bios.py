import logging
import sys

import torch
from fairlib.src.networks.augmentation_layer import Augmentation_layer

from fairlib.src.networks.utils import BaseModel
from sklearn.metrics import accuracy_score

from fairlib.src.networks import get_main_model, MLP
from fairlib.src.utils.utils import seed_everything
from fairlib.src import base_options
from fairlib.src import dataloaders
import os
import sys
from sklearn.model_selection import ParameterGrid

import torch
import torch.nn as nn
import torch.nn.functional as F
from data_generator import load_train_dataloaders

class dummy_args:
    def __init__(self, dataset, data_dir):
        # Creating objects

        self.dataset = dataset
        self.data_dir = data_dir
        self.regression = False
        self.GBT = False
        self.BT = None
        # self.BT = BT
        # self.BTObj = BTObj
        self.adv_BT = None
        self.adv_decoupling = False
        self.encoder_architecture = "Fixed"
        self.emb_size = 768
        self.num_classes = 28
        self.batch_size = 4


def eval_epoch(model, iterator, device):
    epoch_loss = 0

    model.eval()

    criterion = model.criterion

    preds = []
    labels = []

    for batch in iterator:

        text = batch[0]
        p_tags = batch[2]

        text = text.to(device)
        p_tags = p_tags.to(device).long()

        predictions = model(text)

        loss = criterion(predictions, p_tags)

        epoch_loss += loss.item()

        predictions = predictions.detach().cpu()

        p_tags = p_tags.cpu().numpy()
        preds += list(torch.argmax(predictions, axis=1).numpy())
        labels += list(p_tags)

    return (epoch_loss / len(iterator)), accuracy_score(labels, preds)


if __name__ == '__main__':

    dataset = 'Bios_gender'
    if sys.argv[1] == 'EEC':
        training_data ='EEC'
    elif sys.argv[1] == 'Bios':
        training_data = 'Bios'
    else:
        training_data = 'dv2_story'
    name_repo = 'demonic_' + dataset + '_' + training_data

    embedding = 768
    data_dir = "../data/bios"

    args = {
        "dataset": dataset,
        "emb_size": embedding,
        "num_classes": 2,
        "batch_size": 128,
        "lr": 0.001,
        "n_hidden": 2,
        "hidden_size": 300,
        "data_dir": data_dir,
        "device_id": 0,
        "exp_id": name_repo,
        "adv_level": "last_hidden",
    }

    debias_options = base_options.BaseOptions()
    debias_state = debias_options.get_state(args=args, silence=True)

    seed_everything(2022)

    model = MLP(debias_state)  # get_main_model(debias_state)

    # Prepare data
    data_args = dummy_args(args['dataset'], args['data_dir'])  # , self.args.BT, self.args.BTObj)
    task_dataloader = dataloaders.loaders.name2loader(data_args)
    dev_data = task_dataloader(args=data_args, split="dev")
    test_data = task_dataloader(args=data_args, split="test")
    eval_dataloader_params = {
        'batch_size': args['batch_size'],
        'shuffle': True}
    dev_generator = torch.utils.data.DataLoader(dev_data, **eval_dataloader_params)
    test_generator = torch.utils.data.DataLoader(test_data, **eval_dataloader_params)

    if training_data == 'Bios':
        train_data = task_dataloader(args=data_args, split="train")
        train_dataloader_params = {
            'batch_size': args['batch_size'],
            'shuffle': True}
        train_generator = torch.utils.data.DataLoader(train_data, **train_dataloader_params)
    else:
        train_generator = load_train_dataloaders(training_data, batch_size=32, model_name="bert-base-uncased")

    device = model.device

    best_performance = 0
    best_epoch = 0

    for epoch in range(50):
        optimizer = model.optimizer
        criterion = model.criterion

        torch.cuda.empty_cache()

        model.train()
        epoch_loss = 0
        for it, batch in enumerate(train_generator):
            if training_data == 'Bios':
                text = batch[0].float().squeeze().to(device)
                p_tags = batch[2].long().squeeze().to(device)
            else:
                text = batch[0].float().squeeze().to(device)
                p_tags = batch[1].long().squeeze().to(device)

            optimizer.zero_grad()

            predictions = model(text)

            loss = criterion(predictions, p_tags)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        print('Loss at epoch : ', epoch, " is ", epoch_loss / len(train_generator))

        eval_loss, eval_accuracy = eval_epoch(model, test_generator, device)

        print('Accuracy at epoch : ', epoch, " is ", eval_accuracy)

        if eval_accuracy > best_performance:
            best_performance = eval_accuracy
            best_epoch = epoch

            filename = './demon_MLP_bios_' + training_data + '.pt'

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, filename)

    print('Best model has accuracy of : ', best_performance, 'at epoch :', best_epoch)
