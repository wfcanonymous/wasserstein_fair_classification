import copy

import numpy as np
import torch.nn as nn
import torch
import logging

from fairlib.src import dataloaders, base_options
from fairlib.src.dataloaders import text2id
from fairlib.src.evaluators import gap_eval_scores
from ..evaluators.leakage_metrices import leakage_evaluation
from torch import autograd
from torch.optim import Adam
import time
from pathlib import Path
from ..evaluators import print_network, present_evaluation_scores, validation_is_best
import pandas as pd
from .knn_labels import KNN_labels
import sys
from torch.utils.data import Sampler
import random
from torch.utils.data import Dataset
from torch import linalg as LA




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


class dummy_args_bert:
    def __init__(self, dataset, data_dir, architecture, encoder=None):
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
        self.encoder_architecture = architecture
        self.emb_size = 768
        self.num_classes = 28
        self.batch_size = 4
        self.text_encoder = encoder


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class OneClassSampler(Sampler):
    """
    Samples elements from a dataset such that each batch only contains
    samples from a single class.
    """

    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        self.class_to_indices = {}
        for i, (_, label) in enumerate(data_source):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(i)
        self.classes = list(self.class_to_indices.keys())

    def __iter__(self):
        # Shuffle the classes
        random.shuffle(self.classes)
        # Iterate over the classes and create batches
        for c in self.classes:
            indices = self.class_to_indices[c]
            batch = []
            for idx in indices:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if len(batch) > 0:
                yield batch

    def __len__(self):
        # Calculate the total number of batches
        num_batches = sum(
            (len(self.class_to_indices[c]) + self.batch_size - 1) // self.batch_size
            for c in self.classes
        )
        return num_batches * self.batch_size


# Define the Critic model
class Critic(nn.Module):
    def __init__(self, nb_linear=4, input_dim=600):
        super(Critic, self).__init__()

        self.hidden_dim = 512
        self.input_dim = input_dim

        if nb_linear == 2:
            self.main = nn.Sequential(
                # 1
                nn.Linear(self.input_dim, self.hidden_dim),
                # nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(inplace=True),

                # 4
                nn.Linear(self.hidden_dim, 1)
            )
        elif nb_linear == 3:
            self.main = nn.Sequential(
                # 1
                nn.Linear(self.input_dim, self.hidden_dim),
                # nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(inplace=True),
                # 2
                nn.Linear(self.hidden_dim, self.hidden_dim),
                # nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(inplace=True),
                # 4
                nn.Linear(self.hidden_dim, 1)
            )
        else:
            self.main = nn.Sequential(
                # 1
                nn.Linear(self.input_dim, self.hidden_dim),
                # nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(inplace=True),

                # 2
                nn.Linear(self.hidden_dim, self.hidden_dim),
                # nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(inplace=True),

                # 3
                nn.Linear(self.hidden_dim, self.hidden_dim),
                # nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(inplace=True),

                # 4
                nn.Linear(self.hidden_dim, 1)
            )

    def forward(self, x):
        return self.main(x)


def compute_gp(critic, rep_1, rep_2, device):
    """
    alpha = torch.rand(BATCH_SIZE, 1)
    differences = fake_data - real_data
    interpolates = real_data + (alpha * differences)
    gradients = torch.autograd.grad(Discriminator(interpolates)[0], interpolates, create_graph=True)[0]
    gradients2 = torch.autograd.grad(Discriminator(real_data)[0], real_data, create_graph=True)[0] # check the gradient on real data points

    slopes = torch.sqrt(torch.sum(torch.square(gradients), dim=1))
    slopes2 = torch.sqrt(torch.sum(torch.square(gradients2), dim=1)) # L2 norm
    gradient_penalty = torch.mean((slopes - 1.)**2)
    disc_cost += LAMBDA * gradient_penalty
    """

    """
     Function for the computation of the gradient penalty
     critic is a NN
     rep1 and rep2 are the probability distributions passed into the critic (for us p(z_y, z_s) and p(z_y)p(z_s)
    """
    batch_size = rep_1.size(0)
    # Start by computing an interpolated distribution between our two distribution
    # We give random weight to the distribution - eps
    eps = torch.rand(batch_size, 1).to(device)
    eps = eps.expand_as(rep_1)
    interpolation = eps * rep_1 + (1 - eps) * rep_2

    # Pass the interpolation into the critic from which we want to retrieve and norm the gradients
    interp_logits = critic(interpolation)
    grad_outputs = torch.ones_like(interp_logits)

    # Computes and returns the sum of gradients of outputs with respect to the inputs.
    gradients = autograd.grad(
        outputs=interp_logits,
        inputs=interpolation,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, 1)
    return torch.mean((grad_norm - 1) ** 2)


# train the main model with adv loss
def train_epoch(model, iterator, args, epoch):
    epoch_loss = 0
    model.train()

    optimizer = model.optimizer
    criterion = model.criterion

    data_t0 = time.time()
    data_t, t = 0, 0

    for it, batch in enumerate(iterator):

        text = batch[0].squeeze()
        tags = batch[1].long().squeeze()
        p_tags = batch[2].float().squeeze()

        text = text.to(args.device)
        tags = tags.to(args.device)
        p_tags = p_tags.to(args.device)

        if args.encoder_architecture == "BERT":
            # Modify the inputs for BERT models
            mask = torch.stack(batch["attention_mask"]).float().squeeze().T
            mask = mask.to(args.device)
            text = (text, mask)

        if args.BT is not None and args.BT == "Reweighting":
            instance_weights = batch[3].float()
            instance_weights = instance_weights.to(args.device)

        if args.regression:
            regression_tags = batch[5].float().squeeze()
            regression_tags = regression_tags.to(args.device)

        data_t += (time.time() - data_t0)
        t0 = time.time()

        optimizer.zero_grad()
        # main model predictions
        if args.gated:
            predictions = model(text, p_tags)
        else:
            predictions = model(text)

        predictions = predictions if not args.regression else predictions.squeeze()

        # main tasks loss
        # add the weighted loss
        if args.BT is not None and args.BT == "Reweighting":
            loss = criterion(predictions, tags if not args.regression else regression_tags)
            loss = torch.mean(loss * instance_weights)
        else:
            loss = criterion(predictions, tags if not args.regression else regression_tags)

        if args.ARL:
            # loss = loss + args.ARL_loss.get_arl_loss(model, batch, predictions, args)
            loss = args.ARL_loss.get_arl_loss(model, batch, predictions, args)

        if args.adv_debiasing:
            # Update discriminator if needed
            if args.adv_update_frequency == "Batch":
                # Update the class-specific discriminator
                if args.adv_gated and (args.adv_gated_type == "Separate"):
                    for tmp_y in range(args.num_classes):
                        tmp_y_mask = list(torch.where(tags == tmp_y)[0].cpu().numpy())
                        if len(tmp_y_mask) > 0:
                            _batch = [i[tmp_y_mask] for i in batch]
                            args.discriminator[tmp_y].train_self_batch(model, _batch)
                else:
                    args.discriminator.train_self_batch(model, batch)

            # get hidden representations
            if args.gated:
                hs = model.hidden(text, p_tags)
            else:
                hs = model.hidden(text)

            # Get adv losses
            if args.adv_gated and (args.adv_gated_type == "Separate"):
                for tmp_y in range(args.num_classes):
                    tmp_y_mask = list(torch.where(tags == tmp_y)[0].cpu().numpy())
                    if len(tmp_y_mask) > 0:
                        tmp_y_adv_losses = args.discriminator[tmp_y].adv_loss(hs[tmp_y_mask], tags[tmp_y_mask],
                                                                              p_tags[tmp_y_mask])

                        for tmp_y_adv_loss in tmp_y_adv_losses:
                            loss = loss - (tmp_y_adv_loss / (args.adv_num_subDiscriminator * args.num_classes))
            else:
                adv_losses = args.discriminator.adv_loss(hs, tags, p_tags)

                for adv_loss in adv_losses:
                    loss = loss - (adv_loss / args.adv_num_subDiscriminator)

        if args.FCL:
            # get hidden representations
            if args.gated:
                hs = model.hidden(text, p_tags)
            else:
                hs = model.hidden(text)

            # update the loss with Fair Supervised Contrastive Loss
            fscl_loss = args.FairSCL(hs, tags, p_tags)
            loss = loss + fscl_loss

        if (args.DyBT is not None) and (args.DyBT == "GroupDifference"):
            loss = loss + args.group_difference_loss(
                predictions, tags, p_tags,
                regression_tags=None if not args.regression else regression_tags,
            )
        optimizer.zero_grad()
        loss.backward()

        # Zero gradients of the cls head
        if it % args.classification_head_update_frequency != 0:
            model.zero_cls_grad()

        optimizer.step()
        epoch_loss += loss.item()
        t += (time.time() - t0)
        data_t0 = time.time()

        if it % args.log_interval == 0:
            logging.info((
                'Epoch: {:4d} [{:7d}/{:7d} ({:2.0f}%)]\tLoss: {:.4f}\t Data Time: {:.2f}s\tTrain Time: {:.2f}s'
            ).format(
                epoch, it * args.batch_size, len(iterator.dataset),
                       100. * it / len(iterator), loss, data_t, t,
            ))
            data_t, t = 0, 0

            if (it != 0) and args.save_batch_results:
                (epoch_test_loss, test_preds, test_labels, test_private_labels) = eval_epoch(
                    model=model,
                    iterator=args.opt.test_generator,
                    args=args)

                (epoch_valid_loss, valid_preds, valid_labels, valid_private_labels) = eval_epoch(
                    model=model,
                    iterator=args.opt.dev_generator,
                    args=args)

                is_best = validation_is_best(
                    valid_preds, valid_labels, valid_private_labels,
                    model, epoch_valid_loss, selection_criterion="DTO",
                    performance_metric="accuracy", fairness_metric="TPR_GAP"
                )

                present_evaluation_scores(
                    valid_preds, valid_labels, valid_private_labels,
                    test_preds, test_labels, test_private_labels,
                    epoch=epoch + (it / len(iterator)), epochs_since_improvement=None, model=model,
                    epoch_valid_loss=None, is_best=is_best
                )

                model.train()

    return epoch_loss / len(iterator)


def train_epoch_w(model, iterator, args, epoch):
    epoch_loss = 0

    # model.train()
    optimizer = model.optimizer
    criterion = model.criterion

    data_t0 = time.time()
    data_t, t = 0, 0

    if args.dataset == "Bios_gender":
        nb_linear_critic = 3

        if args.adv_level == 'input':
            input_size = 768
        elif args.adv_level == 'last_hidden':
            input_size = 300
        else:
            input_size = args.num_classes

        critic = Critic(nb_linear_critic, input_dim=input_size)
        critic.to(args.device)
        optimizer_critic = torch.optim.RMSprop(critic.parameters(), lr=5e-5)
        torch.cuda.empty_cache()

        w_gp = 0
        w_ct = 0
        M = 0.1

        alpha = 1
        # beta = 0

        critic.train()
        model.eval()

        epoch_loss_critic = 0
        for ni in range(5):
            if args.batch_per_class:
                batch = next(iter(iterator))
                print(len(batch))
                text = batch[0][0].squeeze().to(args.device)
                tags = batch[1].long().squeeze()
                demonic_rep = batch[0][2].squeeze().to(args.device)
            else:
                batch = next(iter(iterator))
                text = batch[0].squeeze().to(args.device)
                demonic_rep = batch[6].squeeze().to(args.device)
                tags = batch[1].long().squeeze()
                # p_tags = batch[2].float().squeeze()

            optimizer_critic.zero_grad()

            z_y = model.hidden(text.to(args.device)).to(args.device)
            z_s = model.hidden(demonic_rep.to(args.device)).to(args.device)

            if args.eo_optimization:
                y_pred = torch.argmax(model(text), axis=1).cpu().numpy()
                true_lab = tags.cpu().numpy()[:, None].reshape(-1)

                true_prediction = np.where(y_pred == true_lab)[0]

                z_s = z_s[true_prediction]
                z_y = z_y[true_prediction]

                if len(true_prediction) <= 1:
                    break

            cat_dependant = torch.hstack((z_s, z_y)).to(args.device)
            cat_independant = torch.hstack((z_s, torch.vstack((z_y[1:], z_y[0])))).to(args.device)

            del z_y, z_s

            loss_dis = - (torch.mean(critic(cat_dependant)) - torch.mean(critic(cat_independant)))

            if w_gp > 0:
                # Compute the gradient
                gradient_penalty = w_gp * compute_gp(critic, cat_dependant, cat_independant, args.device)

                # Loss penalized with the gradient (minus because we maximize this)
                loss_dis = loss_dis - gradient_penalty

            del cat_dependant, cat_independant

            loss_dis.backward()

            optimizer_critic.step()

            if w_gp == 0:
                for pi in critic.parameters():
                    pi.data.clamp_(-0.01, 0.01)

            epoch_loss_critic += loss_dis.item()

        epoch_loss_critic = epoch_loss_critic / 5
        print('Loss critic : ', epoch_loss_critic)

        critic.eval()
    model.train()

    for it, batch in enumerate(iterator):

        """text = batch[0].squeeze()
        tags = batch[1].long().squeeze()
        p_tags = batch[2].float().squeeze()"""

        """if args.batch_per_class:
            text = batch[0][0].squeeze().to(args.device)
            demonic_rep = batch[0][2].squeeze().to(args.device)
            tags = batch[1].long().squeeze()
            p_tags = batch[0][1].float().squeeze()
        else:"""

        text = batch[0].squeeze().to(args.device)
        # demonic_rep = batch[6].squeeze().to(args.device)
        tags = batch[1].long().squeeze()
        p_tags = batch[2].float().squeeze()

        """if args.encoder_architecture != "Fixed":
            # Modify the inputs for models like BERT
            mask = torch.stack(batch["attention_mask"]).float().squeeze().T
            mask = mask.to(args.device)
            text = (text, mask)"""

        """if args.BT is not None and args.BT == "Reweighting":
            instance_weights = batch[3].float()
            instance_weights = instance_weights.to(args.device)"""

        """if args.regression:
            regression_tags = batch[5].float().squeeze()
            regression_tags = regression_tags.to(args.device)"""

        text = text.to(args.device)
        tags = tags.to(args.device)
        p_tags = p_tags.to(args.device)

        data_t += (time.time() - data_t0)
        t0 = time.time()

        optimizer.zero_grad()
        # main model predictions
        """if args.gated:
            predictions = model(text, p_tags)
        else:"""
        predictions = model(text)

        # print('Hidden representation : ', hs)

        # predictions = predictions if not args.regression else predictions.squeeze()

        # main tasks loss
        # add the weighted loss
        """if args.BT is not None and args.BT == "Reweighting":
            loss = criterion(predictions, tags)  # if not args.regression else regression_tags)
            loss = torch.mean(loss * instance_weights)
        else:"""

        # sys.exit(0)

        loss = criterion(predictions, tags)  # if not args.regression else regression_tags)

        if args.dataset == "Bios_gender":
            z_y = model.hidden(text.to(args.device)).to(args.device)
            z_s = model.hidden(demonic_rep.to(args.device)).to(args.device)

            if args.eo_optimization:
                y_pred = torch.argmax(predictions, axis=1).cpu().numpy()
                true_lab = tags.cpu().numpy()[:, None].reshape(-1)

                true_prediction = np.where(y_pred == true_lab)[0]

                z_s = z_s[true_prediction]
                z_y = z_y[true_prediction]

                if len(true_prediction) <= 1:
                    loss_dis = 0
                else:
                    cat_dependant = torch.hstack((z_s, z_y)).to(args.device)
                    cat_independant = torch.hstack((z_s, torch.vstack((z_y[1:], z_y[0])))).to(args.device)
                    del z_y, z_s
                    loss_dis = (torch.mean(critic(cat_dependant)) - torch.mean(critic(cat_independant)))
            else:
                cat_dependant = torch.hstack((z_s, z_y)).to(args.device)
                cat_independant = torch.hstack((z_s, torch.vstack((z_y[1:], z_y[0])))).to(args.device)
                del z_y, z_s
                loss_dis = (torch.mean(critic(cat_dependant)) - torch.mean(critic(cat_independant)))

            loss += args.beta * loss_dis

        # Simulating fairness without demographics in using KNN based labels
        """if args.knn_labels:
            p_tags = KNN_labels(
                criterion = criterion, tags = tags if not args.regression else regression_tags, 
                predictions = predictions, text = text, model = model, loss = loss, 
                p = args.knn_labels_p, k = args.knn_labels_k)
            batch = batch.copy()
            batch[2] = p_tags

            if args.UKNN_debiasing and (args.UKNN_lambda != 0):
                loss = loss + args.UKNN_loss(
                    predictions, tags, p_tags, 
                    regression_tags = None if not args.regression else regression_tags,
                )"""

        """if args.ARL:
            # loss = loss + args.ARL_loss.get_arl_loss(model, batch, predictions, args)
            loss = args.ARL_loss.get_arl_loss(model, batch, predictions, args)"""

        """if args.adv_debiasing:
            # Update discriminator if needed
            if args.adv_update_frequency == "Batch":
                # Update the class-specific discriminator
                if args.adv_gated and (args.adv_gated_type == "Separate"):
                    for tmp_y in range(args.num_classes):
                        tmp_y_mask = list(torch.where(tags == tmp_y)[0].cpu().numpy())
                        if len(tmp_y_mask) > 0:
                            _batch = [i[tmp_y_mask] for i in batch]
                            args.discriminator[tmp_y].train_self_batch(model, _batch)
                else:
                    args.discriminator.train_self_batch(model, batch)

            # get hidden representations
            if args.gated:
                hs = model.hidden(text, p_tags)
            else:
                hs = model.hidden(text)

            # Get adv losses
            if args.adv_gated and (args.adv_gated_type == "Separate"):
                for tmp_y in range(args.num_classes):
                    tmp_y_mask = list(torch.where(tags == tmp_y)[0].cpu().numpy())
                    if len(tmp_y_mask) > 0:
                        tmp_y_adv_losses = args.discriminator[tmp_y].adv_loss(hs[tmp_y_mask], tags[tmp_y_mask], p_tags[tmp_y_mask])

                        for tmp_y_adv_loss in tmp_y_adv_losses:
                            loss = loss - (tmp_y_adv_loss / (args.adv_num_subDiscriminator * args.num_classes))
            else:
                adv_losses = args.discriminator.adv_loss(hs, tags, p_tags)

                for adv_loss in adv_losses:
                    loss = loss - (adv_loss / args.adv_num_subDiscriminator)"""

        """if args.FCL:
            # get hidden representations
            if args.gated:
                hs = model.hidden(text, p_tags)
            else:
                hs = model.hidden(text)

            # update the loss with Fair Supervised Contrastive Loss
            fscl_loss = args.FairSCL(hs, tags, p_tags)
            loss = loss + fscl_loss"""

        """if (args.DyBT is not None) and (args.DyBT == "GroupDifference"):
            loss = loss + args.group_difference_loss(
                predictions, tags, p_tags, 
                regression_tags = None if not args.regression else regression_tags,
                )"""

        optimizer.zero_grad()
        loss.backward()

        # Zero gradients of the cls head
        if it % args.classification_head_update_frequency != 0:
            model.zero_cls_grad()

        optimizer.step()
        epoch_loss += loss.item()
        t += (time.time() - t0)
        data_t0 = time.time()

        if it % args.log_interval == 0:
            if args.dataset == "Bios_gender":
                logging.info((
                    'Epoch: {:4d} [{:7d}/{:7d} ({:2.0f}%)]\tLoss: {:.4f}\tLoss critic: {:.4f}\t Data Time: {:.2f}s\tTrain Time: {:.2f}s'
                ).format(
                    epoch, it * args.batch_size, len(iterator.dataset),
                           100. * it / len(iterator), loss, loss_dis, data_t, t,
                ))
            else:
                logging.info((
                    'Epoch: {:4d} [{:7d}/{:7d} ({:2.0f}%)]\tLoss: {:.4f}\t Data Time: {:.2f}s\tTrain Time: {:.2f}s'
                ).format(
                    epoch, it * args.batch_size, len(iterator.dataset),
                           100. * it / len(iterator), loss, data_t, t,
                ))
            data_t, t = 0, 0

            if (it != 0) and args.save_batch_results:
                (epoch_test_loss, test_preds, test_labels, test_private_labels) = eval_epoch(
                    model=model,
                    iterator=args.opt.test_generator,
                    args=args)

                (epoch_valid_loss, valid_preds, valid_labels, valid_private_labels) = eval_epoch(
                    model=model,
                    iterator=args.opt.dev_generator,
                    args=args)

                is_best = validation_is_best(
                    valid_preds, valid_labels, valid_private_labels,
                    model, epoch_valid_loss, selection_criterion="DTO",
                    performance_metric="accuracy", fairness_metric="TPR_GAP"
                )

                present_evaluation_scores(
                    valid_preds, valid_labels, valid_private_labels,
                    test_preds, test_labels, test_private_labels,
                    epoch=epoch + (it / len(iterator)), epochs_since_improvement=None, model=model,
                    epoch_valid_loss=None, is_best=is_best
                )

                if args.dataset == 'Bios_gender':
                    logging.info('Loss: {:.4f}\tLoss critic: {:.4f}'.format(loss, loss_dis))
                else:
                    logging.info('Loss: {:.4f}'.format(loss, ))

                model.train()

    return epoch_loss / len(iterator)


# to evaluate the main model
def eval_epoch(model, iterator, args, demon=None):
    epoch_loss = 0
    device = args.device

    model.eval()

    criterion = model.criterion

    preds = []
    labels = []
    private_labels = []
    demon_prediction = []

    for batch in iterator:

        text = batch[0]

        tags = batch[1]
        p_tags = batch[2]

        text = text.to(device)
        tags = tags.to(device).long()
        p_tags = p_tags.to(device).float()

        if args.encoder_architecture != "Fixed":
            # Modify the inputs for models like BERT
            mask = torch.stack(batch["attention_mask"]).float().squeeze().T
            mask = mask.to(args.device)
            text = (text, mask)

        if args.BT is not None and args.BT == "Reweighting":
            instance_weights = batch[3].float()
            instance_weights = instance_weights.to(device)

        if args.regression:
            regression_tags = batch[5].squeeze()
            regression_tags = regression_tags.to(args.device)

        # main model predictions
        if args.gated:
            predictions = model(text, p_tags)
        else:
            predictions = model(text)

        if demon is not None:
            demon_pred = demon(text)
            demon_pred = demon_pred.squeeze().detach().cpu()
            demon_prediction += list(torch.argmax(demon_pred, axis=1).numpy())

        predictions = predictions if not args.regression else predictions.squeeze()

        # add the weighted loss
        if args.BT is not None and args.BT == "Reweighting":
            loss = criterion(predictions, tags if not args.regression else regression_tags)
            loss = torch.mean(loss * instance_weights)
        else:
            loss = criterion(predictions, tags if not args.regression else regression_tags)

        epoch_loss += loss.item()

        predictions = predictions.detach().cpu()

        if args.regression:
            preds += list(predictions.numpy())
            tags = regression_tags.cpu().numpy()
        else:
            tags = tags.cpu().numpy()
            preds += list(torch.argmax(predictions, axis=1).numpy())
        labels += list(tags)

        private_labels += list(batch[2].cpu().numpy())

    if demon is not None:
        return (epoch_loss / len(iterator)), preds, labels, private_labels, demon_prediction
    else:
        return (epoch_loss / len(iterator)), preds, labels, private_labels



def train_epoch_critic(model, iterator, args, epoch):
    pass


def train_epoch_classif(model, iterator, args, epoch):
    pass


class BaseModel(nn.Module):

    def init_for_training(self):

        self.device = self.args.device
        self.to(self.device)

        self.learning_rate = self.args.lr
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
            weight_decay=self.args.weight_decay,
        )

        if self.args.BT and self.args.BT == "Reweighting":
            reduction = "none"
        else:
            reduction = "mean"

        if self.args.regression:
            self.criterion = torch.nn.MSELoss(reduction=reduction)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction)

        self.best_valid_loss = 1e+5

        print_network(self, verbose=True)

    def init_hyperparameters(self):
        if self.args.activation_function == "ReLu":
            self.AF = nn.ReLU()
        elif self.args.activation_function == "Tanh":
            self.AF = nn.Tanh()
        elif self.args.activation_function == "LeakyReLU":
            self.AF = nn.LeakyReLU()
        else:
            raise "not implemented yet"

        if self.args.batch_norm:
            self.BN = nn.BatchNorm1d(self.args.hidden_size)
        else:
            self.BN = None

        assert (self.args.dropout >= 0) and (self.args.dropout <= 1), "Probability must be in the range from 0 to 1"
        if self.args.dropout > 0:
            self.dropout = nn.Dropout(p=self.args.dropout)
        else:
            self.dropout = None

    def zero_cls_grad(self):
        """Clears the gradients of cls layers

        """
        for group in self.cls_parameter:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    def train_self(self, **opt_pairs):

        path = "results/dev/Bios_gender/" if self.args.dataset == 'Bios_gender' else "results/dev/Moji/"

        # Overwrite the arguments
        """dataloader_opt_keys = ["train_generator", "dev_generator", "test_generator"]
        _generators = {k: opt_pairs.get(k, None) for k in dataloader_opt_keys}

        print(_generators)

        self.args.opt.train_generator = _generators["train_generator"] if _generators[
                                                                              "train_generator"] is not None else self.args.opt.train_generator
        self.args.opt.dev_generator = _generators["dev_generator"] if _generators[
                                                                          "dev_generator"] is not None else self.args.opt.dev_generator
        self.args.opt.test_generator = _generators["test_generator"] if _generators[
                                                                            "test_generator"] is not None else self.args.opt.test_generator

        print(type(self.args.opt.train_generator.dataset), self.args.opt.train_generator.dataset)"""

        data_args = dummy_args(self.args.dataset, self.args.data_dir)  # , self.args.BT, self.args.BTObj)
        task_dataloader = dataloaders.loaders.name2loader(data_args)

        train_data = task_dataloader(args=data_args, split="train")  # , dataset_size=dataset_size)
        dev_data = task_dataloader(args=data_args, split="dev")  # , dataset_size=dataset_size)
        test_data = task_dataloader(args=data_args, split="test")  # , dataset_size=dataset_size)

        # print(dev_data.y)
        # print(dev_data.X)
        # print(dev_data.protected_label)
        # print(dev_data.demX)
        # sys.exit(0)

        """if self.args.batch_per_class:
            eval_dataloader_params = {
                'batch_size': self.args.batch_size,
                'shuffle': False}

            _train_data = []
            for i in range(len(train_data.y)):
                _train_data.append(
                    ([train_data.X[i], train_data.protected_label[i], train_data.demX[i]], train_data.y[i]))
            train_dataset = CustomDataset(_train_data)
            train_sampler = OneClassSampler(train_dataset, batch_size=self.args.batch_size)
            self.args.opt.train_generator = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler,
                                                                        shuffle=False)
            
            self.args.opt.dev_generator = torch.utils.data.DataLoader(dev_data, **eval_dataloader_params)
            self.args.opt.test_generator = torch.utils.data.DataLoader(test_data, **eval_dataloader_params)
        else:"""

        train_dataloader_params = {
            'batch_size': self.args.batch_size,
            'shuffle': True}

        eval_dataloader_params = {
            'batch_size': self.args.batch_size,
            'shuffle': True}
        self.args.opt.train_generator = torch.utils.data.DataLoader(train_data, **train_dataloader_params)
        self.args.opt.dev_generator = torch.utils.data.DataLoader(dev_data, **eval_dataloader_params)
        self.args.opt.test_generator = torch.utils.data.DataLoader(test_data, **eval_dataloader_params)

        # Reinit the train loader for FairBatch
        """if (self.args.DyBT is not None) and (self.args.DyBT in ["FairBatch", "GeneralizedFB"]):
            from .DyBT import init_sampler
            DyBT_sampler = init_sampler(self, self.args)
            # Replace the tran iterator with fairbatch version

            self.args.opt.train_generator = torch.utils.data.DataLoader(self.args.opt.train_generator.dataset,
                                                                        sampler=DyBT_sampler, num_workers=0)
            logging.info("Reinitialized DyBT sampler for dataloader")"""

        epochs_since_improvement = 0
        # best_valid_loss = 1e+5

        loss_critic_sv = []
        loss_classif_sv = []
        loss_global_sv = []

        acc_test = []
        gap_test = []
        leakage_test_bck = []

        for epoch in range(self.args.opt.epochs):

            # Early stopping
            """if epochs_since_improvement >= self.args.epochs_since_improvement:
                break"""

            # One epoch's training
            epoch_train_loss = train_epoch(
                model=self,
                iterator=self.args.opt.train_generator,
                args=self.args,
                epoch=epoch)

            # One epoch's validation
            (epoch_valid_loss, valid_preds,
             valid_labels, valid_private_labels) = eval_epoch(
                model=self,
                iterator=self.args.opt.dev_generator,
                args=self.args)

            # Update discriminator if needed
            if self.args.adv_debiasing and self.args.adv_update_frequency == "Epoch":
                self.args.discriminator.train_self(self)

            # Check if there was an improvement
            # is_best = epoch_valid_loss < best_valid_loss
            # best_valid_loss = min(epoch_valid_loss, best_valid_loss)
            is_best = validation_is_best(
                valid_preds, valid_labels, valid_private_labels,
                self, epoch_valid_loss, selection_criterion="DTO",
                performance_metric="accuracy", fairness_metric="TPR_GAP"
            )

            if not is_best:
                epochs_since_improvement += 1
                logging.info("Epochs since last improvement: %d" % (epochs_since_improvement,))
            else:
                epochs_since_improvement = 0

            if epoch % 10 == 0:
                logging.info("Evaluation at Epoch %d" % (epoch,))

                (epoch_test_loss, test_preds,
                 test_labels, test_private_labels) = eval_epoch(
                    model=self,
                    iterator=self.args.opt.test_generator,
                    args=self.args)

                present_evaluation_scores(
                    valid_preds, valid_labels, valid_private_labels,
                    test_preds, test_labels, test_private_labels,
                    epoch, epochs_since_improvement, self, epoch_valid_loss,
                    is_best,
                )

                test_scores, test_confusion_matrices = gap_eval_scores(
                    y_pred=test_preds,
                    y_true=test_labels,
                    protected_attribute=test_private_labels,
                    args=self.args,
                )

                test_leakage, dev_leakage = leakage_evaluation(self,
                                                               self.args.adv_level,
                                                               training_generator=self.args.opt.train_generator,
                                                               validation_generator=self.args.opt.dev_generator,
                                                               test_generator=self.args.opt.test_generator,
                                                               device=self.args.device,
                                                               augmentation=False
                                                               )

                acc_test.append(test_scores['accuracy'])
                gap_test.append([test_scores['TPR_GAP'], test_scores['FPR_GAP'], test_scores['PPR_GAP']])
                leakage_test_bck.append(test_leakage)

        base_name = path + self.args.exp_id
        np.save(base_name + '/loss_critic.npy', loss_critic_sv)
        np.save(base_name + '/loss_classif.npy', loss_classif_sv)
        np.save(base_name + '/loss.npy', loss_global_sv)

        np.save(base_name + '/acc_test.npy', acc_test)
        np.save(base_name + '/gap_test.npy', gap_test)
        np.save(base_name + '/leakage_test.npy', leakage_test_bck)


    def extract_hidden_representations(self, split):
        import numpy as np

        hidden = []
        labels = []
        private_labels = []
        regression_labels = []

        if split == "train":
            iterator = self.args.train_generator
        elif split == "dev":
            iterator = self.args.dev_generator
        elif split == "test":
            iterator = self.args.test_generator
        else:
            raise NotImplementedError

        for batch in iterator:

            text = batch[0].squeeze()
            tags = batch[1].squeeze()
            p_tags = batch[2].squeeze()

            labels += list(tags.cpu().numpy())
            private_labels += list(p_tags.cpu().numpy())

            text = text.to(self.args.device)
            tags = tags.to(self.args.device).long()
            p_tags = p_tags.to(self.args.device).float()

            if self.args.regression:
                regression_tags = batch[5].float().squeeze()
                regression_labels += list(regression_tags.cpu().numpy())
                regression_tags = regression_tags.to(self.args.device)

            # Extract hidden representations
            if self.args.gated:
                hidden_state = self.hidden(text, p_tags)
            else:
                hidden_state = self.hidden(text)

            hidden.append(hidden_state.detach().cpu().numpy())

        hidden = np.concatenate(hidden, 0)

        hidden = np.array(hidden)
        labels = np.array(labels)
        private_labels = np.array(private_labels)
        regression_labels = np.array(regression_labels) if self.args.regression else None

        return hidden, labels, private_labels, regression_labels

    def train_fair_wasserstein(self, mlp_critic, path_critic, **opt_pairs):
        if self.args.dataset == 'Moji':
            print(self.args.same_mlp)
            assert not self.args.same_mlp, 'The argument same_MLP must be set to False for Moji dataset'

        path = "results/dev/Bios_gender/" if self.args.dataset == 'Bios_gender' else "results/dev/Moji/"

        same_batch = self.args.same_batch
        same_mlp = self.args.same_mlp

        # Data processing
        data_args = dummy_args(self.args.dataset, self.args.data_dir)  # , self.args.BT, self.args.BTObj)
        task_dataloader = dataloaders.loaders.name2loader(data_args)

        # dataset_size = 0.1
        train_data = task_dataloader(args=data_args, split="train")  # , dataset_size=dataset_size)
        dev_data = task_dataloader(args=data_args, split="dev")  # , dataset_size=dataset_size)
        test_data = task_dataloader(args=data_args, split="test")  # , dataset_size=dataset_size)

        if same_batch:
            train_dataloader_params = {
                'batch_size': self.args.batch_size,
                'shuffle': True}

            eval_dataloader_params = {
                'batch_size': self.args.batch_size,
                'shuffle': True}
            self.args.opt.train_generator = torch.utils.data.DataLoader(train_data, **train_dataloader_params)
            self.args.opt.dev_generator = torch.utils.data.DataLoader(dev_data, **eval_dataloader_params)
            self.args.opt.test_generator = torch.utils.data.DataLoader(test_data, **eval_dataloader_params)
        else:
            # data for classifier
            train_dataloader_params = {
                'batch_size': self.args.batch_size,
                'shuffle': True}

            eval_dataloader_params = {
                'batch_size': self.args.batch_size,
                'shuffle': True}
            self.args.opt.train_generator_classif = torch.utils.data.DataLoader(train_data, **train_dataloader_params)
            self.args.opt.dev_generator = torch.utils.data.DataLoader(dev_data, **eval_dataloader_params)
            self.args.opt.test_generator = torch.utils.data.DataLoader(test_data, **eval_dataloader_params)

            # data for critic
            _train_data = []
            for i in range(len(train_data.y)):
                if self.args.dataset == 'Bios_gender':
                    if self.args.same_mlp:
                        _train_data.append(
                            ([train_data.X[i], train_data.protected_label[i], train_data.demX[i]], train_data.y[i]))
                    else:
                        _train_data.append(
                            ([train_data.X[i], train_data.protected_label[i], torch.empty(1)],
                             train_data.y[i]))
                else:
                    _train_data.append(
                        ([train_data.X[i], train_data.protected_label[i]], train_data.y[i]))

            train_dataset = CustomDataset(_train_data)
            train_sampler = OneClassSampler(train_dataset, batch_size=self.args.batch_size)
            self.args.opt.train_generator_critic = torch.utils.data.DataLoader(train_dataset,
                                                                               batch_sampler=train_sampler,
                                                                               shuffle=False)

        # Start training
        epochs_since_improvement = 0
        # best_valid_loss = 1e+5

        optimizer = self.optimizer
        criterion = self.criterion

        nb_linear_critic = 3

        if self.args.adv_level == 'input':
            input_size = 2 * 768
        elif self.args.adv_level == 'last_hidden':
            input_size = 2 * 300
        else:
            input_size = self.args.num_classes + 2

        print("Critic input dim : ", input_size)

        critic = Critic(nb_linear_critic, input_dim=input_size)
        critic.to(self.args.device)
        optimizer_critic = torch.optim.RMSprop(critic.parameters(), lr=self.args.lr_critic)  # 5e-5)
        torch.cuda.empty_cache()

        # if we use a different MLP for representation load MLP
        if not same_mlp:
            checkpoint = torch.load(path_critic)
            mlp_critic.load_state_dict(checkpoint['model_state_dict'])
            mlp_critic.eval()

        w_gp = self.args.w_gp

        data_t0 = time.time()
        data_t, t = 0, 0

        loss_critic_sv = []
        loss_classif_sv = []
        loss_global_sv = []

        acc_test = []
        gap_test = []
        acc_val = []
        gap_val = []
        leakage_test_bck = []
        leakage_dev_bck = []

        for epoch in range(10001):
            # Train critic
            epoch_loss = 0
            critic.train()
            self.eval()
            epoch_loss_critic = 0
            for ni in range(self.args.ni):
                if same_batch:
                    batch = next(iter(self.args.opt.train_generator))
                    text = batch[0].squeeze().to(self.args.device)
                    demonic_rep = None  # batch[6].squeeze().to(
                    # self.args.device) if self.args.dataset == 'Bios_gender' else None
                    tags = batch[1].long().squeeze()
                    # p_tags = batch[2].float().squeeze()
                else:
                    batch = next(iter(self.args.opt.train_generator_critic))
                    text = batch[0][0].squeeze().to(self.args.device)
                    demonic_rep = batch[0][2].squeeze().to(
                        self.args.device) if self.args.dataset == 'Bios_gender' else None
                    tags = batch[1].long().squeeze().to(self.args.device)
                    # p_tags_critic = batch_critic[0][1].float().squeeze().to(self.args.device)

                optimizer_critic.zero_grad()

                if same_mlp:
                    z_y = self.hidden(text.to(self.args.device), return_first_hidden=self.args.return_first_hidden).to(
                        self.args.device)
                    z_s = self.hidden(demonic_rep.to(self.args.device),
                                      return_first_hidden=self.args.return_first_hidden).to(self.args.device)
                else:
                    z_y = self.hidden(text.to(self.args.device), return_first_hidden=self.args.return_first_hidden).to(
                        self.args.device)
                    z_s = mlp_critic.hidden(text.to(self.args.device),
                                            return_first_hidden=self.args.return_first_hidden).to(self.args.device)

                if self.args.eo_optimization:
                    y_pred = torch.argmax(self(text), axis=1).cpu().numpy()
                    true_lab = tags.cpu().numpy()[:, None].reshape(-1)

                    true_prediction = np.where(y_pred == true_lab)[0]

                    z_s = z_s[true_prediction]
                    z_y = z_y[true_prediction]

                    if len(true_prediction) <= 1:
                        break

                if len(z_y) != len(z_s):
                    print('z_y and z_s have different shapes')
                    break

                cat_dependant = torch.hstack((z_s, z_y)).to(self.args.device)
                cat_independant = torch.hstack((z_s, torch.vstack((z_y[1:], z_y[0])))).to(self.args.device)

                del z_y, z_s

                if self.args.KL:
                    loss_dis = - (torch.mean(critic(cat_dependant)) - torch.log(
                        torch.mean(torch.exp(critic(cat_independant)))))
                else:
                    # print("Shapes :")
                    # print(cat_dependant.shape)
                    # print(cat_independant.shape)
                    loss_dis = - (torch.mean(critic(cat_dependant)) - torch.mean(critic(cat_independant)))

                if w_gp > 0:
                    # Compute the gradient
                    gradient_penalty = w_gp * compute_gp(critic, cat_dependant, cat_independant, self.args.device)
                    print('Gradient penalty done at epoch', epoch)
                    # Loss penalized with the gradient (minus because we maximize this)
                    loss_dis = loss_dis - gradient_penalty

                del cat_dependant, cat_independant

                loss_dis.backward()

                optimizer_critic.step()

                if self.args.KL:
                    pass
                else:
                    if w_gp == 0:
                        for pi in critic.parameters():
                            pi.data.clamp_(-self.args.clamp, self.args.clamp)  # -0.01, 0.01)

                epoch_loss_critic += loss_dis.item()

            # Train classifier
            epoch_loss_classif = 0
            epoch_loss_global = 0
            epoch_loss_critic = 0
            batch_dropout_proba = random.uniform(0, 1)
            for nj in range(self.args.nj):
                critic.eval()
                self.train()

                if same_batch:
                    batch = next(iter(self.args.opt.train_generator))
                    text = batch[0].squeeze().to(self.args.device)
                    demonic_rep = None  # batch[6].squeeze().to(
                    # self.args.device) if self.args.dataset == 'Bios_gender' else None
                    tags = batch[1].long().squeeze().to(self.args.device)
                else:
                    batch_classif = next(iter(self.args.opt.train_generator_classif))
                    text_classif = batch_classif[0].squeeze().to(self.args.device)
                    # demonic_rep_classif = batch_classif[6].squeeze().to(self.args.device)
                    tags_classif = batch_classif[1].long().squeeze().to(self.args.device)
                    # p_tags_classif = batch_classif[2].float().squeeze().to(self.args.device)

                    batch_critic = next(iter(self.args.opt.train_generator_critic))
                    text_critic = batch_critic[0][0].squeeze().to(self.args.device)
                    demonic_rep_critic = batch_critic[0][2].squeeze().to(
                        self.args.device) if self.args.dataset == 'Bios_gender' else None
                    tags_critic = batch_critic[1].long().squeeze().to(self.args.device)
                    p_tags_critic = batch_critic[0][1].float().squeeze().to(self.args.device)

                data_t += (time.time() - data_t0)
                t0 = time.time()

                optimizer.zero_grad()
                # main model predictions
                if same_batch:
                    predictions = self(text)
                    loss_classif = criterion(predictions, tags)  # alpha *
                else:
                    predictions = self(text_classif)
                    loss_classif = criterion(predictions, tags_classif)  # alpha *

                epoch_loss_classif += loss_classif.item()

                if batch_dropout_proba < self.args.dropout_w:
                    if same_mlp:
                        if same_batch:
                            z_y = self.hidden(text.to(self.args.device),
                                              return_first_hidden=self.args.return_first_hidden).to(self.args.device)
                            z_s = self.hidden(demonic_rep.to(self.args.device),
                                              return_first_hidden=self.args.return_first_hidden).to(self.args.device)
                        else:
                            z_y = self.hidden(text_critic.to(self.args.device),
                                              return_first_hidden=self.args.return_first_hidden).to(self.args.device)
                            z_s = self.hidden(demonic_rep_critic.to(self.args.device),
                                              return_first_hidden=self.args.return_first_hidden).to(self.args.device)
                    else:
                        if same_batch:
                            z_y = self.hidden(text.to(self.args.device),
                                              return_first_hidden=self.args.return_first_hidden).to(self.args.device)
                            z_s = mlp_critic.hidden(text.to(self.args.device),
                                                    return_first_hidden=self.args.return_first_hidden).to(
                                self.args.device)
                        else:
                            z_y = self.hidden(text_critic.to(self.args.device),
                                              return_first_hidden=self.args.return_first_hidden).to(self.args.device)
                            z_s = mlp_critic.hidden(text_critic.to(self.args.device),
                                                    return_first_hidden=self.args.return_first_hidden).to(
                                self.args.device)

                    if len(z_y) != len(z_s):
                        print('z_y and z_s have different shapes')
                        break

                    regularization_term = True
                    if self.args.eo_optimization:
                        if same_batch:
                            y_pred = torch.argmax(predictions, axis=1).cpu().numpy()
                        else:
                            y_pred = torch.argmax(self(text_critic), axis=1).cpu().numpy()

                        if same_batch:
                            true_lab = tags.cpu().numpy()[:, None].reshape(-1)
                        else:
                            true_lab = tags_critic.cpu().numpy()[:, None].reshape(-1)

                        true_prediction = np.where(y_pred == true_lab)[0]

                        z_s = z_s[true_prediction]
                        z_y = z_y[true_prediction]

                        if len(true_prediction) <= 1:
                            regularization_term = False
                        else:
                            cat_dependant = torch.hstack((z_s, z_y)).to(self.args.device)
                            cat_independant = torch.hstack((z_s, torch.vstack((z_y[1:], z_y[0])))).to(self.args.device)
                            del z_y, z_s
                            if self.args.KL:
                                loss_dis = - (torch.mean(critic(cat_dependant)) - torch.log(
                                    torch.mean(torch.exp(critic(cat_independant)))))
                            else:
                                loss_dis = (torch.mean(critic(cat_dependant)) - torch.mean(critic(cat_independant)))
                    else:
                        cat_dependant = torch.hstack((z_s, z_y)).to(self.args.device)
                        cat_independant = torch.hstack((z_s, torch.vstack((z_y[1:], z_y[0])))).to(self.args.device)
                        del z_y, z_s
                        if self.args.KL:
                            loss_dis = - (torch.mean(critic(cat_dependant)) - torch.log(
                                torch.mean(torch.exp(critic(cat_independant)))))
                        else:
                            loss_dis = (torch.mean(critic(cat_dependant)) - torch.mean(critic(cat_independant)))
                    if regularization_term:
                        if self.args.beta_auto == 'auto1':
                            alpha = (self.args.beta * loss_dis.detach()) / loss_classif.detach()
                            loss = alpha * loss_classif + loss_dis
                        elif self.args.beta_auto == 'auto2':
                            alpha = (loss_dis.detach() - loss_classif.detach()).T * loss_dis.detach() / \
                                    LA.norm(loss_classif.detach() - loss_dis.detach())
                            loss = alpha * loss_classif + (1 - alpha) * loss_dis
                        else:
                            loss = self.args.alpha * loss_classif + self.args.beta * loss_dis
                        epoch_loss_critic += loss_dis.item()
                    else:
                        loss = loss_classif
                else:
                    loss = loss_classif

                epoch_loss_global += loss.item()

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                epoch_loss += loss.item()
                t += (time.time() - t0)
                data_t0 = time.time()

            loss_critic_sv.append(epoch_loss_critic / self.args.ni)
            loss_classif_sv.append(epoch_loss_classif / self.args.nj)
            loss_global_sv.append(epoch_loss_global / self.args.nj)

            self.eval()

            # Validation
            # One epoch's validation
            (epoch_valid_loss, valid_preds,
             valid_labels, valid_private_labels) = eval_epoch(
                model=self,
                iterator=self.args.opt.dev_generator,
                args=self.args)

            # Check if there was an improvement
            # is_best = epoch_valid_loss < best_valid_loss
            # best_valid_loss = min(epoch_valid_loss, best_valid_loss)
            is_best = validation_is_best(
                valid_preds, valid_labels, valid_private_labels,
                self, epoch_valid_loss, selection_criterion="DTO",
                performance_metric="accuracy", fairness_metric="TPR_GAP"
            )

            if not is_best:
                epochs_since_improvement += 1
                logging.info("Epochs since last improvement: %d" % (epochs_since_improvement,))
            else:
                epochs_since_improvement = 0

            if epoch % 50 == 0:  # self.args.checkpoint_interval
                logging.info("Evaluation at Epoch %d" % (epoch,))

                (epoch_test_loss, test_preds,
                 test_labels, test_private_labels) = eval_epoch(
                    model=self,
                    iterator=self.args.opt.test_generator,
                    args=self.args)

                present_evaluation_scores(
                    valid_preds, valid_labels, valid_private_labels,
                    test_preds, test_labels, test_private_labels,
                    epoch, epochs_since_improvement, self, epoch_valid_loss,
                    is_best,
                )

                valid_scores, valid_confusion_matrices = gap_eval_scores(
                    y_pred=valid_preds,
                    y_true=valid_labels,
                    protected_attribute=valid_private_labels,
                    args=self.args,
                )

                test_scores, test_confusion_matrices = gap_eval_scores(
                    y_pred=test_preds,
                    y_true=test_labels,
                    protected_attribute=test_private_labels,
                    args=self.args,
                )

                test_leakage, dev_leakage = leakage_evaluation(self,
                                                               self.args.adv_level,
                                                               training_generator=self.args.opt.train_generator,
                                                               validation_generator=self.args.opt.dev_generator,
                                                               test_generator=self.args.opt.test_generator,
                                                               device=self.args.device,
                                                               augmentation=False
                                                               )

                print('Epoch : ', epoch,
                      ' accuracy: {:.4f}\tTPR_GAP: {:.4f}\tFPR_GAP: {:.4f}\tPPR_GAP: {:.4f}\tLeakage: {:.4f}'
                      .format(test_scores['accuracy'],
                              test_scores['TPR_GAP'],
                              test_scores['FPR_GAP'],
                              test_scores['PPR_GAP'],
                              test_leakage))

                file = path + self.args.exp_id + "/intermediate_results.txt"
                with open(file, "a") as f:
                    print('Epoch : ', epoch,
                          ' accuracy: {:.4f}\tTPR_GAP: {:.4f}\tFPR_GAP: {:.4f}\tPPR_GAP: {:.4f}\tLeakage: {:.4f}'
                          .format(test_scores['accuracy'],
                                  test_scores['TPR_GAP'],
                                  test_scores['FPR_GAP'],
                                  test_scores['PPR_GAP'],
                                  test_leakage), file=f)

                acc_test.append(test_scores['accuracy'])
                gap_test.append([test_scores['TPR_GAP'], test_scores['FPR_GAP'], test_scores['PPR_GAP']])
                acc_val.append(valid_scores['accuracy'])
                gap_val.append([valid_scores['TPR_GAP'], valid_scores['FPR_GAP'], valid_scores['PPR_GAP']])
                leakage_test_bck.append(test_leakage)
                leakage_dev_bck.append(dev_leakage)

        base_name = path + self.args.exp_id
        np.save(base_name + '/loss_critic.npy', loss_critic_sv)
        np.save(base_name + '/loss_classif.npy', loss_classif_sv)
        np.save(base_name + '/loss.npy', loss_global_sv)

        np.save(base_name + '/acc_test.npy', acc_test)
        np.save(base_name + '/gap_test.npy', gap_test)
        np.save(base_name + '/leakage_test.npy', leakage_test_bck)
        np.save(base_name + '/acc_val.npy', acc_val)
        np.save(base_name + '/gap_val.npy', gap_val)
        np.save(base_name + '/leakage_dev.npy', leakage_dev_bck)

