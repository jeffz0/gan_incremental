import matplotlib.pyplot as plt
import argparse
import copy
import random
import time
import pdb
import os, sys
import json
from models.base import GAN

from utils import visualize_featmap as visualize, adversarial_loss, visualize_graph, print_statement, return_statement, visualize_featmap_classes

import numpy as np
import factory
from torch.autograd import Variable

import torch
import torch.utils.model_zoo as model_zoo
from torch import nn, optim
from torch.nn import functional as F
import torchvision


class classification_only(GAN):
    """Implementation of Learning without Forgetting.

    :param args: An argparse parsed arguments object.
    """

    def __init__(self, args, disc, disc2, gen):
        super().__init__()
        self.args = args
        self.disc = disc

    # ----------
    # Public API
    # ----------

    def _train(self, train_loader, test_loader, output_dir, train_output_dir, test_output_dir, model_output_dir):
        args = self.args
        disc = self.disc
        
        train_losses = {}
        test_losses = {}
        
        if args.optimizer == 'sgd':
            optimizer_d = optim.SGD(disc.parameters(), lr=args.lr, momentum=0.9)
        elif args.optimizer == 'adam':
            optimizer_d = optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))
        
        for epoch in range(args.epochs_cls):
            self.disc.train()
            train_loss = self._train_cls(self.args, epoch+1, disc, train_loader, optimizer_d)
            train_losses[epoch] = train_loss
            # Frequency to output and visualize results
            if (epoch+1) % args.visualize_freq == 0 or epoch == 0:
                test_loss = self._test_2(args, epoch+1, disc, test_loader, test_output_dir)
                test_losses[epoch] = test_loss

                torch.save(disc.state_dict(), model_output_dir + 'disc_cls.pth')
                print("Saved model")

                with open(output_dir +'/train_losses.txt', 'w') as f:
                    json.dump(train_losses, f)
                with open(output_dir +'/test_losses.txt', 'w') as f:
                    json.dump(test_losses, f)
                    

    def _train_cls(self, args, epoch, disc, train_loader, optimizer_d):
        cls_criterion = nn.CrossEntropyLoss()
        _loss_g, _loss_cls_gen, _loss_adv_gen = 0., 0., 0.
        _loss_d, _loss_cls, _loss_adv = 0., 0., 0.
        ypred, ypred_gen = [], []
        ytrue, ytrue_gen = [], []
    #     gen_loss_lst = []
        for i, (inputs, featmaps, targets, indexes) in enumerate(train_loader):
            inputs, featmaps, targets = inputs.to(args.device), featmaps.to(args.device), targets.to(args.device)
            # Optimize Discriminator
            loss = 0
            optimizer_d.zero_grad()
#             featmaps, feats, logits_cls = disc(inputs)
            feats, logits_cls, logits_adv = disc(featmaps)
            loss_cls = cls_criterion(logits_cls, targets.long())
            _loss_cls += loss_cls.item()
            loss = loss_cls.clone()

            preds = F.softmax(logits_cls, dim=1).argmax(dim=1).cpu().numpy().tolist()
            ypred.extend(preds)
            ytrue.extend(targets)

            loss.backward()
            optimizer_d.step()
  
        acc = round((np.array(ypred) == np.array(ytrue)).sum() / len(ytrue), 4)  
        acc_gen = round((np.array(ypred_gen) == np.array(ytrue_gen)).sum() / len(ytrue_gen), 4)  
        if epoch % args.visualize_freq == 0 or epoch == 1:
            print_statement(epoch, i, acc, acc_gen, _loss_cls, _loss_cls_gen, _loss_adv, _loss_adv_gen)
        return return_statement(i, acc, acc_gen, _loss_cls, _loss_cls_gen, _loss_adv, _loss_adv_gen)

   
    def _test_2(self, args, epoch, disc, test_loader, test_output_dir):
        mse_criterion = nn.MSELoss()
        cls_criterion = nn.CrossEntropyLoss()
        _loss, _loss_cls, _loss_cls_gen = 0., 0., 0.
        ypred, ypred_gen = [], []
        ytrue, ytrue_gen = [], []
        for i, (inputs, featmaps, targets, indexes) in enumerate(test_loader):
            loss = 0
            inputs, featmaps, targets = inputs.to(args.device), featmaps.to(args.device), targets.to(args.device)

#             featmaps, feats, logits_cls = disc(inputs)
            feats, logits_cls, logits_adv = disc(featmaps)
            loss_cls = cls_criterion(logits_cls, targets.long())
            loss = loss_cls
            _loss_cls += loss_cls.item()
            preds = F.softmax(logits_cls, dim=1).argmax(dim=1).cpu().numpy().tolist()
            ypred.extend(preds)
            ytrue.extend(targets)

            _loss += loss.item()

        acc = round((np.array(ypred) == np.array(ytrue)).sum() / len(ytrue), 4)  
        acc_gen = round((np.array(ypred_gen) == np.array(ytrue_gen)).sum() / len(ytrue), 4)  

        print("Test 2 Set Epoch {}, Training Iteration {}".format(epoch, i))
        print("Accuracy: {}, Accuracy gen: {}".format(acc, acc_gen))
        print("Loss: {}, Loss_cls: {}, Loss_cls_gen: {}"
              .format(_loss/(i+1),_loss_cls/(i+1),_loss_cls_gen/(i+1)))
        return {"Test_acc": acc,"Test_acc_gen": acc_gen,"Loss":_loss/(i+1), "Loss_cls":_loss_cls/(i+1), "Loss_cls_gen":_loss_cls_gen/(i+1)}
