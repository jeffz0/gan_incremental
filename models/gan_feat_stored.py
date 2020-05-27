import matplotlib.pyplot as plt
import argparse
import copy
import random
import time
import pdb
import os, sys
import json
from models.base import GAN

from utils import visualize_featmap as visualize, adversarial_loss, visualize_graph, print_statement, return_statement

import numpy as np
import factory
from torch.autograd import Variable

import torch
import torch.utils.model_zoo as model_zoo
from torch import nn, optim
from torch.nn import functional as F
import torchvision


class gan_feat_stored(GAN):
    """Implementation of Learning without Forgetting.

    :param args: An argparse parsed arguments object.
    """

    def __init__(self, args, disc, disc2, gen):
        super().__init__()
        self.args = args
        self.disc = disc
        self.disc2 = disc2
        self.gen = gen

    # ----------
    # Public API
    # ----------

    def _train(self, train_loader, test_loader, output_dir, train_output_dir, test_output_dir, model_output_dir):
        args = self.args
        disc = self.disc
        disc2 = self.disc2
        gen = self.gen
        
        train_losses = {}
        test_losses = {}
        
        if args.optimizer == 'sgd':
            optimizer_g = optim.SGD(gen.parameters(), lr=args.lr/args.gen_lr_factor, momentum=0.9)
            optimizer_d = optim.SGD(disc.parameters(), lr=args.lr, momentum=0.9)
            optimizer_d2 = optim.SGD(disc2.parameters(), lr=args.lr, momentum=0.9)
        elif args.optimizer == 'adam':
            optimizer_g = optim.Adam(gen.parameters(), lr=args.lr/args.gen_lr_factor, betas=(0.5, 0.999))
            optimizer_d = optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))
            optimizer_d2 = optim.Adam(disc2.parameters(), lr=args.lr, betas=(0.5, 0.999))
        
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
                    
        self._build_statistics(args, train_loader, disc)
        
        for epoch in range(args.epochs_cls, args.epochs_cls+args.epochs):
            self.disc.train()
            self.gen.train()
            
            if args.joint:
                # Training discriminator and generator with same batch
                train_loss = self._train_joint(self.args, epoch+1, disc, gen, train_loader, optimizer_d, optimizer_g, train_output_dir)
            else:
                # Training discriminator for one epoch, then generator for one epoch
                train_loss = self._train_alt(self.args, epoch+1, disc, gen, train_loader, optimizer_d, optimizer_g, train_output_dir)
            train_losses[epoch] = train_loss
            
            disc.eval()
            gen.eval()
            
            # Frequency to output and visualize results
            if (epoch+1) % args.visualize_freq == 0 or epoch == args.epochs_cls:
                test_loss = self._test(args, epoch+1, disc, gen, test_loader, test_output_dir)
                test_losses[epoch] = test_loss

                torch.save(disc.state_dict(), model_output_dir + 'disc.pth')
                torch.save(gen.state_dict(), model_output_dir + 'gen.pth')
                print("Saved model")

                with open(output_dir +'/train_losses.txt', 'w') as f:
                    json.dump(train_losses, f)
                with open(output_dir +'/test_losses.txt', 'w') as f:
                    json.dump(test_losses, f)
                    
                visualize_graph(train_losses, epoch+1, output_dir)
                
        # Validate how well training on generated feature maps work on real images
        if args.phase_2:
            disc.eval()
            gen.eval()
            for epoch in range(args.epochs_cls+args.epochs, args.epochs_cls+args.epochs+args.epochs_2):
                disc2.train()
                train_loss = self._train_2(args, epoch+1, disc2, gen, train_loader, optimizer_d2, train_output_dir)
                train_losses[epoch] = train_loss
                disc2.eval()
                if (epoch+1) % args.visualize_freq == 0 or epoch == args.epochs_cls+args.epochs:
                    test_loss = self._test_2(args, epoch+1, disc2, test_loader, test_output_dir)
                    test_losses[epoch] = test_loss

                    torch.save(disc2.state_dict(), model_output_dir + 'disc2.pth')
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
        for i, (inputs, featmaps, targets) in enumerate(train_loader):
            inputs, featmaps, targets = inputs.to(args.device), featmaps.to(args.device), targets.to(args.device)
            # Optimize Discriminator
            loss = 0
            optimizer_d.zero_grad()
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

    def _train_alt(self, args, epoch, disc, gen, train_loader, optimizer_d, optimizer_g, train_output_dir):
        cls_criterion = nn.CrossEntropyLoss()
        _loss_g, _loss_cls_gen, _loss_adv_gen = 0., 0., 0.
        _loss_d, _loss_cls, _loss_adv = 0., 0., 0.
        ypred, ypred_gen = [], []
        ytrue, ytrue_gen = [], []
    #     gen_loss_lst = []
        for i, (inputs, featmaps, targets) in enumerate(train_loader):
            inputs, featmaps, targets = inputs.to(args.device), featmaps.to(args.device), targets.to(args.device)
            # Optimize Discriminator
            loss = 0
            optimizer_d.zero_grad()
            feats, logits_cls, logits_adv = disc(featmaps)
            loss_cls = cls_criterion(logits_cls, targets.long())
            _loss_cls += loss_cls.item()
            loss = loss_cls.clone()
            
            if args.adv:
                feats, gen_targets = self._sample_vecs(inputs.shape[0])
                feats = feats.to(args.device)
                gen_image = gen(feats.unsqueeze(2).unsqueeze(3).detach())
                feats_gen, logits_cls_gen, logits_adv_gen = disc(gen_image)
                loss_adv = (adversarial_loss(logits_adv, is_real=True, is_disc=True, type_=args.adv_type) + adversarial_loss(logits_adv_gen, is_real=False, is_disc=True, type_=args.adv_type))
                _loss_adv += loss_adv.item()
                loss += args.adv_w*loss_adv.clone()/2.

            preds = F.softmax(logits_cls, dim=1).argmax(dim=1).cpu().numpy().tolist()
            ypred.extend(preds)
            ytrue.extend(targets)

            loss.backward()
            optimizer_d.step()

        disc.eval()

        for i, (inputs, featmaps, targets) in enumerate(train_loader):
            inputs, featmaps, targets = inputs.to(args.device), featmaps.to(args.device), targets.to(args.device)
            loss = 0
            optimizer_g.zero_grad()

            # Optimize Generator
            feats, gen_targets = self._sample_vecs(inputs.shape[0])
            feats, gen_targets = feats.to(args.device), gen_targets.to(args.device)
            gen_image = gen(feats.unsqueeze(2).unsqueeze(3).detach())

            feats_gen, logits_cls_gen, logits_adv_gen = disc(gen_image)

            loss_cls_gen = cls_criterion(logits_cls_gen, gen_targets.long())
            _loss_cls_gen += loss_cls_gen.item() 
            loss = args.cls_w*loss_cls_gen.clone()

            if args.adv:
                loss_adv_gen = adversarial_loss(logits_adv_gen, is_real=True, is_disc=False, type_=args.adv_type)
                _loss_adv_gen += loss_adv_gen.item()
                loss += args.adv_w*loss_adv_gen.clone()

            preds_gen = F.softmax(logits_cls_gen, dim=1).argmax(dim=1).cpu().numpy().tolist()
            ypred_gen.extend(preds_gen)
            ytrue_gen.extend(gen_targets)

            loss.backward()
            optimizer_g.step()

        disc.train()    
        acc = round((np.array(ypred) == np.array(ytrue)).sum() / len(ytrue), 4)  
        acc_gen = round((np.array(ypred_gen) == np.array(ytrue_gen)).sum() / len(ytrue_gen), 4)  
        if epoch % args.visualize_freq == 0 or epoch == 1:
            print_statement(epoch, i, acc, acc_gen, _loss_cls, _loss_cls_gen, _loss_adv, _loss_adv_gen)
            visualize(featmaps[0], gen_image[0],
                      out_dir = train_output_dir + str(epoch) + "_" + str(i) + ".jpg")
        return return_statement(i, acc, acc_gen, _loss_cls, _loss_cls_gen, _loss_adv, _loss_adv_gen)

    def _train_joint(self, args, epoch, disc, gen, train_loader, optimizer_d, optimizer_g, train_output_dir):
        cls_criterion = nn.CrossEntropyLoss()
        _loss_g, _loss_cls_gen, _loss_adv_gen = 0., 0., 0.
        _loss_d, _loss_cls, _loss_adv = 0., 0., 0.
        ypred, ypred_gen = [], []
        ytrue, ytrue_gen = [], []
    #     gen_loss_lst = []
        for i, (inputs, featmaps, targets) in enumerate(train_loader):
            inputs, featmaps, targets = inputs.to(args.device), featmaps.to(args.device), targets.to(args.device)
            # Optimize Discriminator
            loss = 0
            optimizer_d.zero_grad()
            feats, logits_cls, logits_adv = disc(featmaps)
            loss_cls = cls_criterion(logits_cls, targets.long())

            _loss_cls += loss_cls.item()
            loss = loss_cls.clone()
            if args.adv:
                feats, gen_targets = self._sample_vecs(inputs.shape[0])
                feats = feats.to(args.device)
                gen_image = gen(feats.unsqueeze(2).unsqueeze(3).detach())

                feats_gen, logits_cls_gen, logits_adv_gen = disc(gen_image)

                loss_adv = (adversarial_loss(logits_adv, is_real=True, is_disc=True, type_=args.adv_type) + adversarial_loss(logits_adv_gen, is_real=False, is_disc=True, type_=args.adv_type))
                _loss_adv += loss_adv.item()
                loss += args.adv_w*loss_adv.clone()/2.

            preds = F.softmax(logits_cls, dim=1).argmax(dim=1).cpu().numpy().tolist()
            ypred.extend(preds)
            ytrue.extend(targets)

            loss.backward()
            optimizer_d.step()

            disc.eval()

            loss = 0
            optimizer_g.zero_grad()

            # Optimize Generator
            feats, gen_targets = self._sample_vecs(inputs.shape[0])
            feats, gen_targets = feats.to(args.device), gen_targets.to(args.device)
            gen_image = gen(feats.unsqueeze(2).unsqueeze(3).detach())

            feats_gen, logits_cls_gen, logits_adv_gen = disc(gen_image)

            loss_cls_gen = cls_criterion(logits_cls_gen, gen_targets.long())
            _loss_cls_gen += loss_cls_gen.item() 
            loss = args.cls_w*loss_cls_gen.clone()

            if args.adv:
                loss_adv_gen = adversarial_loss(logits_adv_gen, is_real=True, is_disc=False, type_=args.adv_type)
                _loss_adv_gen += loss_adv_gen.item()
                loss += args.adv_w*loss_adv_gen.clone()

            preds_gen = F.softmax(logits_cls_gen, dim=1).argmax(dim=1).cpu().numpy().tolist()
            ypred_gen.extend(preds_gen)
            ytrue_gen.extend(gen_targets)

            loss.backward()
            optimizer_g.step()

            disc.train()    

        acc = round((np.array(ypred) == np.array(ytrue)).sum() / len(ytrue), 4)  
        acc_gen = round((np.array(ypred_gen) == np.array(ytrue_gen)).sum() / len(ytrue_gen), 4)  
        
        if epoch % args.visualize_freq == 0 or epoch == 1:
            print_statement(epoch, i, acc, acc_gen, _loss_cls, _loss_cls_gen, _loss_adv, _loss_adv_gen)
            visualize(featmaps[0], gen_image[0],
                      out_dir = train_output_dir + str(epoch) + "_" + str(i) + ".jpg")
        return return_statement(i, acc, acc_gen, _loss_cls, _loss_cls_gen, _loss_adv, _loss_adv_gen)



    def _test(self, args, epoch, disc, gen, test_loader, test_output_dir):
        mse_criterion = nn.MSELoss()
        cls_criterion = nn.CrossEntropyLoss()
        _loss_g, _loss_cls_gen, _loss_adv_gen = 0., 0., 0.
        _loss_d, _loss_cls, _loss_adv = 0., 0., 0.
        _loss = 0.
        ypred, ypred_gen = [], []
        ytrue, ytrue_gen = [], []
        for i, (inputs, featmaps, targets) in enumerate(test_loader):
            loss = 0
            inputs, featmaps, targets = inputs.to(args.device), featmaps.to(args.device), targets.to(args.device)

            feats, logits_cls, logits_adv = disc(featmaps)
            loss_cls = cls_criterion(logits_cls, targets.long())
            loss = loss_cls
            _loss_cls += loss_cls.item()

            preds = F.softmax(logits_cls, dim=1).argmax(dim=1).cpu().numpy().tolist()
            ypred.extend(preds)
            ytrue.extend(targets)
            
            feats, gen_targets = self._sample_vecs(inputs.shape[0])
            feats, gen_targets = feats.to(args.device), gen_targets.to(args.device)
            gen_image = gen(feats.unsqueeze(2).unsqueeze(3).detach())
            feats_gen, logits_cls_gen, logits_adv_gen = disc(gen_image)
            loss_cls_gen = cls_criterion(logits_cls_gen, gen_targets.long())
            loss += args.cls_w*loss_cls_gen
            _loss_cls_gen += loss_cls_gen.item()   

            if args.adv:
                loss_adv = (adversarial_loss(logits_adv, is_real=True, is_disc=True, type_=args.adv_type) + adversarial_loss(logits_adv_gen, is_real=False, is_disc=True, type_=args.adv_type))
                _loss_adv += loss_adv.item()
                loss += args.adv_w*loss_adv.clone()/2.


                loss_adv_gen = adversarial_loss(logits_adv_gen, is_real=True, is_disc=False, type_=args.adv_type)
                _loss_adv_gen += loss_adv_gen.item()
                loss += args.adv_w*loss_adv_gen.clone()
            preds_gen = F.softmax(logits_cls_gen, dim=1).argmax(dim=1).cpu().numpy().tolist()
            ypred_gen.extend(preds_gen)
            ytrue_gen.extend(gen_targets)
            _loss += loss.item()

            if i%10 == 0:
                visualize(featmaps[0], gen_image[0],
                          out_dir = test_output_dir + str(epoch) + "_" + str(i) + ".jpg")

        acc = round((np.array(ypred) == np.array(ytrue)).sum() / len(ytrue), 4)  
        acc_gen = round((np.array(ypred_gen) == np.array(ytrue_gen)).sum() / len(ytrue_gen), 4)  

        print("Test Set Epoch {}, Training Iteration {}".format(epoch, i))
        print("Accuracy: {}, Accuracy gen: {}".format(acc, acc_gen))
        print("Loss: {}, Loss_cls: {}, Loss_cls_gen: {}"
              .format(_loss/(i+1),_loss_cls/(i+1),_loss_cls_gen/(i+1)))
        if args.adv:
            print("Loss_adv: {}, Loss_adv_gen: {}"
                  .format(_loss_adv/(i+1),_loss_adv_gen/(i+1)))
        return return_statement(i, acc, acc_gen, _loss_cls, _loss_cls_gen, _loss_adv, _loss_adv_gen)

    def _train_2(self, args, epoch, disc, gen, train_loader, optimizer_d, train_output_dir):
        cls_criterion = nn.CrossEntropyLoss()
        _loss_g, _loss_cls_gen, _loss_adv_gen = 0., 0., 0.
        _loss_d, _loss_cls, _loss_adv = 0., 0., 0.
        ypred, ypred_gen = [], []
        ytrue, ytrue_gen = [], []
    #     gen_loss_lst = []
        for i, (inputs, featmaps, targets) in enumerate(train_loader):
            inputs, featmaps, targets = inputs.to(args.device), featmaps.to(args.device), targets.to(args.device)
            # Optimize Discriminator

            loss = 0
            optimizer_d.zero_grad()

            feats, gen_targets = self._sample_vecs(inputs.shape[0])
            feats, gen_targets = feats.to(args.device), gen_targets.to(args.device)
            gen_image = gen(feats.unsqueeze(2).unsqueeze(3).detach())

            feats_gen, logits_cls_gen, logits_adv_gen = disc(gen_image)

            loss_cls_gen = cls_criterion(logits_cls_gen, gen_targets.long())
            loss = loss_cls_gen
            _loss_cls_gen += loss_cls_gen.item()  

            preds_gen = F.softmax(logits_cls_gen, dim=1).argmax(dim=1).cpu().numpy().tolist()
            ypred_gen.extend(preds_gen)
            ytrue_gen.extend(gen_targets)

            loss.backward()
            optimizer_d.step()

            feats, logits_cls, logits_adv = disc(featmaps)
            loss_cls = cls_criterion(logits_cls, targets.long())
            _loss_cls += loss_cls.item()

            preds = F.softmax(logits_cls, dim=1).argmax(dim=1).cpu().numpy().tolist()
            ypred.extend(preds)
            ytrue.extend(targets)


        acc = round((np.array(ypred) == np.array(ytrue)).sum() / len(ytrue), 4)  
        acc_gen = round((np.array(ypred_gen) == np.array(ytrue_gen)).sum() / len(ytrue_gen), 4)  
        if epoch % args.visualize_freq == 0 or epoch == args.epochs+1:
            print("Epoch {}, Training Iteration {}".format(epoch, i))
            print("Accuracy: {}, Accuracy gen: {}".format(acc, acc_gen))
            print("Loss_cls: {}, Loss_cls_gen: {}"
                  .format(_loss_cls/(i+1),_loss_cls_gen/(i+1)))
            visualize(featmaps[0], gen_image[0],
                      out_dir = train_output_dir + str(epoch) + "_" + str(i) + ".jpg")
        return {"Train_acc": acc,"Train_acc_gen": acc_gen,"Loss_cls":_loss_cls/(i+1), "Loss_cls_gen":_loss_cls_gen/(i+1)}

    def _test_2(self, args, epoch, disc, test_loader, test_output_dir):
        mse_criterion = nn.MSELoss()
        cls_criterion = nn.CrossEntropyLoss()
        _loss, _loss_cls, _loss_cls_gen = 0., 0., 0.
        ypred, ypred_gen = [], []
        ytrue, ytrue_gen = [], []
        for i, (inputs, featmaps, targets) in enumerate(test_loader):
            loss = 0
            inputs, featmaps, targets = inputs.to(args.device), featmaps.to(args.device), targets.to(args.device)

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


    def _build_statistics(self, args, train_loader, disc):
        print("Building & updating dataset with previous class statistics.")
        self._class_features =  np.zeros((10, 5000, 512))
        class_idx = [0]*10
        disc.eval()
        for i, (inputs, featmaps, targets) in enumerate(train_loader):
            featmaps = featmaps.to(args.device)
            # Optimize Discriminator
            feats, logits_cls, logits_adv = disc(featmaps)
            for i, target in enumerate(targets.detach().cpu().numpy().astype(int)):
                self._class_features[target, class_idx[target]] = feats[i].detach().cpu().numpy()
                class_idx[target] += 1
        print(class_idx)
        disc.train()
        
    def _sample_vecs(self, batch_size):
        vecs = []
        vec_targets = []
        for _ in range(batch_size):
            label = np.random.randint(10)
            vecs.append(torch.Tensor(self._class_features[label][np.random.choice(self._class_features[label].shape[0], 1)][0]))
#             vecs.append(torch.Tensor(self._class_features[label][np.random.choice(50, 1)][0]))
            vec_targets.append(torch.Tensor([label]))
        vecs = torch.stack(vecs)
        vec_targets = torch.stack(vec_targets).squeeze(1)
#         import pdb
#         pdb.set_trace()
        return vecs, vec_targets