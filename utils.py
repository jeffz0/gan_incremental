import matplotlib.pyplot as plt
import copy
import random
import time
import pdb
import os, sys

import numpy as np
import torch
import torch.utils.model_zoo as model_zoo
from torch import nn, optim
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms

def train_step(args, autoencoder, inputs, targets, criterion):
    inputs, targets = inputs.to(args.device), targets.to(args.device)
    inputs_normalized = inputs
    if args.tanh:
        inputs_normalized = inputs_normalized*2-1.
    features, gen_image = autoencoder(inputs, fixed=args.fixed)
    if args.resnet:
        loss = criterion(gen_image, targets)
    else:
        loss = criterion(gen_image, inputs_normalized)
    
    if args.resnet:
        return loss, targets, gen_image
    return loss, inputs, gen_image


def visualize(input, gen_image, out_dir=None, featmap=True):
    if featmap:  
        for j in range(8):
            plt.subplot(2,8,j+1)
            plt.imshow((input[0*8+j].detach().cpu().numpy()+1.)/2)
            plt.axis('off')
        for j in range(8):
            plt.subplot(2,8,8+j+1)
            plt.imshow((gen_image[0*8+j].detach().cpu().numpy()+1.)/2)
            plt.axis('off')
    #         plt.imshow((gen_image.detach().cpu().numpy().transpose(1,2,0)+1.)/2)
        if out_dir is not None:
            plt.savefig(out_dir)
        plt.show()
    
    else:
        plt.subplot(1,2,1)
        plt.imshow((input.detach().cpu().numpy().transpose(1,2,0)+1.)/2)
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow((gen_image.detach().cpu().numpy().transpose(1,2,0)+1.)/2)
        plt.axis('off')
        if out_dir is not None:
            plt.savefig(out_dir)
        plt.show()
        plt.close()
    
    
def train_step_featmap(args, model, inputs, targets, criterion):
    inputs, targets = inputs.to(args.device), targets.to(args.device)
    features, featmaps, gen_image = model(inputs, fixed=args.fixed)
    loss = criterion(gen_image, featmaps)
    return loss, featmaps, gen_image

def train_step_gan(args, model, inputs, targets, criterion):
    inputs, targets = inputs.to(args.device), targets.to(args.device)
    features, featmaps, gen_image = model(inputs, fixed=args.fixed)
    loss = criterion(gen_image, featmaps)
    return loss, featmaps, gen_image
  
    
def visualize_classes(input, gen_image, labels, cls_count, out_dir, featmap=True):
    if featmap:
        for i in range(len(input)):
            label = int(labels[i].detach().cpu().numpy())
            if cls_count[label] < 5:
                plt.figure(figsize=(8,6))
                for j in range(8):
                    plt.subplot(2,8,j+1)
                    plt.imshow((input[i][0*8+j].detach().cpu().numpy()+1.)/2)
                    plt.axis('off')
                for j in range(8):
                    plt.subplot(2,8,8+j+1)
                    plt.imshow((gen_image[i][0*8+j].detach().cpu().numpy()+1.)/2)
                    plt.axis('off')

                plt.savefig(os.path.join(out_dir, str(label), str(cls_count[label]) + '.jpg'))
                plt.show()
                plt.close()

                cls_count[label] += 1
    else:
        for i in range(len(input)):
            label = int(labels[i].detach().cpu().numpy())
            if cls_count[label] < 5:
                plt.figure(figsize=(8,6))
                plt.subplot(1,2,1)
                plt.imshow((input[i].detach().cpu().numpy().transpose(1,2,0)+1.)/2)           
                plt.axis('off')
                plt.subplot(1,2,2)
                plt.imshow((gen_image[i].detach().cpu().numpy().transpose(1,2,0)+1.)/2)
                plt.axis('off')
                plt.savefig(os.path.join(out_dir, str(label), str(cls_count[label]) + '.jpg'))
                plt.show()
                plt.close()

                cls_count[label] += 1
    return cls_count
    
    
def visualize_graph(train_losses, epochs, out_dir, recon=False, mse=False):
    loss_cls = []
    loss_cls_gen = []
    loss_adv = []
    loss_adv_gen = []
    acc = []
    acc_gen = []
    if recon:
        recon_lst = []
    if mse:
        mse_lst = []
    for i in range(epochs):
        loss_cls.append(train_losses[i]['Loss_cls'])
        loss_cls_gen.append(train_losses[i]['Loss_cls_gen'])
        loss_adv.append(train_losses[i]['Loss_adv'])
        loss_adv_gen.append(train_losses[i]['Loss_adv_gen'])
        acc.append(train_losses[i]['Train_acc'])
        acc_gen.append(train_losses[i]['Train_acc_gen'])   
        if "Loss_recon" in train_losses[i] and recon:
            recon_lst.append(train_losses[i]['Loss_recon'])  
        if "Loss_mse" in train_losses[i] and mse:
            mse_lst.append(train_losses[i]['Loss_mse'])  
    
    
    plt.figure(figsize=(10,6))
    plt.title("Classification Loss")
    plt.plot(loss_cls, label='Real')
    plt.plot(loss_cls_gen, label='Gen')
    plt.legend()
    plt.savefig(out_dir + '/cls_loss.jpg')
    plt.show()
    
    plt.figure(figsize=(10,6))
    plt.title("Adversarial Loss")
    plt.plot(loss_adv, label='Real')
    plt.plot(loss_adv_gen, label='Gen')
    plt.legend()
    plt.savefig(out_dir + '/adv_loss.jpg')
    plt.show()
    
    plt.figure(figsize=(10,6))
    plt.title("Accuracy")
    plt.plot(acc, label='Real')
    plt.plot(acc_gen, label='Gen')
    plt.legend()
    plt.savefig(out_dir + '/acc.jpg')
    plt.show()
    
    if recon:
        plt.figure(figsize=(10,6))
        plt.title("Cosine Similarity Loss")
        plt.plot(recon_lst)
        plt.savefig(out_dir + '/recon.jpg')
        plt.show()

    if mse:
        plt.figure(figsize=(10,6))
        plt.title("MSE featuremap Loss")
        plt.plot(mse_lst)
        plt.savefig(out_dir + '/mse.jpg')
        plt.show() 
    
    
def adversarial_loss(outputs, is_real, is_disc=None, type_='nsgan', target_real_label=1.0, target_fake_label=0.0):
        real_label = torch.tensor(target_real_label)
        fake_label = torch.tensor(target_fake_label)

        if type_ == 'nsgan':
            criterion = nn.BCELoss()

        elif type_ == 'lsgan':
            criterion = nn.MSELoss()

        elif type_ == 'hinge':
            criterion = nn.ReLU()
            
        if type_ == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (real_label if is_real else fake_label).expand_as(outputs).to(outputs.device)
            loss = criterion(torch.sigmoid(outputs), labels)
            return loss

        
def print_statement(epoch, i, acc, acc_gen, _loss_cls, _loss_cls_gen, _loss_adv=None, _loss_adv_gen=None, _loss_recon=None, _loss_mse=None):
    print("Epoch {}, Training Iteration {}".format(epoch, i))
    print("Accuracy: {}, Accuracy gen: {}".format(acc, acc_gen))
    print("Loss_cls: {}, Loss_cls_gen: {}"
          .format(_loss_cls/(i+1),_loss_cls_gen/(i+1)))
    if _loss_adv is not None:
        print("Loss_adv: {}, Loss_adv_gen: {}"
          .format(_loss_adv/(i+1),_loss_adv_gen/(i+1)))
    if _loss_recon is not None:
        print("Loss_recon: {}"
          .format(_loss_recon/(i+1)))
    if _loss_mse is not None:
        print("Loss_recon: {}"
          .format(_loss_mse/(i+1)))
        
def return_statement(i, acc, acc_gen, _loss_cls, _loss_cls_gen, _loss_adv, _loss_adv_gen, _recon=None, _loss_mse=None):
    if _loss_mse is not None:
        if _recon is not None:
            return {"Train_acc": acc,"Train_acc_gen": acc_gen,"Loss_cls":_loss_cls/(i+1), "Loss_cls_gen":_loss_cls_gen/(i+1),"Loss_adv":_loss_adv/(i+1), "Loss_adv_gen":_loss_adv_gen/(i+1), "Loss_recon":_recon/(i+1), "Loss_mse":_loss_mse/(i+1)}
        return {"Train_acc": acc,"Train_acc_gen": acc_gen,"Loss_cls":_loss_cls/(i+1), "Loss_cls_gen":_loss_cls_gen/(i+1),"Loss_adv":_loss_adv/(i+1), "Loss_adv_gen":_loss_adv_gen/(i+1), "Loss_mse":_loss_mse/(i+1)}
    if _recon is not None:
        return {"Train_acc": acc,"Train_acc_gen": acc_gen,"Loss_cls":_loss_cls/(i+1), "Loss_cls_gen":_loss_cls_gen/(i+1),"Loss_adv":_loss_adv/(i+1), "Loss_adv_gen":_loss_adv_gen/(i+1), "Loss_recon":_recon/(i+1)}
    return {"Train_acc": acc,"Train_acc_gen": acc_gen,"Loss_cls":_loss_cls/(i+1), "Loss_cls_gen":_loss_cls_gen/(i+1),"Loss_adv":_loss_adv/(i+1), "Loss_adv_gen":_loss_adv_gen/(i+1)}
    
