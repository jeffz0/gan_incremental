import matplotlib.pyplot as plt
import argparse
import copy
import random
import time
import pdb
import os, sys
import json

from utils import visualize_featmap as visualize, adversarial_loss, visualize_graph

import numpy as np
import factory
from torch.autograd import Variable

import torch
import torch.utils.model_zoo as model_zoo
from torch import nn, optim
from torch.nn import functional as F
import torchvision


parser = argparse.ArgumentParser()

# Model
parser.add_argument("--data", type=str, default="featmap", help="type of training data")
parser.add_argument("--model", type=str, default="gan_feat", help="model for training")
parser.add_argument("--disc", type=str, default="no_resnet", help="type of discriminator")
parser.add_argument("--gen", type=str, default="", help="type of generator")
parser.add_argument("--deconv", action="store_true", default=False)
parser.add_argument("--name", type=str, default="")

# Data
parser.add_argument("--workers", type=int, default=3, help="workers for dataloader")
parser.add_argument("--epochs", type=int, default=40, help="number of epochs of training")
parser.add_argument("--epochs_cls", type=int, default=10, help="number of epochs of training")
parser.add_argument("--epochs_2", type=int, default=10, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")

# Training 
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--gen_lr_factor", type=float, default=3, help="generator is factor less than disc")
parser.add_argument("--optimizer", type=str, default="adam", help="options=[sgd, adam]")
parser.add_argument("--cls_w", type=float, default=1, help="classification weight")
parser.add_argument("--adv", action="store_true", default=False)
parser.add_argument("--adv_w", type=float, default=1, help="adversarial weight")
parser.add_argument("--recon", action="store_true", default=False)
parser.add_argument("--adv_r", type=float, default=1, help="recon weight")
parser.add_argument("--random_z", action="store_true", default=False)
parser.add_argument("--joint", action="store_true", default=False)
parser.add_argument("--adv_type", type=str, default="hinge", help="options=[hinge,nsgan,lsgan]")
parser.add_argument("--phase_2", action="store_true", default=False)
parser.add_argument("--index_sampling", action="store_true", default=False)
parser.add_argument("--mse", action="store_true", default=False)
parser.add_argument("--mse_w", type=float, default=1, help="mse weight")

parser.add_argument("--stored_features", action="store_true", default=False)
parser.add_argument("--stored_features_pretrained", action="store_true", default=False)

# Misc
parser.add_argument("--test", action="store_true", default=False)
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--visualize_freq", type=int, default=20, help="visualization frequency during training")
parser.add_argument('--device', type=int, default=0)

args = parser.parse_args()

FloatTensor = torch.FloatTensor

if __name__ == "__main__":   
    start_time = time.time()
    
    model_name = "data_{}_disc_{}_gen_{}_deconv_{}_gan_{}_{}_lr_{}_gen_factor_{}_cls_{}_".format(args.data, args.disc, args.gen, args.deconv, args.adv_type, args.optimizer, args.lr, args.gen_lr_factor, args.cls_w)
#     if args.adv:
#         model_name = "{}adv_{}_".format(model_name, args.adv_w)
    if args.recon:
        model_name = "{}recon_{}_".format(model_name, args.adv_r)
    if args.mse:
        model_name = "{}mse_{}_".format(model_name, args.mse_w)
    if args.joint:
        model_name = "{}joint_".format(model_name)
    if args.phase_2:
        model_name = "{}phase2_".format(model_name)
    if args.index_sampling:
        model_name = "{}index_sampling_".format(model_name)
    if args.stored_features:
        model_name = "{}stored_features_".format(model_name)
    if args.stored_features_pretrained:
        model_name = "{}stored_features_pretrained_".format(model_name)
            
    model_name = model_name + str(args.name)
    
    if not os.path.exists("outputs"):
        os.mkdir("outputs")
    if not os.path.exists("outputs/" + args.model):
        os.mkdir("outputs/" + args.model)
    output_dir = os.path.join("outputs/" + args.model, model_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    print("Results and models will be saved in ", output_dir)
        
    model_output_dir = os.path.join(output_dir,"model/")
    if not os.path.exists(model_output_dir):
        os.mkdir(model_output_dir)
    train_output_dir = os.path.join(output_dir,"train/")
    if not os.path.exists(train_output_dir):
        os.mkdir(train_output_dir)
    test_output_dir = os.path.join(output_dir,"test/")
    if not os.path.exists(test_output_dir):
        os.mkdir(test_output_dir)
        
    for i in range(10):
        if not os.path.exists(os.path.join(test_output_dir, str(i))):
            os.mkdir(os.path.join(test_output_dir, str(i)))

    # Load data
    train_loader, test_loader = factory.get_data("featmap", args.batch_size, args.workers)

    # Load models
    disc = factory.get_discriminator(args.data, args.disc).to(args.device)
    disc2 = factory.get_discriminator(args.data, args.disc).to(args.device)
    gen = factory.get_generator(args.data, args.gen, args.deconv).to(args.device)
    
    model = factory.get_model(args, disc, disc2, gen)

    if args.test:
        with open(output_dir +'/config.txt', 'w') as f:
            json.dump(args.__dict__, f)

        disc.load_state_dict(torch.load(model_output_dir + 'disc.pth'))
        disc2.load_state_dict(torch.load(model_output_dir + 'disc2.pth'))
        gen.load_state_dict(torch.load(model_output_dir + 'gen.pth'))
        print("TESTING, Loaded model")

        disc.eval()
        disc2.eval()
        gen.eval()
#         model._build_statistics(args, train_loader, disc)
        model._build_statistics_index(args, train_loader, disc)
#         test_loss = model._test(args, args.epochs_cls+args.epochs+1, disc, gen, test_loader, test_output_dir)
        test_loss = model._test(args, args.epochs, disc, gen, test_loader, test_output_dir)
#         with open(output_dir +'/train_losses.txt', 'r') as f:
#             train_losses = json.load(f)

#         visualize_graph(train_losses, args.epochs_cls+args.epochs+1, output_dir, recon=True)
        print("Outputted test results")
    else:
        with open(output_dir +'/config.txt', 'w') as f:
            json.dump(args.__dict__, f)
            
        model.train(train_loader, test_loader, output_dir, train_output_dir, test_output_dir, model_output_dir)

    print("Finished in --- %s seconds ---" % (time.time() - start_time))
        
        
        
      

