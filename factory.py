import torch
from torch import optim

from convnet import generator, discriminator, resnet
from models import gan_feat, gan_random_feat, gan_feat_stored, gan_feat_cls, gan_feat_stored_mse, classification_only, gan_random_z, gan_random_z_feat
import torchvision.transforms as transforms
import data
    
def get_convnet(pretrained=False, layer_4=False, fixed=False, tanh=False):
    return resnet.resnet18(pretrained = pretrained, fixed= fixed, tanh=tanh)

def get_discriminator(data_name, disc_name):
    if disc_name == "resnet":
        return get_convnet()
    elif disc_name == "resnet_pretrained":
        return get_convnet(pretrained=True, fixed=True)
    elif data_name == "image":
        return discriminator.discriminator_image()
    elif data_name == "feature":
        return discriminator.discriminator_feature()
    elif data_name == "featmap":
        if disc_name == "no_resnet":
            return discriminator.discriminator_no_resnet()
        return discriminator.discriminator()
    
    raise NotImplementedError("Unknwon discriminator type {}.".format(convnet_type))
    
def get_generator(data_name, gen_name, deconv):
    if data_name == "image":
        return generator.generator_image(deconv)
    if data_name == "feature":
        return generator.generator_feature()
    elif data_name == "featmap":
        if gen_name == "embed":
            return generator.generator_emb(deconv)
        return generator.generator(deconv)
    
    raise NotImplementedError("Unknwon generator type {}.".format(convnet_type))

    
def get_model(args, disc, disc2, gen):
    if args.model == "gan_feat":
        return gan_feat.gan_feat(args, disc, disc2, gen)
    elif args.model == "gan_random_feat":
        return gan_random_feat.gan_random_feat(args, disc, disc2, gen)
    elif args.model == "gan_random_z":
        return gan_random_z.gan_random_z(args, disc, disc2, gen)
    elif args.model == "gan_random_z_feat":
        return gan_random_z_feat.gan_random_z_feat(args, disc, disc2, gen)
    elif args.model == "gan_feat_stored":
        return gan_feat_stored.gan_feat_stored(args, disc, disc2, gen)
    elif args.model == "gan_feat_stored_mse":
        return gan_feat_stored_mse.gan_feat_stored_mse(args, disc, disc2, gen)
    elif args.model == "gan_feat_cls":
        return gan_feat_cls.gan_feat_cls(args, disc, disc2, gen)
    elif args.model == "classification_only":
        return classification_only.classification_only(args, disc, disc2, gen)

def get_data(data_name, batch_size, workers, imagenet):
#     if data_name == "image":
#         from data import CIFAR10
    if data_name == "featmap" or data_name == "image":
        from data import CIFAR10_featmap as CIFAR10
        
    if data_name == "feature":
        from data import CIFAR10_feature as CIFAR10
        
    if imagenet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    
    trainset = CIFAR10(root='./data', train=True,
                       download=True, transform=transform, imagenet=imagenet)
    testset = CIFAR10(root='./data', train=False,
                      download=True, transform=transform, imagenet=imagenet)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=workers)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return train_loader, test_loader



