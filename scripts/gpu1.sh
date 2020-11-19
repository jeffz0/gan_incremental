#!/bin/sh

# python3 main.py --device 1 --model gan_feat_stored --adv --adv_type hinge --optimizer adam --lr 1e-5 --joint --cls_w 2 --phase_2 --gen_lr_factor 0.5 --recon;
# python3 main.py --device 1 --model gan_feat_stored --adv --adv_type hinge --optimizer adam --lr 1e-5 --joint --cls_w 5 --phase_2 --gen_lr_factor 0.5 --recon;
# python3 main.py --device 1 --model gan_feat_stored --adv --adv_type hinge --optimizer adam --lr 1e-5 --joint --cls_w 1 --phase_2 --gen_lr_factor 2 --recon;

# python3 main.py --device 1 --model gan_feat_stored --adv --adv_type hinge --optimizer adam --lr 1e-4 --joint --cls_w 1 --phase_2 --gen_lr_factor 2 --recon;
# python3 main.py --device 1 --model gan_feat_stored --adv --adv_type hinge --optimizer adam --lr 1e-4 --joint --cls_w 2 --phase_2 --gen_lr_factor 4 --recon;
# python3 main.py --device 1 --model gan_feat_stored --adv --adv_type hinge --optimizer adam --lr 1e-4 --joint --cls_w 5 --phase_2 --gen_lr_factor 4 --recon;

# python3 main.py --device 1 --model gan_feat_stored --adv --adv_type nsgan --optimizer adam --lr 1e-5 --joint --cls_w 1 --phase_2 --gen_lr_factor 2 --recon;
# python3 main.py --device 1 --model gan_feat_stored --adv --adv_type nsgan --optimizer adam --lr 1e-5 --joint --cls_w 2 --phase_2 --gen_lr_factor 0.5 --recon;
# python3 main.py --device 1 --model gan_feat_stored --adv --adv_type nsgan --optimizer adam --lr 1e-5 --joint --cls_w 5 --phase_2 --gen_lr_factor 0.5 --recon;

# python3 main.py --device 1 --model gan_feat_stored --adv --adv_type nsgan --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 2 --recon;
# python3 main.py --device 1 --model gan_feat_stored --adv --adv_type nsgan --optimizer adam --lr 1e-5 --cls_w 2 --phase_2 --gen_lr_factor 0.5 --recon;
# python3 main.py --device 1 --model gan_feat_stored --adv --adv_type nsgan --optimizer adam --lr 1e-5 --cls_w 5 --phase_2 --gen_lr_factor 0.5 --recon;

# python3 main.py --device 1 --model gan_feat_stored --adv --adv_type nsgan --optimizer adam --lr 1e-4 --joint --cls_w 1 --phase_2 --gen_lr_factor 2 --recon;
# python3 main.py --device 1 --model gan_feat_stored --adv --adv_type nsgan --optimizer adam --lr 1e-4 --joint --cls_w 2 --phase_2 --gen_lr_factor 1 --recon;
# python3 main.py --device 1 --model gan_feat_stored --adv --adv_type nsgan --optimizer adam --lr 1e-4 --joint --cls_w 5 --phase_2 --gen_lr_factor 1 --recon;


# python3 main.py --device 1 --model gan_feat_stored --adv --adv_type hinge --optimizer adam --lr 1e-4 --joint --cls_w 2 --phase_2 --gen_lr_factor 4 --recon --adv_r 5;

# python3 main.py --device 1 --model gan_feat_stored --adv --adv_type hinge --optimizer adam --lr 1e-4 --joint --cls_w 10 --phase_2 --gen_lr_factor 4 --recon;
# python3 main.py --device 1 --model gan_feat_stored --adv --adv_type hinge --optimizer adam --lr 1e-4 --joint --cls_w 5 --phase_2 --gen_lr_factor 4 --recon --adv_r 5;

# python3 main.py --device 1 --model gan_feat_stored --adv --adv_type nsgan --optimizer adam --lr 1e-5 --joint --cls_w 10 --phase_2 --gen_lr_factor 0.5 --recon;
# python3 main.py --device 1  --model gan_feat_stored --adv --adv_type nsgan --optimizer adam --lr 1e-5 --joint --cls_w 5 --phase_2 --gen_lr_factor 0.5 --recon --adv_r 5;

# python3 main.py --device 1 --model gan_feat_stored --adv --adv_type nsgan --optimizer adam --lr 1e-5 --cls_w 10 --phase_2 --gen_lr_factor 0.5 --recon;
# python3 main.py --device 1 --model gan_feat_stored --adv --adv_type nsgan --optimizer adam --lr 1e-5 --cls_w 5 --phase_2 --gen_lr_factor 0.5 --recon --adv_r 5;



# python3 main.py --device 1 --model gan_feat_stored_mse --adv --adv_type hinge --optimizer adam --lr 1e-4 --joint --cls_w 5 --phase_2 --gen_lr_factor 4 --recon --name same_sample;

# python3 main.py --device 0 --model gan_feat_stored_mse --adv --adv_type nsgan --optimizer adam --lr 1e-4 --joint --cls_w 5 --phase_2 --gen_lr_factor 4 --recon --name same_sample;

# python3 main.py --device 1 --model gan_feat_stored --adv --adv_type hinge --optimizer adam --lr 1e-5 --joint --cls_w 1 --phase_2 --gen_lr_factor 0.5 --recon;

# python3 main.py --device 1 --model gan_feat_stored --adv --adv_type hinge --optimizer adam --lr 1e-4 --cls_w 1 --phase_2 --gen_lr_factor 1 --recon;

# python3 main.py --device 1 --model gan_feat_stored_mse --adv --adv_type hinge --optimizer adam --lr 1e-4 --joint --cls_w 1 --phase_2 --gen_lr_factor 4 --recon --mse --mse_w 1;

# # python3 main.py --device 1 --model gan_feat_stored --adv --adv_type nsgan --optimizer adam --lr 1e-5 --joint --cls_w 1 --phase_2 --gen_lr_factor 0.5 --recon;
# # python3 main.py --device 1 --model gan_feat_stored --adv --adv_type nsgan --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 0.5 --recon;
# python3 main.py --device 1 --model gan_feat_stored_mse --adv --adv_type nsgan --optimizer adam --lr 1e-4 --joint --cls_w 1 --phase_2 --gen_lr_factor 1 --recon --mse --mse_w 1;

# # python3 main.py --device 1 --model gan_feat_stored --adv --adv_type nsgan --optimizer adam --lr 1e-5 --joint --cls_w 1 --phase_2 --gen_lr_factor 1 --recon;
# python3 main.py --device 1 --model gan_feat_stored_mse --adv --adv_type nsgan --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 1 --recon --mse --mse_w 1;

# python3 main.py --device 1 --model gan_feat_stored_mse --adv --adv_type nsgan --optimizer adam --lr 1e-4 --joint --cls_w 1 --phase_2 --gen_lr_factor 4 --recon --mse --mse_w 1;

# # python3 main.py --device 1 --model gan_feat_stored --adv --adv_type nsgan --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 4 --recon;

# python3 main.py --device 1 --model gan_feat_stored_mse --adv --adv_type lsgan --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 0.5 --recon --mse --mse_w 1;

# python3 main.py --device 1 --model gan_feat_stored_mse --adv --adv_type lsgan --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 1 --recon --mse --mse_w 1;

# python3 main.py --device 1 --model gan_feat_stored_mse --adv --adv_type hinge --optimizer adam --lr 1e-4 --joint --cls_w 1 --phase_2 --gen_lr_factor 4 --recon --mse --mse_w 0.1;

# python3 main.py --device 1 --model gan_feat_stored_mse --adv --adv_type nsgan --optimizer adam --lr 1e-4 --joint --cls_w 1 --phase_2 --gen_lr_factor 1 --recon --mse --mse_w 0.1;

# python3 main.py --device 1 --model gan_feat_stored_mse --adv --adv_type nsgan --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 1 --recon --mse --mse_w 0.1;

# python3 main.py --device 1 --model classification_only --disc no_resnet --optimizer adam --lr 1e-4 --epochs_cls 10 --name epochs_10;

# python3 main.py --device 1 --model classification_only --disc no_resnet --optimizer adam --lr 1e-4 --epochs_cls 50 --name epochs_50;

# python3 main.py --device 1 --model classification_only --disc block4 --optimizer adam --lr 1e-4 --epochs_cls 10 --name epochs_10;

# python3 main.py --device 1 --model classification_only --disc block4 --optimizer adam --lr 1e-4 --epochs_cls 50 --name epochs_50;

# python3 main.py --device 1 --model gan_feat_stored_mse --adv --adv_type hinge --optimizer adam --lr 1e-4 --joint --cls_w 1 --phase_2 --gen_lr_factor 4 --recon --mse --mse_w 10 --adv_r 10 --epochs_cls 20 --stored_features;

# python3 main.py --device 1 --model gan_feat_stored_mse --adv --adv_type nsgan --optimizer adam --lr 1e-4 --joint --cls_w 1 --phase_2 --gen_lr_factor 1 --recon --mse --mse_w 10 --adv_r 10 --epochs_cls 20 --stored_features;

# python3 main.py --device 1 --model gan_feat_stored_mse --adv --adv_type nsgan --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 1 --recon --mse --mse_w 10 --adv_r 10 --epochs_cls 20 --stored_features;

# python3 main.py --device 1 --model gan_feat_stored_mse --adv --adv_type nsgan --optimizer adam --lr 1e-4 --joint --cls_w 1 --phase_2 --gen_lr_factor 4 --recon --mse --mse_w 10 --adv_r 10 --epochs_cls 20 --stored_features;

# python3 main.py --device 1 --model gan_feat_stored_mse --adv --adv_type lsgan --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 0.5 --recon --mse --mse_w 10 --adv_r 10 --epochs_cls 20 --stored_features;

# python3 main.py --device 1 --model gan_feat_stored_mse --adv --adv_type lsgan --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 1 --recon --mse --mse_w 10 --adv_r 10 --epochs_cls 20 --stored_features;

# python3 main.py --device 1 --model gan_feat_stored_mse --adv --adv_type hinge --optimizer adam --lr 1e-4 --joint --cls_w 1 --phase_2 --gen_lr_factor 4 --recon --mse --mse_w 100 --adv_r 10 --epochs_cls 0 --stored_features --resume;

# python3 main.py --device 1 --model gan_feat_stored_mse --adv --adv_type nsgan --optimizer adam --lr 1e-4 --joint --cls_w 1 --phase_2 --gen_lr_factor 1 --recon --mse --mse_w 100 --adv_r 10 --epochs_cls 0 --stored_features --resume;

# python3 main.py --device 1 --model gan_feat_stored_mse --adv --adv_type nsgan --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 1 --recon --mse --mse_w 100 --adv_r 10 --epochs_cls 0 --stored_features --resume;

# python3 main.py --device 1 --model gan_feat_stored_mse --adv --adv_type nsgan --optimizer adam --lr 1e-4 --joint --cls_w 1 --phase_2 --gen_lr_factor 4 --recon --mse --mse_w 100 --adv_r 10 --epochs_cls 0 --stored_features --resume;

# python3 main.py --device 1 --model gan_feat_stored_mse --adv --adv_type lsgan --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 0.5 --recon --mse --mse_w 100 --adv_r 10 --epochs_cls 0 --stored_features --resume;

# python3 main.py --device 1 --model gan_feat_stored_mse --adv --adv_type lsgan --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 1 --recon --mse --mse_w 100 --adv_r 10 --epochs_cls 0 --stored_features --resume;


# python3 main.py --device 1 --model gan_feat_stored_mse --adv --adv_type hinge --optimizer adam --lr 1e-4 --joint --cls_w 1 --phase_2 --gen_lr_factor 4 --recon --mse --mse_w 10 --adv_r 1 --epochs_cls 0 --stored_features --resume;

# python3 main.py --device 1 --model gan_feat_stored_mse --adv --adv_type nsgan --optimizer adam --lr 1e-4 --joint --cls_w 1 --phase_2 --gen_lr_factor 1 --recon --mse --mse_w 10 --adv_r 1 --epochs_cls 0 --stored_features --resume;

# python3 main.py --device 1 --model gan_feat_stored_mse --adv --adv_type nsgan --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 1 --recon --mse --mse_w 10 --adv_r 1 --epochs_cls 0 --stored_features --resume;

# python3 main.py --device 1 --model gan_feat_stored_mse --adv --adv_type nsgan --optimizer adam --lr 1e-4 --joint --cls_w 1 --phase_2 --gen_lr_factor 4 --recon --mse --mse_w 10 --adv_r 1 --epochs_cls 0 --stored_features --resume;

# python3 main.py --device 1 --model gan_feat_stored_mse --adv --adv_type lsgan --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 0.5 --recon --mse --mse_w 10 --adv_r 1 --epochs_cls 0 --stored_features --resume;

# python3 main.py --device 1 --model gan_feat_stored_mse --adv --adv_type lsgan --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 1 --recon --mse --mse_w 10 --adv_r 1 --epochs_cls 0 --stored_features;

# python3 main.py --device 1 --data image --model gan_feat_stored_mse --adv --adv_type lsgan --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 1 --mse --mse_w 0.1 --adv_r 1 --epochs_cls 0 --stored_features_pretrained --disc image --gen image --joint;
# python3 main.py --device 1 --data image --model gan_feat_stored_mse --adv --adv_type lsgan --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 1 --mse --mse_w 0.1 --adv_r 10 --epochs_cls 0 --stored_features_pretrained --disc image --gen image --joint;
# python3 main.py --device 1 --data image --model gan_feat_stored_mse --adv --adv_type lsgan --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 1 --mse --mse_w 0.1 --adv_r 0.1 --epochs_cls 0 --stored_features_pretrained --disc image --gen image --joint;
# python3 main.py --device 1 --data image --model gan_feat_stored_mse --adv --adv_type lsgan --optimizer adam --lr 1e-4 --cls_w 1 --phase_2 --gen_lr_factor 1 --mse --mse_w 10 --adv_r 1 --epochs_cls 0 --stored_features_pretrained --disc image --gen image --joint;

# python3 main.py --device 1 --data image --model gan_feat_stored_mse --adv --adv_type lsgan --optimizer adam --lr 1e-4 --cls_w 1 --phase_2 --gen_lr_factor 1 --mse --mse_w 1 --adv_r 1 --epochs_cls 0 --stored_features_pretrained --disc image --gen image --joint;
# python3 main.py --device 1 --data image --model gan_feat_stored_mse --optimizer adam --lr 1e-4 --cls_w 1 --phase_2 --gen_lr_factor 1 --mse --mse_w 1 --epochs_cls 0 --stored_features_pretrained --disc image --gen image --joint;
# python3 main.py --device 1 --data image --model gan_feat_stored_mse --optimizer adam --lr 1e-4 --cls_w 1 --phase_2 --gen_lr_factor 1 --mse --mse_w 10 --epochs_cls 0 --stored_features_pretrained --disc image --gen image --joint;

python3 main.py --device 1 --model gan_random_z --optimizer adam --lr 3e-5 --adv --adv_type hinge --cls_w 1 --phase_2 --gen_lr_factor 1 --epochs_cls 20 --epochs 40 --visualize 20;
python3 main.py --device 1 --model gan_random_z --optimizer adam --lr 3e-5 --adv --adv_type lsgan --cls_w 1 --phase_2 --gen_lr_factor 1 --epochs_cls 20 --epochs 40 --visualize 20 --joint;
python3 main.py --device 1 --model gan_random_z --optimizer adam --lr 3e-5 --adv --adv_type nsgan --cls_w 1 --phase_2 --gen_lr_factor 1 --epochs_cls 20 --epochs 40 --visualize 20 --joint;
python3 main.py --device 1 --model gan_random_z --optimizer adam --lr 3e-5 --adv --adv_type nsgan --cls_w 1 --phase_2 --gen_lr_factor 1 --epochs_cls 20 --epochs 40 --visualize 20;