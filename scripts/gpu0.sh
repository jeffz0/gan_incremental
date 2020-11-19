#!/bin/sh
python3 main.py --device 0 --data feature --model gan_random_z_feat --optimizer adam --lr 3e-5 --adv --adv_type lsgan --cls_w 1 --phase_2 --gen_lr_factor 1 --epochs_cls 20 --epochs 40 --visualize 20;
python3 main.py --device 0 --data feature --model gan_random_z_feat --optimizer adam --lr 3e-5 --adv --adv_type lsgan --cls_w 1 --phase_2 --gen_lr_factor 1 --epochs_cls 20 --epochs 40 --visualize 20 --joint;

python3 main.py --device 0 --data feature --model gan_random_z_feat --optimizer adam --lr 3e-5 --adv --adv_type hinge --cls_w 1 --phase_2 --gen_lr_factor 1 --epochs_cls 20 --epochs 40 --visualize 20;
python3 main.py --device 0 --data feature --model gan_random_z_feat --optimizer adam --lr 3e-5 --adv --adv_type hinge --cls_w 1 --phase_2 --gen_lr_factor 1 --epochs_cls 20 --epochs 40 --visualize 20 --joint;

python3 main.py --device 0 --data feature --model gan_random_z_feat --optimizer adam --lr 3e-5 --adv --adv_type nsgan --cls_w 1 --phase_2 --gen_lr_factor 1 --epochs_cls 20 --epochs 40 --visualize 20;
python3 main.py --device 0 --data feature --model gan_random_z_feat --optimizer adam --lr 3e-5 --adv --adv_type nsgan --cls_w 1 --phase_2 --gen_lr_factor 1 --epochs_cls 20 --epochs 40 --visualize 20 --joint;






