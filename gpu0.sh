#!/bin/sh

python3 main.py --device 0 --model gan_feat_stored --adv --adv_type nsgan --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 2 --recon;
python3 main.py --device 0 --model gan_feat_stored --adv --adv_type nsgan --optimizer adam --lr 1e-5 --cls_w 2 --phase_2 --gen_lr_factor 1 --recon;
python3 main.py --device 0 --model gan_feat_stored --adv --adv_type nsgan --optimizer adam --lr 1e-5 --cls_w 5 --phase_2 --gen_lr_factor 1 --recon;

python3 main.py --device 0 --model gan_feat_stored --adv --adv_type nsgan --optimizer adam --lr 1e-4 --joint --cls_w 1 --phase_2 --gen_lr_factor 2 --recon;
python3 main.py --device 0 --model gan_feat_stored --adv --adv_type nsgan --optimizer adam --lr 1e-4 --joint --cls_w 2 --phase_2 --gen_lr_factor 4 --recon;
python3 main.py --device 0 --model gan_feat_stored --adv --adv_type nsgan --optimizer adam --lr 1e-4 --joint --cls_w 5 --phase_2 --gen_lr_factor 4 --recon;

python3 main.py --device 0 --model gan_feat_stored --adv --adv_type nsgan --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 2 --recon;
python3 main.py --device 0 --model gan_feat_stored --adv --adv_type nsgan --optimizer adam --lr 1e-5 --cls_w 2 --phase_2 --gen_lr_factor 4 --recon;
python3 main.py --device 0 --model gan_feat_stored --adv --adv_type nsgan --optimizer adam --lr 1e-5 --cls_w 5 --phase_2 --gen_lr_factor 4 --recon;

python3 main.py --device 0 --model gan_feat_stored --adv --adv_type lsgan --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 2 --recon;
python3 main.py --device 0 --model gan_feat_stored --adv --adv_type lsgan --optimizer adam --lr 1e-5 --cls_w 2 --phase_2 --gen_lr_factor 0.5 --recon;
python3 main.py --device 0 --model gan_feat_stored --adv --adv_type lsgan --optimizer adam --lr 1e-5 --cls_w 5 --phase_2 --gen_lr_factor 0.5 --recon;

python3 main.py --device 0 --model gan_feat_stored --adv --adv_type lsgan --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 2 --recon;
python3 main.py --device 0 --model gan_feat_stored --adv --adv_type lsgan --optimizer adam --lr 1e-5 --cls_w 2 --phase_2 --gen_lr_factor 1 --recon;
python3 main.py --device 0 --model gan_feat_stored --adv --adv_type lsgan --optimizer adam --lr 1e-5 --cls_w 5 --phase_2 --gen_lr_factor 1 --recon;

