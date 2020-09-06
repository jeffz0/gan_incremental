#!/bin/sh
# python3 main.py --device 0 --model gan_feat_cls --adv --adv_type hinge --optimizer adam --lr 1e-4 --joint --cls_w 1 --phase_2 --gen_lr_factor 0.5 --test;
# python3 main.py --device 0 --model gan_feat_cls --adv --adv_type hinge --optimizer adam --lr 1e-4 --cls_w 1 --phase_2 --gen_lr_factor 0.5 --test;
python3 main.py --device 0 --model gan_feat_cls --adv --adv_type hinge --optimizer adam --lr 1e-5 --joint --cls_w 1 --phase_2 --gen_lr_factor 0.5 --recon;
# python3 main.py --device 0 --model gan_feat_cls --adv --adv_type hinge --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 0.5  --test;
# python3 main.py --device 0 --model gan_feat_cls --adv --adv_type hinge --optimizer adam --lr 1e-4 --joint --cls_w 1 --phase_2 --gen_lr_factor 1  --test;
python3 main.py --device 0 --model gan_feat_cls --adv --adv_type hinge --optimizer adam --lr 1e-4 --cls_w 1 --phase_2 --gen_lr_factor 1 --recon;
# python3 main.py --device 0 --model gan_feat_cls --adv --adv_type hinge --optimizer adam --lr 1e-5 --joint --cls_w 1 --phase_2 --gen_lr_factor 1  --test;
# python3 main.py --device 0 --model gan_feat_cls --adv --adv_type hinge --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 1  --test;
python3 main.py --device 0 --model gan_feat_cls --adv --adv_type hinge --optimizer adam --lr 1e-4 --joint --cls_w 1 --phase_2 --gen_lr_factor 4  --recon;
# python3 main.py --device 0 --model gan_feat_cls --adv --adv_type hinge --optimizer adam --lr 1e-4 --cls_w 1 --phase_2 --gen_lr_factor 4  --test;
# python3 main.py --device 0 --model gan_feat_cls --adv --adv_type hinge --optimizer adam --lr 1e-5 --joint --cls_w 1 --phase_2 --gen_lr_factor 4  --test;
# python3 main.py --device 0 --model gan_feat_cls --adv --adv_type hinge --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 4  --test;

# python3 main.py --device 0 --model gan_feat_cls --adv --adv_type nsgan --optimizer adam --lr 1e-4 --joint --cls_w 1 --phase_2 --gen_lr_factor 0.5  --test;
# python3 main.py --device 0 --model gan_feat_cls --adv --adv_type nsgan --optimizer adam --lr 1e-4 --cls_w 1 --phase_2 --gen_lr_factor 0.5 --test;
python3 main.py --device 0 --model gan_feat_cls --adv --adv_type nsgan --optimizer adam --lr 1e-5 --joint --cls_w 1 --phase_2 --gen_lr_factor 0.5 --recon;
python3 main.py --device 0 --model gan_feat_cls --adv --adv_type nsgan --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 0.5 --recon;
python3 main.py --device 0 --model gan_feat_cls --adv --adv_type nsgan --optimizer adam --lr 1e-4 --joint --cls_w 1 --phase_2 --gen_lr_factor 1 --recon;
# python3 main.py --device 0 --model gan_feat_cls --adv --adv_type nsgan --optimizer adam --lr 1e-4 --cls_w 1 --phase_2 --gen_lr_factor 1 --test;
python3 main.py --device 0 --model gan_feat_cls --adv --adv_type nsgan --optimizer adam --lr 1e-5 --joint --cls_w 1 --phase_2 --gen_lr_factor 1 --recon;
python3 main.py --device 0 --model gan_feat_cls --adv --adv_type nsgan --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 1 --recon;
python3 main.py --device 0 --model gan_feat_cls --adv --adv_type nsgan --optimizer adam --lr 1e-4 --joint --cls_w 1 --phase_2 --gen_lr_factor 4 --recon;
# python3 main.py --device 0 --model gan_feat_cls --adv --adv_type nsgan --optimizer adam --lr 1e-4 --cls_w 1 --phase_2 --gen_lr_factor 4 --test;
# python3 main.py --device 0 --model gan_feat_cls --adv --adv_type nsgan --optimizer adam --lr 1e-5 --joint --cls_w 1 --phase_2 --gen_lr_factor 4 --test;
python3 main.py --device 0 --model gan_feat_cls --adv --adv_type nsgan --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 4 --recon;

# python3 main.py --device 0 --model gan_feat_cls --adv --adv_type lsgan --optimizer adam --lr 1e-4 --joint --cls_w 1 --phase_2 --gen_lr_factor 0.5 --test;
# python3 main.py --device 0 --model gan_feat_cls --adv --adv_type lsgan --optimizer adam --lr 1e-4 --cls_w 1 --phase_2 --gen_lr_factor 0.5 --test;
# python3 main.py --device 0 --model gan_feat_cls --adv --adv_type lsgan --optimizer adam --lr 1e-5 --joint --cls_w 1 --phase_2 --gen_lr_factor 0.5 --test;
python3 main.py --device 0 --model gan_feat_cls --adv --adv_type lsgan --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 0.5 --recon;
# python3 main.py --device 0 --model gan_feat_cls --adv --adv_type lsgan --optimizer adam --lr 1e-4 --joint --cls_w 1 --phase_2 --gen_lr_factor 1 --test;
# python3 main.py --device 0 --model gan_feat_cls --adv --adv_type lsgan --optimizer adam --lr 1e-4 --cls_w 1 --phase_2 --gen_lr_factor 1 --test;
# python3 main.py --device 0 --model gan_feat_cls --adv --adv_type lsgan --optimizer adam --lr 1e-5 --joint --cls_w 1 --phase_2 --gen_lr_factor 1 --test;
python3 main.py --device 0 --model gan_feat_cls --adv --adv_type lsgan --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 1 --recon;
# python3 main.py --device 0 --model gan_feat_cls --adv --adv_type lsgan --optimizer adam --lr 1e-4 --joint --cls_w 1 --phase_2 --gen_lr_factor 4 --test;
# python3 main.py --device 0 --model gan_feat_cls --adv --adv_type lsgan --optimizer adam --lr 1e-4 --cls_w 1 --phase_2 --gen_lr_factor 4 --test;
# python3 main.py --device 0 --model gan_feat_cls --adv --adv_type lsgan --optimizer adam --lr 1e-5 --joint --cls_w 1 --phase_2 --gen_lr_factor 4 --test;
# python3 main.py --device 0 --model gan_feat_cls --adv --adv_type lsgan --optimizer adam --lr 1e-5 --cls_w 1 --phase_2 --gen_lr_factor 4 --test;


