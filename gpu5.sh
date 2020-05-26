#!/bin/sh
python3 main.py --device 0 --model gan_feat_cls --adv --adv_type hinge --optimizer adam --lr 1e-4 --joint --cls_w 1 --phase_2;
python3 main.py --device 0 --model gan_feat_cls --adv --adv_type hinge --optimizer adam --lr 1e-4 --joint --cls_w 0.1 --phase_2;
python3 main.py --device 0 --model gan_feat_cls --adv --adv_type hinge --optimizer adam --lr 1e-4 --cls_w 1 --phase_2;
python3 main.py --device 0 --model gan_feat_cls --adv --adv_type hinge --optimizer adam --lr 1e-4 --cls_w 0.1 --phase_2;
