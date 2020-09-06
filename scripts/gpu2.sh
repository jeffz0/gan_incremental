#!/bin/sh
python3 main.py --device 0 --adv --adv_type lsgan --optimizer sgd --lr 1e-2 --joint --cls_w 1 --phase_2;
python3 main.py --device 0 --adv --adv_type lsgan --optimizer sgd --lr 1e-2 --joint --cls_w 0.1 --phase_2;
python3 main.py --device 0 --adv --adv_type lsgan --optimizer sgd --lr 1e-2 --cls_w 1 --phase_2;
python3 main.py --device 0 --adv --adv_type lsgan --optimizer sgd --lr 1e-2 --cls_w 0.1 --phase_2;
python3 main.py --device 0 --adv --adv_type lsgan --optimizer sgd --lr 1e-3 --joint --cls_w 1 --phase_2;
python3 main.py --device 0 --adv --adv_type lsgan --optimizer sgd --lr 1e-3 --joint --cls_w 0.1 --phase_2;
python3 main.py --device 0 --adv --adv_type lsgan --optimizer sgd --lr 1e-3 --cls_w 1 --phase_2;
python3 main.py --device 0 --adv --adv_type lsgan --optimizer sgd --lr 1e-3 --cls_w 0.1 --phase_2;

python3 main.py --device 0 --adv --adv_type lsgan --optimizer adam --lr 1e-4 --joint --cls_w 1 --phase_2;
python3 main.py --device 0 --adv --adv_type lsgan --optimizer adam --lr 1e-4 --joint --cls_w 0.1 --phase_2;
python3 main.py --device 0 --adv --adv_type lsgan --optimizer adam --lr 1e-4 --cls_w 1 --phase_2;
python3 main.py --device 0 --adv --adv_type lsgan --optimizer adam --lr 1e-4 --cls_w 0.1 --phase_2;
python3 main.py --device 0 --adv --adv_type lsgan --optimizer adam --lr 1e-5 --joint --cls_w 1 --phase_2;
python3 main.py --device 0 --adv --adv_type lsgan --optimizer adam --lr 1e-5 --joint --cls_w 0.1 --phase_2;
python3 main.py --device 0 --adv --adv_type lsgan --optimizer adam --lr 1e-5 --cls_w 1 --phase_2;
python3 main.py --device 0 --adv --adv_type lsgan --optimizer adam --lr 1e-5 --cls_w 0.1 --phase_2;


