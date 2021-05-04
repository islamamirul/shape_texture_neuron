#!/usr/bin/env bash

python main.py --model vit_base_patch16_224  --image_size 224

python main.py --model vit_large_patch16_224 --image_size 224
python main.py --model vit_base_patch16_384 --image_size 384
python main.py --model vit_small_patch16_224 --image_size 224

python main.py --model vit_base_patch32_384 --image_size 384
python main.py --model vit_large_patch32_384 --image_size 384
python main.py  --model vit_large_patch16_384 --image_size 384


python main.py --model vit_deit_tiny_patch16_224 --image_size 224
python main.py --model vit_deit_small_patch16_224 --image_size 224
python main.py --model vit_deit_base_patch16_224 --image_size 224
python main.py --model vit_deit_base_patch16_384 --image_size 384

python main.py --model vit_deit_tiny_distilled_patch16_224 --image_size 224
python main.py --model vit_deit_small_distilled_patch16_224 --image_size 224
python main.py --model vit_deit_base_distilled_patch16_224 --image_size 224
python main.py --model vit_deit_base_distilled_patch16_384 --image_size 384

