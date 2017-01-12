#!/bin/bash -e
clear

# intended as a companion to ami-125b2c72
# http://thecloudmarket.com/image/ami-125b2c72--cs231n-caffe-torch7-keras-lasagne-v2

# preferable machine: g2.2xlarge / g2.8xlarge

echo "============================================"
echo "Neural network project setup"
echo "============================================"

# no sudo on pip (doesn't identify)

# for neural network
pip install gym
pip install gym[atari]
sudo apt-get update
sudo apt-get install python-opencv

# for cron
pip install boto

# [global]
# device = gpu
# floatX = float32
export THEANO_FLAGS=floatX=float32,device=gpu

git clone https://github.com/xR86/rn-project