#!/bin/bash

### 给你这个作业起个名字，方便识别不同的作业
#SBATCH --job-name=lstm

### 指定该作业需要多少个节点
#SBATCH --nodes=2

### 指定该作业需要多少个CPU
#SBATCH --ntasks=16

### 指定该作业在哪个队列上执行
### 目前可用的GPU队列有 titan/tesla
#SBATCH --partition=titan

### 申请一块GPU卡
#SBATCH --gres=gpu:1 

nvidia-smi

### 执行你的作业
python train.py