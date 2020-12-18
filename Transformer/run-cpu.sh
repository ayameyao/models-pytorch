#!/bin/bash

### 给你这个作业起个名字，方便识别不同的作业
#SBATCH --job-name=ucf101

### 指定该作业需要多少个节点
#SBATCH --nodes=2

### 指定该作业需要多少个CPU
#SBATCH --ntasks=16

### 指定该作业在哪个队列上执行
### 目前可用的队列有 cpu/fat/titan/tesla
#SBATCH --partition=fat


### 执行你的作业
python train.py