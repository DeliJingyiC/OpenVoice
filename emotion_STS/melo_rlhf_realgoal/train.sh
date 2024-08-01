
#!/bin/bash

CONFIG=$1
GPUS=$2
# NODE=$3
MODEL_NAME=$(basename "$(dirname $CONFIG)")
CHECKPOINT_DIR="/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf_realgoal/logs_emo/example"

PORT=10906

# torchrun --nproc_per_node=$GPUS \
#         --master_port=$PORT \
#     /home/ubuntu/OpenVoice/emotion_STS/melo_rlhf/train.py --c $CONFIG --model $MODEL_NAME --pretrain_rl /home/ubuntu/OpenVoice/emotion_STS/melo_rlhf/pretrained_RL
rm -rf /home/ubuntu/OpenVoice/emotion_STS/melo_rlhf_realgoal/buffer/*
while : # auto-resume: the code sometimes crash due to bug of gloo on some gpus
do

torchrun --nproc_per_node=$GPUS \
        --master_port=$PORT \
    /home/ubuntu/OpenVoice/emotion_STS/melo_rlhf_realgoal/train.py --c $CONFIG --model $MODEL_NAME  --pretrain_G $CHECKPOINT_DIR --pretrain_D $CHECKPOINT_DIR --pretrain_dur $CHECKPOINT_DIR --pretrain_rl /home/ubuntu/OpenVoice/emotion_STS/melo_rlhf_realgoal/pretrained_RL_emo/example

for PID in $(ps -aux | grep $CONFIG | grep python | awk '{print $2}')
do
    echo $PID
    kill -9 $PID
done
sleep 30
done