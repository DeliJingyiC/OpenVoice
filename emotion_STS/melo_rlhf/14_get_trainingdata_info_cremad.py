#!/home/ubuntu/OpenVoice/.openvoiceenv/bin/python
import os
import pandas as pd
import csv
from sklearn.utils import shuffle
import random
from glob import glob


directory = '/home/ubuntu/efs/sts/CREMA-D/AudioWAV'
wav_pattern='**/*.wav'
emo_pattern='**/*.wav.emo.npy'
df = pd.DataFrame()
adict={}
# count_wav=0
# for file in glob(os.path.join(directory, wav_pattern), recursive=True):
#     count_wav+=1
# count_emo=0
for file in glob(os.path.join(directory, emo_pattern), recursive=True):
    # count_emo+=1
# print(count_wav)
# print(count_emo)
    # print(file)
    # input()
    spk=file.split('/')[-1].split('_')[0]
    txt=file.split('/')[-1].split('_')[1]
    emo=file.split('/')[-1].split('_')[2]
    tense=file.split('/')[-1].split('_')[3].split('.')[0]
    wav_file=file.split('.')[0]+'.wav'
    # print(wav_file)
    # input()
    adict['spk']=spk
    adict['txt']=txt
    adict['emo']=emo
    adict['tense']=tense
    adict['emo_address']=file
    adict['wav_address']=wav_file

    df = pd.concat([df, pd.DataFrame([adict])],ignore_index=True)

df.to_csv('/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf/data/example/cremad_info.csv')
