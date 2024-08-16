#!/home/ubuntu/OpenVoice/.openvoiceenv/bin/python
import os
import pandas as pd
import csv
from sklearn.utils import shuffle
import random
from glob import glob
from tqdm import tqdm


##document convert audio info
def find_files(directory, pattern='**/*.wav'):
    # dic={}
    # txt_path='/home/ubuntu/OpenVoice/data/full_dataset/goemotions_1.csv'
    # txt_data=pd.read_csv(txt_path)
    # txt_data=pd.DataFrame(txt_data)
    # outputfile ='/home/ubuntu/efs/sts/train/convert/doc_text.csv'
    # outputfile_lo ='/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf/data/example/doc_text.csv'

    # for file in tqdm(glob(os.path.join(directory, pattern), recursive=True)):
    #     dic['path']=file
    #     name=file.split('.')[0].split('/')[-1].split('_')
    #     dic['speaker']=name[0]
    #     dic['text']=txt_data['text'][int(name[1])]
    #     dic['text_id']=name[1]
    #     dic['emo']=name[2]
    #     df=pd.DataFrame([dic])
    #     df.to_csv(outputfile,sep=',',index=False,header=True,mode='a')
    #     df.to_csv(outputfile_lo,sep=',',index=False,header=True,mode='a')

# document tts audio inf:
    dic={}
    outputfile ='/home/ubuntu/efs/sts/train/tts/doc.csv'
    outputfile_lo ='/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf/data/example/doc.csv'
    for file in tqdm(glob(os.path.join(directory, pattern), recursive=True)):
        wavname=file.split('.')[0]+'.wav'
        dic['path']=wavname
        name=file.split('.')[0].split('/')[-1].split('_')
        dic['text_id']=name[0]
        dic['emo']=name[1]
        dic['emo_emb']=file
        df=pd.DataFrame([dic])
        df.to_csv(outputfile,sep=',',index=False,mode='a')
        df.to_csv(outputfile_lo,sep=',',index=False,mode='a')

directory='/home/ubuntu/efs/sts/train/convert'
directory2='/home/ubuntu/efs/sts/train/tts'
pattern='**/*.wav'
pattern2='**/*.emo.npy'

# find_files(directory,pattern)
find_files(directory2,pattern2)
