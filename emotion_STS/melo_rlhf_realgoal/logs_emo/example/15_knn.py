#!/home/ubuntu/OpenVoice/.openvoiceenv/bin/python

import os
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import torch


features = []
labels = []
val=[]
df = pd.DataFrame()
adict={}
real_audio_file='/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf_realgoal/data/example/esd_info.csv'
real_audio_data=pd.read_csv(real_audio_file)
df_real=pd.DataFrame(real_audio_data)
for i_r, row_r in df_real.iterrows():
    emo_emb=row_r['emo_address']
    label=row_r['emo']
    emo_emb = torch.FloatTensor(np.load(emo_emb))
    features.append(emo_emb)
    labels.append(label)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(features, labels)
test_emb='/home/ubuntu/efs/sts/Emotion_Speech_Dataset/0019/Sad/0019_001053.wav.emo.npy'
test= torch.FloatTensor(np.load(test_emb))

y_pred = knn.predict(test)
print(y_pred)