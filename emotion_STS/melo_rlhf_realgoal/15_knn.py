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
    print('row_r',row_r)
    emo_emb=row_r['emo_address']
    label=row_r['emo']
    emo_emb = torch.FloatTensor(np.load(emo_emb))
    features.append(emo_emb)
    labels.append(label)
    print('label',label)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)
knn = KNeighborsClassifier(n_neighbors=3)
features_list = [f.numpy() for f in features] 
features_array = np.vstack(features_list)
knn.fit(features_array, labels)
test_emb_dir='/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf_realgoal/logs_emo/example/evaluation/G_60120_eng'
pattern='**/*.emo.npy'
emo_dict={'sur':'surprise', 'sup':'surprise','sad':'sad','ang':'angry','hap':'happy','neu':'neutral'}
corr=0
count=0
for file in glob(os.path.join(test_emb_dir, pattern), recursive=True):
    test_emb=file
    test= torch.FloatTensor(np.load(test_emb))
    test = np.array(test).reshape(1, -1)
    y_pred = knn.predict(test)
    file_emo=file.split('/')[-1].split('_')[-3]
    file_emo=emo_dict[file_emo]
    count+=1
    if file_emo ==y_pred[0]:
        corr+=1
    print('file',file)
    print('y_pred',y_pred)
print('acc', corr/count)