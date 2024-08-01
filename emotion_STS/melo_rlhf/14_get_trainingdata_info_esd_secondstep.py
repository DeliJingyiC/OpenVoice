#!/home/ubuntu/OpenVoice/.openvoiceenv/bin/python
import os
import pandas as pd
import csv
from sklearn.utils import shuffle
import random
from glob import glob


file_ad='/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf/data/example/esd_info.csv'
df = pd.DataFrame()
adict={}

real_audio_data=pd.read_csv(file_ad)
df_sorted = real_audio_data.sort_values(by=['spk', 'emo', 'txt'])
df_sorted['txt_normalized'] = df_sorted.groupby(['spk', 'emo'])['txt'].rank(method='min').astype(int)

# Optional: Reset the index if needed
df_sorted = df_sorted.reset_index(drop=True)


df_sorted.to_csv('/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf/data/example/esd_info_txtnormal.csv')
