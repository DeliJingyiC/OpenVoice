#!/home/ubuntu/OpenVoice/.openvoiceenv/bin/python
import os
import pandas as pd
import csv
from sklearn.utils import shuffle
d={}

ref_file='/home/ubuntu/OpenVoice/data/tts_audio.csv'
convert_file='/home/ubuntu/OpenVoice/data/convert_audio.csv'
ref_csv_data=pd.read_csv(ref_file)
conv_csv_data=pd.read_csv(convert_file)

df_ref=pd.DataFrame(ref_csv_data)
df_ref=shuffle(df_ref)
df_conv=pd.DataFrame(conv_csv_data)
df_conv=shuffle(df_conv)

# print(len(df_conv))
# input()

for i, row in df_ref[:2].iterrows():
    emo_path=row['path']+".emo.npy"
    emo=row['emotion']
    for j, row_conv in df_conv[:20].iterrows():
        if row_conv['emotion']==emo:
            goal_path=row_conv['path']
            speaker=row_conv['speaker']
        for k, row_conv_input in df_conv[20:40].iterrows():
            if row_conv_input['emotion'] !=emo and row_conv_input['speaker']==speaker:
                input_path=row_conv_input['path']
                print(i)
                txt=f'{goal_path}|{speaker}|{input_path}|EN|{emo_path}'
                with open('/home/ubuntu/OpenVoice/emotion_STS/melo/data/example/metadata_test.list', 'a') as tf:
                    tf.write(f'{txt}\r\n') 


    
