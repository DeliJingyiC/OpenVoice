#!/home/ubuntu/OpenVoice/.openvoiceenv/bin/python
from sklearn.utils import shuffle
import os
import torch
# from openvoice import se_extractor
# from openvoice.api import BaseSpeakerTTS, ToneColorConverter
# from MeloTTS.melo.api import TTS
# from MeCab import Tagger
import pandas as pd
# import random
from glob import glob
import wave
import contextlib
from tqdm import tqdm
import random

device = "cuda:0" if torch.cuda.is_available() else "cpu"
ref_file='/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf/data/example/doc.csv'
convert_file='/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf/data/example/doc_text.csv'
real_audio_file='/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf/data/example/esd_info_txtnormal.csv'
real_audio_data=pd.read_csv(real_audio_file)
ref_csv_data=pd.read_csv(ref_file)
conv_csv_data=pd.read_csv(convert_file)
conv_csv_data=shuffle(conv_csv_data,n_samples=100000).values.tolist()
ref_csv_data=shuffle(ref_csv_data,n_samples=451).values.tolist()
real_audio_data=shuffle(real_audio_data,n_samples=7145).values.tolist()

# conv_csv_data.remove(['path', 'speaker', 'text', 'text_id', 'emo'])
# real_emo=['ANG', 'DIS','FEA','HAP','NEU','SAD']

emo_dic = {
    'emo': [[row[5], row[7]] for row in real_audio_data],
    'wav': [[row[6],row[7]] for row in real_audio_data],
    # 'txt':[row[7] for row in real_audio_data]
}


conv_csv_data1 = [row for row in conv_csv_data if row != ['path', 'speaker', 'text', 'text_id', 'emo']]
ref_csv_data1 = [row for row in ref_csv_data if row != ['path', 'text_id', 'emo', 'emo_emb']]
        
def get_non_matching_paths(emo_dic):
    while True:
        real_path = random.choice(emo_dic['wav'])[0]
        real_emo_path = random.choice(emo_dic['emo'])[0]
        r_emo = real_path.split('/')[-2].lower()
        r_emo_p = real_emo_path.split('/')[-2].lower()
        if r_emo != r_emo_p:
            return real_path, real_emo_path
def get_matching_paths(emo_dic):
    while True:
        real_path_list = random.choice(emo_dic['wav'])
        real_path=real_path_list[0]
        real_txt=real_path_list[1]

        real_emo_list = random.choice(emo_dic['emo'])
        real_emo_path=real_emo_list[0]
        real_emo_txt=real_emo_list[1]


        r_emo = real_path.split('/')[-2].lower()
        r_emo_p = real_emo_path.split('/')[-2].lower()
        r_speak_p=real_emo_path.split('/')[-3]
        r_speak = real_path.split('/')[-3]
        # print('r_speak_p',r_speak_p)
        # print('r_speak',r_speak)
        # input()
        # r_word=real_path.split('_')[1]
        # r_word_p=real_emo_path.split('_')[1]
        if r_emo != r_emo_p and r_speak_p==r_speak and real_txt==real_emo_txt:
            matching_paths = [
        path[0] for path in emo_dic['wav']
        if path[0].split('/')[-2].lower() == r_emo_p and path[0].split('/')[-3] == r_speak and path[1] == real_txt
    ]
            goal_path_real=random.choice(matching_paths)
        
            return real_path, real_emo_path,goal_path_real

# for i_ref in tqdm(ref_csv_data1):
#     emo=i_ref[2]
#     emopath=i_ref[3]

#     real_path, real_emo_path = get_non_matching_paths(emo_dic)
    
#     for j_conv in (conv_csv_data1):
#         if j_conv[4]==emo:
#             goal_path=j_conv[0]
#             speaker=j_conv[1]
#             text=j_conv[3]
#             for k in (conv_csv_data1):
#                 if k[4] !=emo and k[1]==speaker and k[3]==text:
#                     input_path=k[0]
#                     txt=f'{goal_path}|{speaker}|{input_path}|{emo}|{emopath}|{real_path}|{real_emo_path}'
#                     print(txt)
#                     with open('/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf/data/example/metadata_train_with_RealAudio2.list', 'a') as tf:
#                         tf.write(f'{txt}\r\n') 

for i_ref in tqdm(ref_csv_data1):
    emo=i_ref[2]
    emopath=i_ref[3]

    real_path, real_emo_path, real_goal_path = get_matching_paths(emo_dic)
    
    for j_conv in (conv_csv_data1):
        if j_conv[4]==emo:
            goal_path=j_conv[0]
            speaker=j_conv[1]
            text=j_conv[3]
            for k in (conv_csv_data1):
                if k[4] !=emo and k[1]==speaker and k[3]==text:
                    input_path=k[0]
                    txt=f'{goal_path}|{speaker}|{input_path}|{emo}|{emopath}|{real_path}|{real_emo_path}|{real_goal_path}'
                    print(txt)
                    with open('/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf/data/example/metadata_train_with_RealAudio_esd.list', 'a') as tf:
                        tf.write(f'{txt}\r\n') 



#         print(j_conv)
#         input()
#     # print(i[0])
#     # input()
# for i, row in tqdm(ref_csv_data['emo'].items()):
#     print(row)
#     emo=row
#     emopath=ref_csv_data.iloc[i]['emo_emb']
#     for j, row_conv in conv_csv_data.iterrows():
#         if row_conv['emo']==emo:
#             goal_path=row_conv['path']
#             speaker=row_conv['speaker']
#             text=row_conv['text_id']
#             df_emo_notsame=conv_csv_data[conv_csv_data['emo']!=emo]
#             df_speaker_same=df_emo_notsame[df_emo_notsame['speaker']==speaker]
#             df_text_same=df_speaker_same[df_speaker_same['text_id']==text]
#             if len(df_text_same) ==0:
#                 continue
#             else:
#                 for k, row_conv_input in df_text_same.iterrows():
#                     input_path=row_conv_input['path']
#                     txt=f'{goal_path}|{speaker}|{input_path}|{emo}|{emopath}'
#                     with open('/home/ubuntu/OpenVoice/emotion_STS/melo/data/example/metadata_train3.list', 'a') as tf:
#                         tf.write(f'{txt}\r\n') 
#             # for k, row_conv_input in df_text_same.iterrows():
            #     print(row_conv_input['path'])
            #     input()
            # print(emo)
            # print(df_emo_notsame)
            # input()
            # for k, row_conv_input in conv_csv_data.iterrows():
            #     if row_conv_input['emo'] !=emo and row_conv_input['speaker']==speaker:
            #         if row_conv_input['text_id']!=text:
            #             continue
            #         else:
            #             input_path=row_conv_input['path']
            #             print(i)
            #             txt=f'{goal_path}|{speaker}|{input_path}|{emo}|{emopath}'
            #             with open('/home/ubuntu/OpenVoice/emotion_STS/melo/data/example/metadata_train2.list', 'a') as tf:
            #                 tf.write(f'{txt}\r\n') 
    # print(i)
    # print(row)
    # print(ref_csv_data.iloc[i])

    # input()

# df_ref=pd.DataFrame(ref_csv_data)
# df_ref=shuffle(df_ref)
# df_conv=pd.DataFrame(conv_csv_data)
# df_conv=shuffle(df_conv)

# # print(len(df_conv))
# # input()

# for i, row in df_ref[:2].iterrows():
#     emo_path=row['path']+".emo.npy"
#     emo=row['emotion']
#     for j, row_conv in df_conv[:20].iterrows():
#         if row_conv['emotion']==emo:
#             goal_path=row_conv['path']
#             speaker=row_conv['speaker']
#         for k, row_conv_input in df_conv[20:40].iterrows():
#             if row_conv_input['emotion'] !=emo and row_conv_input['speaker']==speaker:
#                 input_path=row_conv_input['path']
#                 print(i)
#                 txt=f'{goal_path}|{speaker}|{input_path}|EN|{emo_path}'
#                 with open('/home/ubuntu/OpenVoice/emotion_STS/melo/data/example/metadata_test.list', 'a') as tf:
#                     tf.write(f'{txt}\r\n') 


    
