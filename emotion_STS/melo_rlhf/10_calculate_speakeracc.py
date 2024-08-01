#!/home/ubuntu/OpenVoice/.openvoiceenv/bin/python
import torch
import torch.nn as nn
import wespeaker

import os
import librosa
import numpy as np
import pandas as pd
from glob import glob
from numpy import dot
from numpy.linalg import norm
from scipy.spatial.distance import cdist

classifier_skid = wespeaker.load_model('english')
###our_SPS_Eval
val=[]
real_audio_file='/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf/logs/example/evaluation/esd_input_ref/eval.csv'
real_audio_data=pd.read_csv(real_audio_file)
df_real=pd.DataFrame(real_audio_data)
for i_r, row_r in df_real.iterrows():
    ref_audio=row_r['reference_audio'].split('.')[0]+'.wav'
    input_audio=row_r['input_audio']
    output_audio=row_r['output_audio']
    # y =classifier_skid.extract_embedding(input_audio,16000)
    # y1 = classifier_skid.extract_embedding(output_audio)
    similarity=classifier_skid.compute_similarity(input_audio,output_audio)
    val.append(similarity)
    print(similarity)
val_sim_mean=np.mean(val)
print('val_sim_mean',val_sim_mean)


######Seq2Seq-EVC
# val=[]
# pattern='**/*.wav'
# directory='/home/ubuntu/OpenVoice/emotion_STS/eval_data/Seq2Seq-EVC'
# directoryesd='/home/ubuntu/OpenVoice/emotion_STS/eval_data/esd/input'
# alist=[]

# for file in glob(os.path.join(directory, pattern), recursive=True):
#     alist.append(file)
# for file2 in glob(os.path.join(directoryesd, pattern), recursive=True):
#     name=file2.split('.')[0].split('/')[-1]
#     print(name)
#     for i in alist:
#         filename=str(i).split('.')[0].split('/')[-1]
#         filename2=filename[0]+'_'+filename[1]

#         if filename ==name:
#             y =classifier_skid.extract_embedding(file2)
#             y1 = classifier_skid.extract_embedding(i)
#             # distance = cdist(y, y1, metric="cosine")[0,0]
#             # cos_sim = dot(y, y1)/(norm(y)*norm(y1))
#             similarity=classifier_skid.compute_similarity(file2,i)
#             # val_2.append(distance)
#             # val.append(cos_sim)
#             val.append(similarity)
# # val_mean=np.mean(val)
# # val_dis=np.mean(val_2)
# val_sim_mean=np.mean(val)
# # print(val)
# # print(val_mean)
# print(val_sim_mean)

########emotionalstartgan
# val=[]
# pattern='**/*.wav'
# directory='/home/ubuntu/OpenVoice/emotion_STS/eval_data/emotionalstartgan'
# directoryesd='/home/ubuntu/OpenVoice/emotion_STS/eval_data/esd/input'
# alist=[]

# for file in glob(os.path.join(directory, pattern), recursive=True):
#     alist.append(file)
# for file2 in glob(os.path.join(directoryesd, pattern), recursive=True):
#     name=file2.split('.')[0].split('/')[-1]
#     print(name)
#     for i in alist:
#         filename=str(i).split('-')[0].split('/')[-1]
#         # filename2=filename[0]+'_'+filename[1]

#         if filename ==name:
#             if filename ==name:
#                 y =classifier_skid.extract_embedding(file2)
#                 y1 = classifier_skid.extract_embedding(i)
#                 # distance = cdist(y, y1, metric="cosine")[0,0]
#                 # cos_sim = dot(y, y1)/(norm(y)*norm(y1))
#                 similarity=classifier_skid.compute_similarity(file2,i)
#                 # val_2.append(distance)
#                 # val.append(cos_sim)
#                 val.append(similarity)
# # val_mean=np.mean(val)
# # val_dis=np.mean(val_2)
# val_sim_mean=np.mean(val)
# # print(val)
# # print(val_mean)
# print(val_sim_mean)

####cycleGAN
# val=[]
# pattern='**/*.wav'
# directory='/home/ubuntu/OpenVoice/emotion_STS/eval_data/cyclegan'
# directoryesd='/home/ubuntu/OpenVoice/emotion_STS/eval_data/esd/input'
# alist=[]

# for file in glob(os.path.join(directory, pattern), recursive=True):
#     alist.append(file)
# for file2 in glob(os.path.join(directoryesd, pattern), recursive=True):
#     name=file2.split('.')[0].split('/')[-1]
#     # print(name)
#     for i in alist:
#         filename=str(i).split('-')[-1].split('.')[0]
#     # filename2=filename[0]+'_'+filename[1]
#     # print(filename)
#     # input()
#         if filename ==name:
#             y =classifier_skid.extract_embedding(file2)
#             y1 = classifier_skid.extract_embedding(i)
#             # distance = cdist(y, y1, metric="cosine")[0,0]
#             # cos_sim = dot(y, y1)/(norm(y)*norm(y1))
#             similarity=classifier_skid.compute_similarity(file2,i)
#             # val_2.append(distance)
#             # val.append(cos_sim)
#             val.append(similarity)
# # val_mean=np.mean(val)
# # val_dis=np.mean(val_2)
# val_sim_mean=np.mean(val)
# # print(val)
# # print(val_mean)
# print(val_sim_mean)

# y =classifier_skid.extract_embedding('/home/ubuntu/OpenVoice/emotion_STS/melo_emo/generate_audio/4362_48858_angry_tmp46800_new.wav')
# y1 = classifier_skid.extract_embedding('/home/ubuntu/OpenVoice/emotion_STS/melo_emo/generate_audio/4362_48858_cheerful.wav')
# similarity=classifier_skid.compute_similarity_withpath('/home/ubuntu/OpenVoice/emotion_STS/melo_emo/eval_input_audio/real/0011_000852_happy.wav'
#                                                        ,'/home/ubuntu/OpenVoice/emotion_STS/melo_emo/eval_ref_audio/real/Neutral-Angry-0013_000017.wav')
# print(similarity)