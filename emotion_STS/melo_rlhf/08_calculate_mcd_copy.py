#!/home/ubuntu/OpenVoice/.openvoiceenv/bin/python
from pymcd.mcd import Calculate_MCD
from glob import glob
import os
import numpy as np
import pandas as pd


# instance of MCD class
# three different modes "plain", "dtw" and "dtw_sl" for the above three MCD metrics
mcd_toolbox = Calculate_MCD(MCD_mode="plain")

# two inputs w.r.t. reference (ground-truth) and synthesized speeches, respectively

####our_SPS_Eval2
val=[]
real_audio_file='/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf/logs/example/evaluation/esd_input_ref/eval.csv'
real_audio_data=pd.read_csv(real_audio_file)
df_real=pd.DataFrame(real_audio_data)
for i_r, row_r in df_real.iterrows():
    ref_audio=row_r['reference_audio'].split('.')[0]+'.wav'
    input_audio=row_r['input_audio']
    output_audio=row_r['output_audio']
    mcd_value = mcd_toolbox.calculate_mcd(input_audio, output_audio)
    val.append(mcd_value)
    print(mcd_value)
val_mean=np.mean(val)
print('mean',val_mean)
# val=[]
# pattern='**/*.wav'
# directory='/home/ubuntu/OpenVoice/emotion_STS/melo_emo/logs/example/evaluation/real'
# directoryesd='/home/ubuntu/OpenVoice/emotion_STS/melo_emo/eval_input_audio'
# alist=[]
# dictionary={}

# for file in glob(os.path.join(directory, pattern), recursive=True):
#     alist.append(file)
# for file2 in glob(os.path.join(directoryesd, pattern), recursive=True):
#     name=file2.split('.')[0].split('/')[-1].split('_')
#     name=name[0]+'_'+name[1]
    
#     for i in alist:
#         filename=str(i).split('.')[0].split('/')[-1].split('_')
#         filename2=filename[0]+'_'+filename[1]
#         # print(filename2)
#         if filename2 ==name:

#             mcd_value = mcd_toolbox.calculate_mcd(i, file2)
#             val.append(mcd_value)
#             dictionary[name]=mcd_value


# # print(val)
# # input()
# val_mean=np.mean(val)
# print('val',val)
# print(val_mean)
# print(dictionary)

####our_SPS_Eval1
# val=[]
# pattern='**/*.wav'
# directory='/home/ubuntu/OpenVoice/emotion_STS/melo_emo/evaluation'
# directoryesd='/home/ubuntu/OpenVoice/emotion_STS/eval_data/esd/input'
# alist=[]

# for file in glob(os.path.join(directory, pattern), recursive=True):
#     alist.append(file)
# for file2 in glob(os.path.join(directoryesd, pattern), recursive=True):
#     name=file2.split('.')[0].split('/')[-1]
#     # print(name)
#     # input()
#     for i in alist:
#         filename=str(i).split('/')[-1].split('_')
#         filename2=filename[0]+'_'+filename[1]
#         # print(filename2)
#         if filename2 ==name:
#             mcd_value = mcd_toolbox.calculate_mcd(i, file2)
#             val.append(mcd_value)
# # print(val)
# # input()
# val_mean=np.mean(val)
# print(val_mean)

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
#         # filename2=filename[0]+'_'+filename[1]

#         if filename ==name:
#             mcd_value = mcd_toolbox.calculate_mcd(i, file2)
#             val.append(mcd_value)
# # print(val)
# # input()
# val_mean=np.mean(val)
# print(val_mean)

# ########emotionalstartgan
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
#             mcd_value = mcd_toolbox.calculate_mcd(i, file2)
#             val.append(mcd_value)
# # print(val)
# # input()
# val_mean=np.mean(val)
# print(val_mean)

######cycleGAN
# val=[]
# pattern='**/*.wav'
# directory='/home/ubuntu/OpenVoice/emotion_STS/eval_data/cyclegan'
# directoryesd='/home/ubuntu/OpenVoice/emotion_STS/eval_data/esd/input'
# alist=[]

# for file in glob(os.path.join(directory, pattern), recursive=True):
#     alist.append(file)
# for file2 in glob(os.path.join(directoryesd, pattern), recursive=True):
#     name=file2.split('.')[0].split('/')[-1]
#     print(name)
#     for i in alist:
#         filename=str(i).split('-')[-1].split('.')[0]
#         # filename2=filename[0]+'_'+filename[1]
#         # print(filename)
#         # input()
#         if filename ==name:
#             mcd_value = mcd_toolbox.calculate_mcd(i, file2)
#             val.append(mcd_value)
# # print(val)
# # input()
# val_mean=np.mean(val)
# print(val_mean)

# mcd_value1 = mcd_toolbox.calculate_mcd('/home/ubuntu/OpenVoice/emotion_STS/melo_emo/evaluation/4362_48858_tmp51000_angry.wav', '/home/ubuntu/OpenVoice/emotion_STS/melo_emo/generate_audio/4362_48858_cheerful.wav')
# mcd_value2 = mcd_toolbox.calculate_mcd('/home/ubuntu/OpenVoice/emotion_STS/melo_emo/evaluation/458_2922_tmp130000_sad.wav', '/home/ubuntu/OpenVoice/emotion_STS/melo_emo/generate_audio/458_2922_cheerful.wav')
# mcd_value3 = mcd_toolbox.calculate_mcd('/home/ubuntu/OpenVoice/emotion_STS/melo_emo/evaluation/302_61159_tmp130000_friendly.wav', '/home/ubuntu/OpenVoice/emotion_STS/melo_emo/generate_audio/302_61159_angry.wav')
# print(mcd_value1)
# print(mcd_value2)
# print(mcd_value3)

