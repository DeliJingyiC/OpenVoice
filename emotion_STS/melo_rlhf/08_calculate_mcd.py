#!/home/ubuntu/OpenVoice/.openvoiceenv/bin/python
from pymcd.mcd import Calculate_MCD
from glob import glob
import os
import numpy as np

# instance of MCD class
# three different modes "plain", "dtw" and "dtw_sl" for the above three MCD metrics
mcd_toolbox = Calculate_MCD(MCD_mode="plain")

# two inputs w.r.t. reference (ground-truth) and synthesized speeches, respectively


####our_SPS_Eval
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

mcd_value = mcd_toolbox.calculate_mcd('/home/ubuntu/OpenVoice/emotion_STS/melo_emo/generate_audio/4362_48858_cheerful.wav', '/home/ubuntu/OpenVoice/emotion_STS/melo_emo/generate_audio/4362_48858_tmp38200_angry.wav')
print(mcd_value)