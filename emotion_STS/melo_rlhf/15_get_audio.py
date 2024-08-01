#!/home/ubuntu/OpenVoice/.openvoiceenv/bin/python
# import matplotlib.pyplot as plt
# import IPython.display as ipd
import torch
import soundfile
import librosa
import os
from glob import glob
import random

# hps = utils.get_hparams_from_file("/home/ubuntu/OpenVoice/emotion_STS/melo/logs_recon&dur/example/config.json")
# print('hps',hps)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# directory="/home/ubuntu/efs/sts/Emotion_Speech_Dataset"
# alist=[]
# pattern='**/*.wav'
# for file in glob(os.path.join(directory, pattern), recursive=True):
#     alist.append(file)
# str='\n'
# f=open('/home/ubuntu/OpenVoice/emotion_STS/melo/09.esd_path.txt','w')
# f.write(str.join(alist))
# f.close

# f=open('/home/ubuntu/OpenVoice/emotion_STS/melo_emo/09.esd_path.txt','r')
# filelist=[]
# for line in f.readlines():
#     line =line.strip('\n')
#     filelist.append(line)
# selected_file=random.sample(filelist,40)
# for i in selected_file:
#     audio, sample_rate = librosa.load(i, sr=44100)
#     filepath=i.split('/')[-1].split('.')[0] +'_'+i.split('/')[-2].lower()+'.wav'

#     output_path=f'/home/ubuntu/OpenVoice/emotion_STS/melo_emo/eval_input_audio/real/{filepath}'
    # soundfile.write(output_path, audio, 44100)

# pri='/home/ubuntu/OpenVoice/data/emo_v/amused_1-15_0001.wav'
# audio, sample_rate = librosa.load(pri, sr=44100)
# output_path='/home/ubuntu/OpenVoice/data/full_dataset/amused_1-15_0001.wav'
# soundfile.write(output_path, audio, 44100)
directory = '/home/ubuntu/OpenVoice/data/emo_v_sam'
directory2= '/home/ubuntu/efs/sts/emo_v/sam'
wav_pattern='**/*.wav'
for file in glob(os.path.join(directory, wav_pattern), recursive=True):
    audio, sample_rate = librosa.load(file, sr=44100)
    output_path=os.path.join(directory2, file.split('/')[-1])
    soundfile.write(output_path, audio, 44100)
    
