#!/home/ubuntu/OpenVoice/.openvoiceenv/bin/python
# import matplotlib.pyplot as plt
# import IPython.display as ipd
import torch
import utils
from api import STS
import os
from glob import glob

hps = utils.get_hparams_from_file("/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf_realgoal/logs_emo/example/config.json")
print('hps',hps)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

config_path='/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf_realgoal/logs_emo/example/config.json'
# ckpt_path='/home/ubuntu/OpenVoice/emotion_STS/melo/logs/example/G_6600.pth'
# ckpt_path='/home/ubuntu/OpenVoice/emotion_STS/melo_emo/ckpt/G_46800.pth'
ckpt_path='/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf_realgoal/logs_emo/example/G_61900.pth'
model = STS(device=device,config_path=config_path,ckpt_path=ckpt_path)
speaker_ids = model.hps.data.spk2id
cwd=os.getcwd()
output_directory=ckpt_path.split('/')[-3] + '/' + ckpt_path.split('/')[-2]
output_dir=f'{cwd}/{output_directory}/evaluation/real'
ckpt_num=ckpt_path.split('/')[-1].split('.')[0]


# for speaker_key in speaker_ids.keys():
# speaker_id = speaker_ids['4362']
output_path=f'{output_dir}/real/neu_angry_ch_G_61900.wav'
# output_path=f'{output_dir}/real/cremad_neutral_angry_G_58650_1_realemo.wav'

directory_path = os.path.dirname(output_path)
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
# if not os.path.exists(output_path):
#     data, samplerate = sf.read(file_path)

# output_path=f'{output_dir}/0013_000017_surprise_46800_new.wav'
# output_path=f'{output_dir}/32_30510_friendly_tmp25500_new.wav'
# output_path=f'{output_dir}/Angry-0013_000017_20_tmp46400_new.wav'
# real_audio_directory='/home/ubuntu/OpenVoice/emotion_STS/melo_emo/eval_input_audio/real'
# emo_directatry='/home/ubuntu/OpenVoice/emotion_STS/eval_data/esd/emo'
# wav_pattern='**/*.wav'
# emo_pattern='**/*.wav.emo.npy'
# # df=pd.DataFrame()
# for file in glob(os.path.join(real_audio_directory, wav_pattern), recursive=True):
#     audiopath=[file]
#     file_emo=file.split('/')[-1].split('_')[-1].split('.')[0].lower()
#     speaker=file.split('/')[-1].split('_')[1] + '_' + file.split('/')[-1].split('_')[2].split('.')[0]
#     for file2 in glob(os.path.join(emo_directatry, emo_pattern), recursive=True):
#         file2_emo=file2.split('/')[-1].split('-')[1].lower()
#         # print(file2)
#         if file_emo !=file2_emo:
#             # print(file2_emo)
#             # print(file_emo)
#             # print('speaker',speaker)
#             # input()
#             output_path=f'{output_dir}/{speaker}_{file2_emo}_{ckpt_num}.wav'
#             emo=file2
#             model.sts_to_file(audiopath,emo,output_path)









# audiopath=['/home/ubuntu/efs/sts/Emotion_Speech_Dataset/0009/Happy/0009_000703.wav']
# audiopath=['/home/ubuntu/OpenVoice/data/esd/ch/input/0010_000015_neu.wav']
audiopath=['/home/ubuntu/efs/sts/Emotion_Speech_Dataset/0009/Neutral/0009_000150.wav']
# audiopath=['/home/ubuntu/efs/sts/Emotion_Speech_Dataset/0006/Neutral/0006_000075.wav']

# emo='/home/ubuntu/efs/sts/Emotion_Speech_Dataset/0009/Happy/0009_000703.wav.emo.npy'
# emo='/home/ubuntu/efs/sts/Emotion_Speech_Dataset/0009/Happy/0009_000707.wav.emo.npy'
emo='/home/ubuntu/OpenVoice/data/esd/ch/ref/0008_000444_ang.wav.emo.npy'

model.sts_to_file(audiopath,emo,output_path)
