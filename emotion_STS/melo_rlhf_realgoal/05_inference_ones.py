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
# ckpt_path='/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf_realgoal/logs_emo/example/G_58630.pth'
ckpt_path='/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf_realgoal/logs_emo/example/G_56270.pth'
model = STS(device=device,config_path=config_path,ckpt_path=ckpt_path)
speaker_ids = model.hps.data.spk2id
cwd=os.getcwd()
output_directory=ckpt_path.split('/')[-3] + '/' + ckpt_path.split('/')[-2]
output_dir=f'{cwd}/{output_directory}/evaluation/real'
ckpt_num=ckpt_path.split('/')[-1].split('.')[0]


# for speaker_key in speaker_ids.keys():
# speaker_id = speaker_ids['4362']
output_path=f'{output_dir}/real/sed852_angry_happy_G_56270_realemo.wav'
# output_path=f'{output_dir}/real/cremad_neutral_angry_G_58650_1_realemo.wav'
print(output_path)
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








# audiopath=['/home/ubuntu/OpenVoice/emotion_STS/melo/generate_audio/IWW_friendly_female.wav']
# audiopath=['/home/ubuntu/OpenVoice/emotion_STS/melo/generate_audio/458_2922_cheerful.wav']
# audiopath=['/home/ubuntu/OpenVoice/emotion_STS/melo_emo/302_61159_angry.wav']
# audiopath=['/home/ubuntu/OpenVoice/emotion_STS/melo_emo/2289_20782_excited.wav']


# audiopath=['/home/ubuntu/OpenVoice/data/full_dataset/4362_48858_cheerful.wav']
# audiopath=['/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf/eval_input_audio/real/0011_000232_neutral.wav']
audiopath=['/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf/eval_input_audio/real/0011_000852_happy.wav']
# audiopath=['/home/ubuntu/efs/sts/emo_v/bea/neutral_225-252_0251.wav']

# audiopath=['/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf_realgoal/eval_input_audio/real/0011_000488_angry.wav']
# audiopath=['/home/ubuntu/efs/sts/CREMA-D/AudioWAV/1091_IOM_NEU_XX.wav']
# audiopath=['/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf/eval_input_audio/real/0014_000892_happy.wav']
# audiopath=['/home/ubuntu/OpenVoice/data/full_dataset/1054_MTI_SAD_XX.wav']
# audiopath=['/home/ubuntu/OpenVoice/emotion_STS/melo_emo/eval_input_audio/real/0011_001486_surprise.wav']



# audiopath=['/home/ubuntu/OpenVoice/emotion_STS/melo_emo/generate_audio/4362_48858_cheerful.wav']
# audiopath=['/home/ubuntu/OpenVoice/emotion_STS/melo/generate_audio/7367_30510_angry.wav']
# audiopath=['/home/ubuntu/OpenVoice/emotion_STS/melo/generate_audio/5561_67864_angry.wav']
# audiopath=['/home/ubuntu/OpenVoice/emotion_STS/melo_emo/generate_audio/34281_angry_tts.wav']
# audiopath=['/home/ubuntu/OpenVoice/emotion_STS/melo_emo/generate_audio/32_30510_angry.wav']
# audiopath=['/home/ubuntu/OpenVoice/emotion_STS/melo_emo/generate_audio/4362_777_angry.wav']
# audiopath=['/home/ubuntu/OpenVoice/emotion_STS/esd/input/0013_000017.wav']

# audiopath=['/home/ubuntu/OpenVoice/emotion_STS/melo/generate_audio/4362_48858_cheerful.wav']

# audiopath=['/home/ubuntu/OpenVoice/emotion_STS/esd/input/0013_000017.wav']

emo='/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf_realgoal/eval_ref_audio/real/Neutral-Angry-0013_000002.wav.emo.npy'
# emo='/home/ubuntu/efs/sts/emo_v/bea/anger_169-196_0161.wav.emo.npy'

# emo='/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf_realgoal/eval_ref_audio/real/Neutral-Happy-0013_000007.wav.emo.npy'
# emo='/home/ubuntu/efs/sts/CREMA-D/AudioWAV/1080_TIE_ANG_XX.wav.emo.npy'
# emo='/home/ubuntu/efs/sts/train/tts/51028_sad_tts.wav.emo.npy'
# emo='/home/ubuntu/efs/CREMA-D/AudioWAV/1001_MTI_ANG_XX.wav.emo.npy'
# emo='/home/ubuntu/OpenVoice/data/full_dataset/34281_angry_tts.wav.emo.npy'
# emo='/home/ubuntu/OpenVoice/emotion_STS/esd/emo/Neutral-Angry-0013_000020.wav.emo.npy'
# emo='/home/ubuntu/efs/sts/train/tts/34503_friendly_tts.wav.emo.npy'
# emo='/home/ubuntu/OpenVoice/emotion_STS/esd/emo/Neutral-Surprise-0013_000017.wav.emo.npy'
# emo='/home/ubuntu/OpenVoice/emotion_STS/esd/emo/Neutral-Angry-0013_000017.wav.emo.npy'

# emo='/home/ubuntu/OpenVoice/emotion_STS/melo/generate_audio/42702_sad_tts.wav.emo.npy'
# emo='/home/ubuntu/OpenVoice/emotion_STS/melo/generate_audio/24542_shouting_tts.wav.emo.npy'


model.sts_to_file(audiopath,emo,output_path)
