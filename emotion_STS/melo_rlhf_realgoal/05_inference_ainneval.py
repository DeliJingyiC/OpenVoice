#!/home/ubuntu/OpenVoice/.openvoiceenv/bin/python
# import matplotlib.pyplot as plt
# import IPython.display as ipd
import torch
import utils
from api import STS
import os
import pandas as pd
from glob import glob

hps = utils.get_hparams_from_file("/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf_realgoal/logs_emo_/example/config.json")
print('hps',hps)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

config_path='/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf_realgoal/logs_emo_/example/config.json'
# ckpt_path='/home/ubuntu/OpenVoice/emotion_STS/melo/logs/example/G_6600.pth'
# ckpt_path='/home/ubuntu/OpenVoice/emotion_STS/melo_emo/ckpt/G_46800.pth'
ckpt_path='/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf_realgoal/logs_emo/example/G_56480.pth'
model = STS(device=device,config_path=config_path,ckpt_path=ckpt_path)
speaker_ids = model.hps.data.spk2id
cwd=os.getcwd()
output_directory=ckpt_path.split('/')[-3] + '/' + ckpt_path.split('/')[-2]
output_dir=f'{cwd}/{output_directory}/evaluation/G_56480_real'
ckpt_num=ckpt_path.split('/')[-1].split('.')[0]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# for speaker_key in speaker_ids.keys():
# speaker_id = speaker_ids['4362']
# output_path=f'{output_dir}/32_30510_friendly_tmp25500.wav'
# output_path=f'{output_dir}/0013_000017_surprise_46800_new.wav'
# output_path=f'{output_dir}/32_30510_friendly_tmp25500_new.wav'
# output_path=f'{output_dir}/Angry-0013_000017_20_tmp46400_new.wav'
# real_audio_directory='/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf_realgoal/eval_input_audio/real'
real_audio_directory='/home/ubuntu/OpenVoice/data/AINN/input'

# real_audio_directory='/home/ubuntu/OpenVoice/data/syn/synthesized_input'
# emo_directatry='/home/ubuntu/OpenVoice/data/syn/synthesized_emo'
emo_directatry='/home/ubuntu/OpenVoice/data/AINN/ref'
AINN_output='/home/ubuntu/OpenVoice/data/AINN/output'
wav_pattern='**/*.wav'
emo_pattern='**/*.wav.emo.npy'
df=pd.DataFrame()
inf={}
for ainn_file in glob(os.path.join(AINN_output, wav_pattern), recursive=True):
    input_name=ainn_file.split('/')[-1].split('_')[0]+'_'+ainn_file.split('/')[-1].split('_')[1]
    ref_name=ainn_file.split('/')[-1].split('_')[2]+'_'+ainn_file.split('/')[-1].split('_')[3]

    for file in glob(os.path.join(real_audio_directory, wav_pattern), recursive=True):
        audiopath=[file]
        file_input=file.split('/')[-1].split('.')[0]

        for file2 in glob(os.path.join(emo_directatry, emo_pattern), recursive=True):
            
            file2_ref=file2.split('/')[-1].split('.')[0]

            if file_input ==input_name and file2_ref==ref_name:
                inf['reference_audio']=file2
                inf['input_audio']=file
                output_path=f'{output_dir}/{input_name}_to_{file2_ref}_{ckpt_num}.wav'
                inf['output_audio']=output_path
                emo=file2
                model.sts_to_file(audiopath,emo,output_path)
                df = pd.concat([df, pd.DataFrame([inf])], ignore_index=True)
df.to_csv(f'{output_dir}/eval.csv')









# audiopath=['/home/ubuntu/OpenVoice/emotion_STS/melo/generate_audio/IWW_friendly_female.wav']
# audiopath=['/home/ubuntu/OpenVoice/emotion_STS/melo/generate_audio/458_2922_cheerful.wav']
# audiopath=['/home/ubuntu/OpenVoice/emotion_STS/melo_emo/302_61159_angry.wav']
# audiopath=['/home/ubuntu/OpenVoice/emotion_STS/melo_emo/2289_20782_excited.wav']


# audiopath=['/home/ubuntu/efs/sts/train/convert/4362_48858_cheerful.wav']
# audiopath=['/home/ubuntu/OpenVoice/emotion_STS/melo_emo/generate_audio/4362_48858_cheerful.wav']
# audiopath=['/home/ubuntu/OpenVoice/emotion_STS/melo/generate_audio/7367_30510_angry.wav']
# audiopath=['/home/ubuntu/OpenVoice/emotion_STS/melo/generate_audio/5561_67864_angry.wav']
# audiopath=['/home/ubuntu/OpenVoice/emotion_STS/melo_emo/generate_audio/34281_angry_tts.wav']
# audiopath=['/home/ubuntu/OpenVoice/emotion_STS/melo_emo/generate_audio/32_30510_angry.wav']
# audiopath=['/home/ubuntu/OpenVoice/emotion_STS/melo_emo/generate_audio/4362_777_angry.wav']
# audiopath=['/home/ubuntu/OpenVoice/emotion_STS/esd/input/0013_000017.wav']

# audiopath=['/home/ubuntu/OpenVoice/emotion_STS/melo/generate_audio/4362_48858_cheerful.wav']

# audiopath=['/home/ubuntu/OpenVoice/emotion_STS/esd/input/0013_000017.wav']

# emo='/home/ubuntu/efs/sts/train/tts/51028_sad_tts.wav.emo.npy'
# emo='/home/ubuntu/OpenVoice/emotion_STS/melo/generate_audio/34281_angry_tts.wav.emo.npy'
# emo='/home/ubuntu/OpenVoice/emotion_STS/esd/emo/Neutral-Angry-0013_000020.wav.emo.npy'
# emo='/home/ubuntu/efs/sts/train/tts/34503_friendly_tts.wav.emo.npy'
# emo='/home/ubuntu/OpenVoice/emotion_STS/esd/emo/Neutral-Surprise-0013_000017.wav.emo.npy'
# emo='/home/ubuntu/OpenVoice/emotion_STS/esd/emo/Neutral-Angry-0013_000017.wav.emo.npy'

# emo='/home/ubuntu/OpenVoice/emotion_STS/melo/generate_audio/42702_sad_tts.wav.emo.npy'
# emo='/home/ubuntu/OpenVoice/emotion_STS/melo/generate_audio/24542_shouting_tts.wav.emo.npy'


# model.sts_to_file(audiopath,emo,output_path)
