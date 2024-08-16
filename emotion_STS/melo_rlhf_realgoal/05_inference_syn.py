#!/home/ubuntu/OpenVoice/.openvoiceenv/bin/python
# import matplotlib.pyplot as plt
# import IPython.display as ipd
import torch
import utils
from api import STS
import os
import pandas as pd
from glob import glob

hps = utils.get_hparams_from_file("/home/ubuntu/OpenVoice/emotion_STS/melo/logs/example/config.json")
print('hps',hps)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

config_path='/home/ubuntu/OpenVoice/emotion_STS/melo/logs/example/config.json'
# ckpt_path='/home/ubuntu/OpenVoice/emotion_STS/melo/logs/example/G_6600.pth'
# ckpt_path='/home/ubuntu/OpenVoice/emotion_STS/melo_emo/ckpt/G_46800.pth'
ckpt_path='/home/ubuntu/OpenVoice/emotion_STS/melo_emo/logs_exp6_logp/example/G_55400.pth'
model = STS(device=device,config_path=config_path,ckpt_path=ckpt_path)
speaker_ids = model.hps.data.spk2id
cwd=os.getcwd()
output_directory=ckpt_path.split('/')[-3] + '/' + ckpt_path.split('/')[-2]
output_dir=f'{cwd}/{output_directory}/evaluation/syn_input_ref'
ckpt_num=ckpt_path.split('/')[-1].split('.')[0]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

real_audio_directory='/home/ubuntu/OpenVoice/emotion_STS/melo_emo/eval_input_audio/synthesized'
emo_directatry='/home/ubuntu/OpenVoice/emotion_STS/melo_emo/eval_ref_audio/synthesized'
text='/home/ubuntu/OpenVoice/emotion_STS/melo_emo/data/example/doc_text.csv'
text=pd.read_csv(text)
text_index_dict = text.set_index('text_id')['text'].to_dict()
# print('text_index_dict',text_index_dict)
# input()
wav_pattern='**/*.wav'
emo_pattern='**/*.wav.emo.npy'
df=pd.DataFrame()
inf={}

for file in glob(os.path.join(real_audio_directory, wav_pattern), recursive=True):
    audiopath=[file]
    file_emo=file.split('/')[-1].split('_')[-1].split('.')[0].lower()
    speaker=file.split('/')[-1].split('_')[1] + '_' + file.split('/')[-1].split('_')[2].split('.')[0]
    for file2 in glob(os.path.join(emo_directatry, emo_pattern), recursive=True):
       ##for eval_ref_audio/synthesized
        # file2_emo=file2.split('/')[-1].split('_')[1].lower()

        ##for eval_ref_audio/real
        # print(file)
        # input()
        file2_emo=file2.split('/')[-1].split('_')[1].lower()
        text_id=file.split('/')[-1].split('_')[1]
        if file_emo !=file2_emo:
            inf['text']=text_index_dict[text_id]
            inf['reference_audio']=file2
            inf['input_audio']=file
            output_path=f'{output_dir}/{speaker}_to_{file2_emo}_{ckpt_num}.wav'
            inf['output_audio']=output_path
            emo=file2
            model.sts_to_file(audiopath,emo,output_path)
            df = pd.concat([df, pd.DataFrame([inf])], ignore_index=True)
df.to_csv(f'{output_dir}/eval.csv')









