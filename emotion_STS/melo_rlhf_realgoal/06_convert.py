#!/home/ubuntu/OpenVoice/.openvoiceenv/bin/python
# import matplotlib.pyplot as plt
# import IPython.display as ipd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import soundfile
import utils
from data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
import librosa
from scipy.io.wavfile import write
import numpy as np
from mel_processing import spectrogram_torch
import se_extractor
from api import STS
hps = utils.get_hparams_from_file("/home/ubuntu/OpenVoice/emotion_STS/melo/logs/example/config.json")
print('hps',hps)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

config_path='/home/ubuntu/OpenVoice/emotion_STS/melo/logs/example/config.json'
# ckpt_path='/home/ubuntu/OpenVoice/emotion_STS/melo/logs/example/G_5000.pth'
ckpt_path='/home/ubuntu/OpenVoice/emotion_STS/melo/logs/example/G_13000.pth'
model = STS(device=device,config_path=config_path,ckpt_path=ckpt_path)
speaker_ids = model.hps.data.spk2id
output_dir='/home/ubuntu/OpenVoice/emotion_STS/melo/generate_audio'
output_path=f'{output_dir}/tmp13000_5.wav'
for speaker_key in speaker_ids.keys():
        speaker_id = speaker_ids[speaker_key]
        speaker_key = speaker_key.lower().replace('_', '-')
        audiopath=['/home/ubuntu/OpenVoice/synthesized_speech/s_txt/IWW_friendly_female.wav']
        # audiopath=['/home/ubuntu/OpenVoice/synthesized_speech/new_ref_/SL_cheerful_female.wav']
        emo_target='/home/ubuntu/OpenVoice/data/synthesized_data/24542_shouting_tts.wav.emo.npy'
        # speaker_id=speaker_id.to(device)
        # audiopath=audiopath.to(device)
        # emo=emo.to(device)
        # output_path=output_path.to(device)
        tone_color_converter.convert(
            audio_src_path=src_path, 
            src_se=source_se_base, 
            tgt_se=target_se, 
            output_path=save_path,
            message=encode_message)
        model.sts_to_file(audiopath,speaker_id,emo_tar=emo_target,output_path)
