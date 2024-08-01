#!/home/ubuntu/OpenVoice/.openvoiceenv/bin/python
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
import os
import librosa
import numpy as np
import pandas as pd
from glob import glob
from numpy import dot
from numpy.linalg import norm
from scipy.spatial.distance import cosine, euclidean


class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits


# load model from hub
device = 'cuda' if torch.cuda.is_available() else "cpu"
model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = EmotionModel.from_pretrained(model_name).to(device)


def process_func(
        x: np.ndarray,
        sampling_rate: int,
        embeddings: bool = False,
) -> np.ndarray:
    r"""Predict emotions or extract embeddings from raw audio signal."""

    # run through processor to normalize signal
    # always returns a batch, so we just get the first entry
    # then we put it on the device
    
    
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0)
    # run through model
    with torch.no_grad():
        y = model(y)[0 if embeddings else 1]

    # convert to numpy
    y = y.detach().cpu().numpy()

    return y


# #
# #
# # def disp(rootpath, wavname):
# #     wav, sr = librosa.load(f"{rootpath}/{wavname}", 16000)
# #     display(ipd.Audio(wav, rate=sr))

# # rootpath = "dataset/nene"
# embs = []
# wavnames = []


# def extract_dir(path):
#     rootpath = path
#     for idx, wavname in enumerate(os.listdir(rootpath)):
#         wav, sr = librosa.load(f"{rootpath}/{wavname}", 16000)
#         emb = process_func(np.expand_dims(wav, 0), sr, embeddings=True)
#         embs.append(emb)
#         wavnames.append(wavname)
#         np.save(f"{rootpath}/{wavname}.emo.npy", emb.squeeze(0))
#         print(idx, wavname)


# def extract_wav(path):
#     wav, sr = librosa.load(path, 16000)
#     emb = process_func(np.expand_dims(wav, 0), sr, embeddings=True)
#     return emb


# def preprocess_one(path):
#     wav, sr = librosa.load(path, 16000)
#     # print('wave.shape',wav.shape)
#     # print('wave.shape',np.expand_dims(wav, 0).shape)
#     # input()
#     emb = process_func(np.expand_dims(wav, 0), sr, embeddings=True)
#     # print('emb')
#     np.save(f"{path}.emo.npy", emb.squeeze(0))
#     return emb


if __name__ == '__main__':

# # ###AINN
#     val=[]
#     pattern='**/*.wav'
#     directory='/home/ubuntu/OpenVoice/data/AINN/output'
#     directoryesd='/home/ubuntu/OpenVoice/data/AINN/ref'
#     alist=[]

#     for file in glob(os.path.join(directory, pattern), recursive=True):
#         alist.append(file)
#     for file2 in glob(os.path.join(directoryesd, pattern), recursive=True):
#         name=file2.split('/')[-1].split('_')[0]
#         # print(file2.split('.')[0].split('-'))
#         # input()
#         emo_l=file2.split('/')[-1].split('_')[-1].split('.')[0]

#         for i in alist:
#             filename=str(i).split('/')[-1].split('_')[0]
#             emo=str(i).split('/')[-1].split('_')[-2]
#             # print('filename',filename)
#             # print('emo',emo)
#             # print('name',name)
#             # print('emo_l',emo_l)
#             # input()

#             if filename ==name and emo==emo_l:
#                 wav, sr = librosa.load(file2, 16000)
#                 wav1, sr1 = librosa.load(i, 16000)
#                 y = process_func(np.expand_dims(wav, 0), sr, embeddings=True)[0]
#                 y1 = process_func(np.expand_dims(wav1, 0), sr, embeddings=True)[0]
#                 cos_sim = dot(y, y1)/(norm(y)*norm(y1))
#                 cos_sim=(cos_sim+1.0)/2
#                 val.append(cos_sim)
#     val_mean=np.mean(val)
#     print(val)
#     print('mean',val_mean)

# ###our_SPS_Eval
#     val=[]
#     pattern='**/*.wav'
#     directory='/home/ubuntu/OpenVoice/emotion_STS/melo_emo/evaluation'
#     directoryesd='/home/ubuntu/OpenVoice/emotion_STS/eval_data/esd/emo'
#     alist=[]

#     for file in glob(os.path.join(directory, pattern), recursive=True):
#         alist.append(file)
#     for file2 in glob(os.path.join(directoryesd, pattern), recursive=True):
#         name=file2.split('.')[0].split('-')[-1]
#         # print(file2.split('.')[0].split('-'))
#         # input()
#         emo_l=file2.split('.')[0].split('-')[-2]
#         emo_l=emo_l.lower()
#         # print(name)
#         # input()
#         for i in alist:
#             filename=str(i).split('/')[-1].split('_')
#             filename2=filename[0]+'_'+filename[1]
#             emo=filename[2]
#             # print(filename)
#             # input()
#             # print(filename2)
#             if filename2 ==name and emo==emo_l:
#                 wav, sr = librosa.load(file2, 16000)
#                 wav1, sr1 = librosa.load(i, 16000)
#                 y = process_func(np.expand_dims(wav, 0), sr, embeddings=True)[0]
#                 y1 = process_func(np.expand_dims(wav1, 0), sr, embeddings=True)[0]
#                 cos_sim = dot(y, y1)/(norm(y)*norm(y1))
#                 (cos_sim=cos_sim+1.0)/2
#                 val.append(cos_sim)
#     val_mean=np.mean(val)
#     print(val)
#     print(val_mean)
######Seq2Seq-EVC
    # val=[]
    # pattern='**/*.wav'
    # directory='/home/ubuntu/OpenVoice/emotion_STS/eval_data/Seq2Seq-EVC'
    # directoryesd='/home/ubuntu/OpenVoice/emotion_STS/eval_data/esd/emo'
    # alist=[]

    # for file in glob(os.path.join(directory, pattern), recursive=True):
    #     alist.append(file)
    # for file2 in glob(os.path.join(directoryesd, pattern), recursive=True):
    #     name=file2.split('.')[0].split('-')[-1]

    #     emo_l=file2.split('.')[0].split('-')[-2]
    #     print(emo_l)
    #     # input()
    #     for i in alist:
    #         filename=str(i).split('.')[0].split('/')[-1]
    #         # filename2=filename[0]+'_'+filename[1]
    #         emo=str(i).split('.')[-2].split('_')[-2]
    #         # print(emo)
    #         # print(filename)
    #         # print(name)

    #         # input()
    #         if filename ==name and emo==emo_l:

    #             wav, sr = librosa.load(file2, 16000)
    #             wav1, sr1 = librosa.load(i, 16000)
    #             y = process_func(np.expand_dims(wav, 0), sr, embeddings=True)[0]
    #             y1 = process_func(np.expand_dims(wav1, 0), sr, embeddings=True)[0]
    #             cos_sim = dot(y, y1)/(norm(y)*norm(y1))
                #   (cos_sim=cos_sim+1.0)/2

    #             val.append(cos_sim)
    # val_mean=np.mean(val)
    # print(val)
    # print(val_mean)

########emotionalstartgan
    # val=[]
    # pattern='**/*.wav'
    # directory='/home/ubuntu/OpenVoice/emotion_STS/eval_data/emotionalstartgan'
    # directoryesd='/home/ubuntu/OpenVoice/emotion_STS/eval_data/esd/emo'
    # alist=[]

    # for file in glob(os.path.join(directory, pattern), recursive=True):
    #     alist.append(file)
    # for file2 in glob(os.path.join(directoryesd, pattern), recursive=True):
    #     name=file2.split('.')[0].split('-')[-1]
    #     emo_l=file2.split('.')[0].split('-')[-2]

    #     for i in alist:
    #         filename=str(i).split('-')[0].split('/')[-1]
    #         # filename2=filename[0]+'_'+filename[1]
    #         emo=str(i).split('-')[-1].split('.')[0]
    #         # print(filename)
    #         # print(name)
    #         # print(emo_l)
    #         # print(emo)
    #         # input()
    #         if filename ==name and emo==emo_l:
    #             wav, sr = librosa.load(file2, 16000)
    #             wav1, sr1 = librosa.load(i, 16000)
    #             y = process_func(np.expand_dims(wav, 0), sr, embeddings=True)[0]
    #             y1 = process_func(np.expand_dims(wav1, 0), sr, embeddings=True)[0]
    #             cos_sim = dot(y, y1)/(norm(y)*norm(y1))
                #   (cos_sim=cos_sim+1.0)/2

    #             val.append(cos_sim)
    # val_mean=np.mean(val)
    # print(val)
    # print(val_mean)

    # #####cycleGAN
    # val=[]
    # pattern='**/*.wav'
    # directory='/home/ubuntu/OpenVoice/emotion_STS/eval_data/cyclegan'
    # directoryesd='/home/ubuntu/OpenVoice/emotion_STS/eval_data/esd/emo'
    # alist=[]

    # for file in glob(os.path.join(directory, pattern), recursive=True):
    #     alist.append(file)
    # for file2 in glob(os.path.join(directoryesd, pattern), recursive=True):
    #     name=file2.split('.')[0].split('-')[-1]
    #     emo_l=file2.split('.')[0].split('-')[-2]
    #     # print(name)
    #     for i in alist:
    #         filename=str(i).split('-')[-1].split('.')[0]
    #     # filename2=filename[0]+'_'+filename[1]
    #     # print(filename)
    #     # input()
    #         emo=str(i).split('.')[0].split('-')[-2]
    #         if filename ==name and emo==emo_l:
    #             wav, sr = librosa.load(file2, 16000)
    #             wav1, sr1 = librosa.load(i, 16000)
    #             y = process_func(np.expand_dims(wav, 0), sr, embeddings=True)[0]
    #             y1 = process_func(np.expand_dims(wav1, 0), sr, embeddings=True)[0]
    #             cos_sim = dot(y, y1)/(norm(y)*norm(y1))
    #             cos_sim=(cos_sim+1.0)/2

    #             val.append(cos_sim)
    # val_mean=np.mean(val)
    # print(val)
    # print(val_mean)

    # wav, sr = librosa.load('/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf_realgoal/logs_emo/example/evaluation/G_56370_real/000881_happy_to_surprise_G_56370.wav', 16000)
    # wav1, sr1 = librosa.load('/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf_realgoal/eval_ref_audio/real/Neutral-Surprise-0013_000017.wav', 16000)
    # y = process_func(np.expand_dims(wav, 0), sr, embeddings=True)[0]
    # y1 = process_func(np.expand_dims(wav1, 0), sr1, embeddings=True)[0]
    # print(y1)
    # cos_sim = dot(y, y1)/(norm(y)*norm(y1))
    # cos_sim=(cos_sim+1.0)/2

    # print(cos_sim)

    #eval rl model:
    # our_SPS_Eval2
    val=[]
    df = pd.DataFrame()
    adict={}
    real_audio_file='/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf_realgoal/logs_emo/example/evaluation/G_56480_real/eval.csv'
    real_audio_data=pd.read_csv(real_audio_file)
    df_real=pd.DataFrame(real_audio_data)
    for i_r, row_r in df_real.iterrows():
        ref_audio=row_r['reference_audio'].split('.')[0]+'.wav'
        input_audio=row_r['input_audio']
        output_audio=row_r['output_audio']
        wav, sr = librosa.load(ref_audio, 16000)
        wav1, sr1 = librosa.load(output_audio, 16000)
        y = process_func(np.expand_dims(wav, 0), sr, embeddings=True)[0]
        y1 = process_func(np.expand_dims(wav1, 0), sr, embeddings=True)[0]
        cos_sim = dot(y, y1)/(norm(y)*norm(y1))
        cos_sim=(cos_sim+1.0)/2
        adict['ref_audio']=ref_audio
        adict['input_audio']=input_audio
        adict['output_audio']=output_audio
        adict['cos_sim']=cos_sim
        df = pd.concat([df, pd.DataFrame([adict])],ignore_index=True)
        val.append(cos_sim)
        print(cos_sim)
    val_mean=np.mean(val)
    print('mean',val_mean)
    save_dir = os.path.dirname(real_audio_file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df.to_csv(f'{save_dir}/emo_eval.csv')
