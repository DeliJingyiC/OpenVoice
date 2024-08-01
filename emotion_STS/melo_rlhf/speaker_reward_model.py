import torch
import torch.nn as nn
# from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import wespeaker
import os
import librosa
import numpy as np
import pandas as pd
from glob import glob
from numpy import dot
from numpy.linalg import norm
import torchaudio


# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-sv')
# model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-sv')
# def speaker_func(x1,x2):
#     audio=[x1,x2]
#     inputs = feature_extractor(audio, padding=True, return_tensors="pt")
#     embeddings = model(**inputs).embeddings
#     embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()

#     # the resulting embeddings can be used for cosine similarity-based retrieval
#     cosine_sim = torch.nn.CosineSimilarity(dim=-1)
#     similarity = cosine_sim(embeddings[0], embeddings[1])
#     return similarity
# device = 'cuda' if torch.cuda.is_available() else "cpu"
model = wespeaker.load_model('english')

def speaker_func(x1,x2,sr):
    x1=x1.float()
    x2=x2.float()
    device=x1.device
    model.set_gpu(device.index)
    model.model = model.model.to(device)
    x1 = x1.to(device)
    x2 = x2.to(device)
     # Ensure model is on the correct device
    model_device = next(model.model.parameters()).device

    def extract_embedding_RL(pcm, sample_rate):


        if sample_rate != model.resample_rate:

            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=model.resample_rate)
            pcm = resampler(pcm.cpu()).to(device)


        # Remove extra dimension if present
        if pcm.dim() == 3 and pcm.size(0) == 1:
            pcm = pcm.squeeze(0)
        feats = model.compute_fbank(pcm, sample_rate=model.resample_rate, cmn=True).to(device)


        feats = feats.unsqueeze(0).to(device)  # Ensure features are on the same device as the model


        model.model.eval()


        with torch.no_grad():


            outputs = model.model(feats)


            outputs = outputs[-1] if isinstance(outputs, tuple) else outputs



        embedding = outputs[0]  # Keep the embedding on the same device

        return embedding

    e1=extract_embedding_RL(x1,sr)

    e2=extract_embedding_RL(x2,sr)


    if e1 is None or e2 is None:
            return 0.0
    else:

        return model.cosine_similarity(e1, e2), e1, e2
        # return [e1, e2]

