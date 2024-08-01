#!/home/ubuntu/OpenVoice/.openvoiceenv/bin/python
# from speechbrain.inference.ASR import EncoderDecoderASR
from evaluate import load
from glob import glob
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import re
from jiwer import wer
# from datasets import load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)
def remove_non_letters(text):
        return re.sub(r'[^a-zA-Z\s]', '', text)
# wer = load("wer")
val=[]
dictionary={}
real_audio_file='/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf_realgoal/logs_emo/example/evaluation/G_56480_real/eval.csv'
real_audio_data=pd.read_csv(real_audio_file)
df_real=pd.DataFrame(real_audio_data)
for i_r, row_r in df_real.iterrows():
    ref_audio=row_r['reference_audio'].split('.')[0]+'.wav'
    input_audio=row_r['input_audio']
    output_audio=row_r['output_audio']
    # y =classifier_skid.extract_embedding(input_audio,16000)
    # y1 = classifier_skid.extract_embedding(output_audio)
    #whisper
    predict_a1 = pipe(input_audio)
    input_a1 = pipe(output_audio)

    predictions1 = [predict_a1['text'].lower()]
    references1 = [input_a1['text'].lower()]
    
    cleaned_predictions1 = [remove_non_letters(s) for s in predictions1]
    cleaned_references1 = [remove_non_letters(s) for s in references1]
    print('predictions1',cleaned_predictions1)
    print('references1',cleaned_references1)

    # wer_score1 = wer.compute(predictions=cleaned_predictions1, references=cleaned_references1)
    wer_score1 = wer(cleaned_predictions1, cleaned_references1)

    val.append(wer_score1)
    # print('file',i)
    # print('input',file2)
    # dictionary[name]=wer_score1
    print('wer_score1',wer_score1)
val_mean=np.mean(val)
# print('val',val)
print('val_mean',val_mean)
# print(dictionary)



# ###AINN
# alist=[]
# val=[]
# dictionary={}
# pattern='**/*.wav'
# # asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-wav2vec2-commonvoice-en", savedir="pretrained_models/asr-wav2vec2-commonvoice-en")
# wer = load("wer")
# directory='/home/ubuntu/OpenVoice/data/AINN/output'
# directoryesd='/home/ubuntu/OpenVoice/data/AINN/input'
# for file in glob(os.path.join(directory, pattern), recursive=True):
#     alist.append(file)
# for file2 in glob(os.path.join(directoryesd, pattern), recursive=True):
#     name=file2.split('/')[-1].split('.')[0]
#     for i in alist:
#         filename=str(i).split('/')[-1].split('_')[0]+'_'+str(i).split('/')[-1].split('_')[1]
#         # print(filename2)
#         if filename ==name:
#             ###speechbrain
#             # predict_a1=asr_model.transcribe_file(i)
#             # input_a1=asr_model.transcribe_file(file2)
#             #whisper
#             predict_a1 = pipe(i)
#             input_a1 = pipe(file2)

#             # print(result["text"])
#             predictions1 = [predict_a1['text']]
#             references1 = [input_a1['text']]
#             # print('predictions1',predictions1)
#             # print('references1',references1)

#             wer_score1 = wer.compute(predictions=predictions1, references=references1)
#             val.append(wer_score1)
#             # print('file',i)
#             # print('input',file2)
#             dictionary[name]=wer_score1
# val_mean=np.mean(val)
# print('val',val)
# print(val_mean)
# print(dictionary)






# predictions1 = [predict_a1]
# references1 = [input_a1]
# wer_score1 = wer.compute(predictions=predictions1, references=references1)

# predict_a2=asr_model.transcribe_file("/home/ubuntu/OpenVoice/emotion_STS/melo_emo/evaluation/458_2922_tmp130000_sad.wav")
# input_a2=asr_model.transcribe_file("/home/ubuntu/OpenVoice/emotion_STS/melo_emo/generate_audio/458_2922_cheerful.wav")
# # wer = load("wer")
# predictions2 = [predict_a2]
# references2 = [input_a2]
# wer_score2 = wer.compute(predictions=predictions2, references=references2)

# predict_a3=asr_model.transcribe_file("/home/ubuntu/OpenVoice/emotion_STS/melo_emo/evaluation/302_61159_tmp130000_friendly.wav")
# input_a3=asr_model.transcribe_file("/home/ubuntu/OpenVoice/emotion_STS/melo_emo/generate_audio/302_61159_angry.wav")
# # wer = load("wer")
# predictions3 = [predict_a3]
# references3 = [input_a3]
# wer_score3 = wer.compute(predictions=predictions3, references=references3)
# print(predict_a1)
# print(input_a1)
# print(wer_score1)
# print(predict_a2)
# print(input_a2)
# print(wer_score2)
# print(predict_a3)
# print(input_a3)
# print(wer_score3)
