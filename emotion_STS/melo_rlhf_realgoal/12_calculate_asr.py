#!/home/ubuntu/OpenVoice/.openvoiceenv/bin/python
# from speechbrain.inference.ASR import EncoderDecoderASR
# from evaluate import load
from glob import glob
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import re
from jiwer import wer
import jieba
import jiwer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from difflib import SequenceMatcher
normalizer = BasicTextNormalizer()
# from datasets import load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

##english
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

def tokenize_chinese(text):
    """Segment Chinese text into words."""
    return ' '.join(jieba.cut(text))

def calculate_wer(predictions, references):
    """Calculate Chinese Word Error Rate (WER)."""
    # Tokenize Chinese text
    pred_text = predictions['text']
    ref_text = references['text']
    ref_tokens = tokenize_chinese(ref_text)
    hyp_tokens = tokenize_chinese(pred_text)
    
    # Calculate WER using jiwer
    wer = jiwer.wer(ref_tokens, hyp_tokens)
    
    return wer

def calculate_character_wer(reference_text, hypothesis_text):
    ref_chars = list(reference_text)
    hyp_chars = list(hypothesis_text)
    
    s, d, i = 0, 0, 0  # Substitutions, deletions, insertions

    matcher = SequenceMatcher(None, ref_chars, hyp_chars)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            s += (i2 - i1)  # Number of substitutions
            i += (j2 - j1)  # Number of substitutions
        elif tag == 'delete':
            d += (i2 - i1)  # Number of deletions
        elif tag == 'insert':
            i += (j2 - j1)  # Number of insertions

    N = len(ref_chars)  # Total number of reference characters
    character_wer = (s + d + i) / N

    return character_wer

# ##chinese
# transcriber = pipeline(
#   "automatic-speech-recognition", 
#   model="jonatasgrosman/whisper-large-zh-cv11"
# )

# transcriber.model.config.forced_decoder_ids = (
#   transcriber.tokenizer.get_decoder_prompt_ids(
#     language="zh", 
#     task="transcribe"
#   )
# )

def remove_non_letters(text):
        return re.sub(r'[^a-zA-Z\s]', '', text)
##English
# wer = load("wer")
val=[]
dictionary={}
real_audio_file='/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf_realgoal/ckpts/ESD_cheng/evaluation/G_56230_real/eval.csv'
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
    cleaned_predictions1 = [normalizer(s) for s in predictions1]
    cleaned_references1 = [normalizer(s) for s in references1]
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

# ##Chinese
# # wer = load("wer")
# val=[]
# dictionary={}
# real_audio_file='/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf_realgoal/ckpts/ESD_cheng/evaluation/G_56230_real/eval.csv'
# real_audio_data=pd.read_csv(real_audio_file)
# df_real=pd.DataFrame(real_audio_data)
# for i_r, row_r in df_real.iterrows():
#     ref_audio=row_r['reference_audio'].split('.')[0]+'.wav'
#     input_audio=row_r['input_audio']
#     output_audio=row_r['output_audio']
#     # y =classifier_skid.extract_embedding(input_audio,16000)
#     # y1 = classifier_skid.extract_embedding(output_audio)
#     #whisper
#     predict_a1 = pipe(input_audio)
#     input_a1 = pipe(output_audio)

#     print('predict_a1',predict_a1)
#     print('input_a1',input_a1)


#     # wer_score1 = wer.compute(predictions=cleaned_predictions1, references=cleaned_references1)
#     wer_score1 = calculate_wer(predict_a1, input_a1)
#     if wer_score1>1:
#          wer_score1=1
#     val.append(wer_score1)
#     # print('file',i)
#     # print('input',file2)
#     # dictionary[name]=wer_score1
#     print('wer_score1',wer_score1)
# val_mean=np.mean(val)
# # print('val',val)
# print('val_mean',val_mean)
# # print(dictionary)



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

# ###seq2seq
# alist=[]
# val=[]
# dictionary={}
# pattern='**/*.wav'
# # asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-wav2vec2-commonvoice-en", savedir="pretrained_models/asr-wav2vec2-commonvoice-en")
# # wer = load("wer")
# directory='/home/ubuntu/OpenVoice/data/cyclegan'
# directoryesd='/home/ubuntu/OpenVoice/data/input'
# for file in glob(os.path.join(directory, pattern), recursive=True):
#     alist.append(file)
# for file2 in glob(os.path.join(directoryesd, pattern), recursive=True):
#     name=file2.split('/')[-1].split('.')[0]
#     # print('name',name)
#     for i in alist:
#         filename=str(i).split('/')[-1].split('-')[-1].split('.')[0]
#         print(i)
#         # print('filename',filename)
#         # input()
#         if filename ==name:
#             ###speechbrain
#             # predict_a1=asr_model.transcribe_file(i)
#             # input_a1=asr_model.transcribe_file(file2)
#             #whisper
#             predict_a1 = pipe(i)
#             input_a1 = pipe(file2)

#             # print(result["text"])
#             predictions1 = [predict_a1['text'].lower()]
#             references1 = [input_a1['text'].lower()]
#             cleaned_predictions1 = [remove_non_letters(s) for s in predictions1]
#             cleaned_references1 = [remove_non_letters(s) for s in references1]
#             print('predictions1',predictions1)
#             print('references1',references1)
#             # input()
#             wer_score1 = wer(cleaned_predictions1, cleaned_references1)
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



# predict_a3=pipe("/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf_realgoal/logs_emo_esd-11/example/evaluation/0007_000775.wav")
# input_a3=pipe("/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf_realgoal/logs_emo/example/evaluation/real/real/hap_sad_ch_G_56740.wav")
# # wer = load("wer")
# predictions3 = [predict_a3]
# references3 = [input_a3]
# print(predict_a3)
# print(input_a3)
# # input()
# wer_score3 = calculate_wer(predict_a3, input_a3)
# # print(predict_a1)
# # print(input_a1)
# # print(wer_score1)
# # print(predict_a2)
# # print(input_a2)
# # print(wer_score2)

# print(wer_score3)
