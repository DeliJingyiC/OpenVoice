
import os
import torch
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
from MeloTTS.melo.api import TTS
from MeCab import Tagger
import pandas as pd
import random
from glob import glob
import wave
import contextlib
from tqdm import tqdm

def find_files(directory, pattern='**/*.wav'):
    for file in glob(os.path.join(directory, pattern), recursive=True):
        duration =cal_duration(file)
        if duration >9:
            return file
        else:
             continue

def cal_duration(file):
    with contextlib.closing(wave.open(file, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return duration
ckpt_base = 'checkpoints/base_speakers/EN'
ckpt_converter = 'checkpoints_v2/converter'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = '/home/ubuntu/efs/sts/train/convert'
output_dir_tts = '/home/ubuntu/efs/sts/train/tts'


base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir_tts, exist_ok=True)

source_se_base = torch.load(f'{ckpt_base}/en_default_se.pth').to(device)


# Run the base speaker tts
# text = "Seems like the driver in the video car had plenty of time to react and slow down, they just chose not to and overreacted."
audio_info={}
conv_info={}
inf={}
df = pd.DataFrame()
df_cv=pd.DataFrame()
df_inf=pd.DataFrame()
emotion=['whispering','default','angry','cheerful','excited','friendly','sad','shouting','terrified']
txt_path='/home/ubuntu/OpenVoice/data/full_dataset/goemotions_1.csv'
txt_data=pd.read_csv(txt_path)
txt_data=pd.DataFrame(txt_data)
speaker_count=0
tts_count=0
convert_count=0
for i in tqdm(random.sample(range(len(txt_data['text'])),7000)):
    for j in range(len(emotion)):
        emo=emotion[j]
        src_path = f'{output_dir_tts}/{i}_{emo}_tts.wav'
        base_speaker_tts.tts(txt_data['text'][i], src_path, speaker=emo, language='English', speed=1.0)
        audio_info['text']=txt_data['text'][i]
        audio_info['text_index']=i
        audio_info['path']=src_path
        audio_info['emotion']=emo
        df = pd.concat([df, pd.DataFrame([audio_info])], ignore_index=True)
        tts_count+=1
        ##speaker ID converter
        for filename in os.listdir('/home/ubuntu/efs/sts/LibriTTS/train-clean-100'):
            file_wav = find_files(f'/home/ubuntu/efs/sts/LibriTTS/train-clean-100/{filename}')
            # print(file_wav)
            # input()
            speaker_count+=1
            convert_count+=1
            reference_speaker =file_wav # This is the voice you want to clone
            if reference_speaker == None:
                continue
            target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True)
            # print(filename)
            # input()
            save_path = f'{output_dir}/{filename}_{i}_{emo}.wav'
            encode_message = "@MyShell"
            tone_color_converter.convert(
            audio_src_path=src_path, 
            src_se=source_se_base, 
            tgt_se=target_se, 
            output_path=save_path,
            message=encode_message)
            conv_info['text']=txt_data['text'][i]
            conv_info['text_index']=i
            conv_info['path']=save_path
            conv_info['speaker']=filename
            conv_info['emotion']=emo
            df_cv = pd.concat([df_cv, pd.DataFrame([conv_info])], ignore_index=True)

inf['text_num']=i+1
inf['speaker_num']=speaker_count
inf['emotion_num']=len(emotion)
inf['tts_synthesized_audio_num']=tts_count
inf['convert_audio_num']=convert_count
df_inf = pd.concat([df_inf, pd.DataFrame([inf])], ignore_index=True)



df.to_csv('/home/ubuntu/efs/sts/train_TTS_audio.csv')
df_cv.to_csv('/home/ubuntu/efs/sts/train_convert_audio.csv')
df_inf.to_csv('/home/ubuntu/efs/sts/train_dataset_inf.csv')
df.to_csv('/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf/train_TTS_audio.csv')
df_cv.to_csv('/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf/train_convert_audio.csv')
df_inf.to_csv('/home/ubuntu/OpenVoice/emotion_STS/melo_rlhf/train_dataset_inf.csv')


"""
texts = {
    'EN_NEWEST': "Did you ever hear a folk tale about a giant turtle?",  # The newest English base speaker model
    'EN': "Did you ever hear a folk tale about a giant turtle?",
    'ES': "El resplandor del sol acaricia las olas, pintando el cielo con una paleta deslumbrante.",
    'FR': "La lueur dorée du soleil caresse les vagues, peignant le ciel d'une palette éblouissante.",
    'ZH': "在这次vacation中，我们计划去Paris欣赏埃菲尔铁塔和卢浮宫的美景。",
    'JP': "彼は毎朝ジョギングをして体を健康に保っています。",
    'KR': "안녕하세요! 오늘은 날씨가 정말 좋네요.",
}

src_path = f'{output_dir}/tmp.wav'

# Speed is adjustable
speed = 1.0

for language, text in texts.items():
    model = TTS(language=language, device=device)
    speaker_ids = model.hps.data.spk2id
    print('speaker_ids.keys()',speaker_ids.keys())
    print('speaker_ids',speaker_ids)
    print('model.hps.data',model.hps)
    input()
    
    for speaker_key in speaker_ids.keys():
        speaker_id = speaker_ids[speaker_key]
        speaker_key = speaker_key.lower().replace('_', '-')
        
        source_se = torch.load(f'checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)
        model.tts_to_file(text, speaker_id, src_path, speed=speed)
        base_speaker_tts.tts(text, src_path, speaker='default', language='English', speed=1.0)
        save_path = f'{output_dir}/output_v2_{speaker_key}.wav'

        # Run the tone color converter
        encode_message = "@MyShell"
        tone_color_converter.convert(
            audio_src_path=src_path, 
            src_se=source_se, 
            tgt_se=target_se, 
            output_path=save_path,
            message=encode_message)
        # tagger = Tagger()"""