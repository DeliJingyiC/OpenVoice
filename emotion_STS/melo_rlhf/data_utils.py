import os
import random
import torch
import torch.utils.data
from tqdm import tqdm
from loguru import logger
import commons
from mel_processing import spectrogram_torch, mel_spectrogram_torch
from utils import load_filepaths_and_text
from utils import load_wav_to_torch_librosa as load_wav_to_torch
import numpy as np
import librosa
from emotion_reward_model import process_func
from speaker_reward_model import speaker_func
import torchaudio



"""Multi speaker version"""

class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_sid_text, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.spk_map = hparams.spk2id
        self.hparams = hparams
        self.disable_bert = getattr(hparams, "disable_bert", False)

        self.use_mel_spec_posterior = getattr(
            hparams, "use_mel_posterior_encoder", False
        )
        if self.use_mel_spec_posterior:
            self.n_mel_channels = getattr(hparams, "n_mel_channels", 80)

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        # self.min_text_len = getattr(hparams, "min_text_len", 1)
        # self.max_text_len = getattr(hparams, "max_text_len", 300)

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()


    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        lengths = []
        skipped = 0
        logger.info("Init dataset...")
        for item in tqdm(
            self.audiopaths_sid_text
        ):
            try:
                goal_syn_audio, spk, input_audio,emo_label, emo = item
            except:
                raise
            audiopath = f"{goal_syn_audio}"
            audiopath_input = f"{input_audio}"
            emopath=f"{emo}"
            # realaudio_input=f'{real_input}'
            # realaudio_ref=f'{real_ref}'

            audiopaths_sid_text_new.append(
                    [audiopath, audiopath_input,emopath]
                )
            lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
            
        logger.info(f'min: {min(lengths)}; max: {max(lengths)}' )
        logger.info(
            "skipped: "
            + str(skipped)
            + ", total: "
            + str(len(self.audiopaths_sid_text))
        )
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text

        audiopath, input_audio,emo_audio,real_input,real_ref = audiopath_sid_text

        # bert, ja_bert, phones, tone, language = self.get_text(
        #     text, word2ph, phones, tone, language, audiopath
        # )

        spec, wav = self.get_audio(audiopath)


        spec_input,wav_input=self.get_audio(input_audio)
        spec_input_real,wav_input_real=self.get_audio(real_input)


        # sid = int(getattr(self.spk_map, sid, '0'))

        # sid = torch.LongTensor([sid])

        emo = torch.FloatTensor(np.load(emo_audio))
        emo_real = torch.FloatTensor(np.load(real_ref))
        ##goal_audio_spec,goal_audio_wav,
        return (spec, wav, spec_input,wav_input,emo,spec_input_real,wav_input_real,emo_real)



    def get_audio(self, filename):
        audio_norm, sampling_rate = load_wav_to_torch(filename, self.sampling_rate)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                "{} {} SR doesn't match target {} SR".format(
                    filename, sampling_rate, self.sampling_rate
                )
            )
        # NOTE: normalize has been achieved by torchaudio
        # audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if self.use_mel_spec_posterior:
            spec_filename = spec_filename.replace(".spec.pt", ".mel.pt")
        try:
            spec = torch.load(spec_filename)
            assert False
        except:
            if self.use_mel_spec_posterior:

                spec = mel_spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.n_mel_channels,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    self.hparams.mel_fmin,
                    self.hparams.mel_fmax,
                    center=False,
                )
            else:

                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )

            spec = torch.squeeze(spec, 0)

            torch.save(spec, spec_filename)
        return spec, audio_norm

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):

        return len(self.audiopaths_sid_text)

class TextAudioSpeakerCollate:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
    
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True
        )

        max_spec_len = max([x[0].size(1) for x in batch])
        max_wav_len = max([x[1].size(1) for x in batch])
        max_spec_input_len = max([len(x[2]) for x in batch])
        max_wav_input_len = max([x[3].size(1) for x in batch])
        max_spec_realinput_len = max([len(x[5]) for x in batch])
        max_wav_realinput_len = max([x[6].size(1) for x in batch])

        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        spec_input_lengths = torch.LongTensor(len(batch))
        wav_input_lengths = torch.LongTensor(len(batch))
        spec_realinput_lengths = torch.LongTensor(len(batch))
        wav_realinput_lengths = torch.LongTensor(len(batch))

        spec_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        spec_input_padded = torch.FloatTensor(len(batch), batch[0][2].size(0), max_spec_input_len)
        wav_input_padded = torch.FloatTensor(len(batch), 1, max_wav_input_len)
        spec_realinput_padded = torch.FloatTensor(len(batch), batch[0][4].size(0), max_spec_realinput_len)
        wav_realinput_padded = torch.FloatTensor(len(batch), 1, max_wav_realinput_len)

        emo = torch.FloatTensor(len(batch),1024)
        realemo = torch.FloatTensor(len(batch),1024)


        spec_padded.zero_()
        wav_padded.zero_()
        spec_input_padded.zero_()
        wav_input_padded.zero_()
        emo.zero_()
        spec_realinput_padded.zero_()
        wav_realinput_padded.zero_()
        realemo.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            spec = row[0]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[1]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)
            
            spec_input = row[2]
            if spec_input.size(1) >1025:
                spec_input=spec_input[:,:1025]

            spec_input_padded[i, :, : spec_input.size(1)] = spec_input
            spec_input_lengths[i] = spec_input.size(1)
            
            wav_input = row[3]
            wav_input_padded[i, :, : wav_input.size(1)] = wav_input
            wav_input_lengths[i] = wav_input.size(1)

            spec_real = row[5]
            spec_realinput_padded[i, :, : spec_real.size(1)] = spec_real
            spec_realinput_lengths[i] = spec_real.size(1)

            wav_real = row[6]
            wav_realinput_padded[i, :, : wav_real.size(1)] = wav_real
            wav_realinput_lengths[i] = wav_real.size(1)

            # input()

            emo[i, :row[4].size(0)] = row[4]
            emo.squeeze(dim=-1)
            realemo[i, :row[7].size(0)] = row[7]
            realemo.squeeze(dim=-1)


            # emo[i]=row[5]



        return (

            spec_padded,
            spec_lengths,
            wav_padded,
            wav_lengths,
            # sid,
            spec_input_padded,
            spec_input_lengths,
            wav_input_padded,
            wav_input_lengths,
            emo,
            spec_realinput_padded,
            spec_realinput_lengths,
            wav_realinput_padded,
            wav_realinput_lengths,
            realemo,
            
            

        )

class TextAudioSpeakerLoader_RL(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_sid_text, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.spk_map = hparams.spk2id
        self.hparams = hparams
        self.disable_bert = getattr(hparams, "disable_bert", False)

        self.use_mel_spec_posterior = getattr(
            hparams, "use_mel_posterior_encoder", False
        )
        if self.use_mel_spec_posterior:
            self.n_mel_channels = getattr(hparams, "n_mel_channels", 80)

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        # self.min_text_len = getattr(hparams, "min_text_len", 1)
        # self.max_text_len = getattr(hparams, "max_text_len", 300)

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)

        # if len(self.audiopaths_sid_text)>10:
        #     self.audiopaths_sid_text=random.sample(self.audiopaths_sid_text,10)
        # else:
        #     random.shuffle(self.audiopaths_sid_text)



        self._filter()


    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        lengths = []
        skipped = 0
        logger.info("Init dataset...")
        for item in tqdm(
            self.audiopaths_sid_text
        ):
            try:
                goal_syn_audio, spk, input_syn_audio,lang, emo_syn,real_input_audio,emo_real = item

            except:
                print(item)
                raise
            audiopath = f"{goal_syn_audio}"
            audiopath_input = f"{input_syn_audio}"
            emopath=f"{emo_syn}"
            audiopath_input_real = f"{real_input_audio}"
            emopath_real=f"{emo_real}"
            # realaudio_input=f'{real_input}'
            # realaudio_ref=f'{real_ref}'
            audiopaths_sid_text_new.append(
                    [audiopath, audiopath_input,emopath,audiopath_input_real,emopath_real]
                )
            lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
            
        logger.info(f'min: {min(lengths)}; max: {max(lengths)}' )
        logger.info(
            "skipped: "
            + str(skipped)
            + ", total: "
            + str(len(self.audiopaths_sid_text))
        )
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
 
        audiopath_syngoal, input_audio_syn,emo_audio_syn,real_input,real_ref = audiopath_sid_text
        #goal syn audio
        spec_syn_goal, wav_syn_goal = self.get_audio(audiopath_syngoal)
        spec_input_syn,wav_input_syn=self.get_audio(input_audio_syn)
        spec_input_real,wav_input_real=self.get_audio(real_input)
        emo = torch.FloatTensor(np.load(emo_audio_syn))
        emo_real = torch.FloatTensor(np.load(real_ref))
        #syn ref/emo audio path
        emo_audio_path=emo_audio_syn.split('.')[0]+'.wav'
        #real ref/emo audio path
        realemo_audio_path=real_ref.split('.')[0]+'.wav'
        spec_emo_real,wav_emo_real=self.get_audio(realemo_audio_path)

        # emo_audio_syn = torch.FloatTensor(np.load(emo_audio_syn))
        # emo_audio_syn = torch.stack(emo_audio_syn)
        # realemo_audio_path = librosa.load(realemo_audio_path, 16000)[0] 
        # wav_emb_real = process_func(np.expand_dims(realemo_audio_path, 0), 16000, embeddings=True)[0] 
        # wav_emb_real = torch.tensor(wav_emb_real)
        # real_input=torchaudio.load(real_input, normalize=False)[0] 
        # input_audio_syn, sample_rate = librosa.load(input_audio_syn, self.sampling_rate)
        # realemo_audio_path=torch.stack(realemo_audio_path)
        return (spec_syn_goal,emo_audio_syn,emo_audio_path,audiopath_syngoal,input_audio_syn,real_ref,realemo_audio_path, real_input, wav_syn_goal, spec_input_syn,wav_input_syn, spec_input_real,wav_input_real,emo,emo_real,spec_emo_real,wav_emo_real)

    def get_audio(self, filename):
        audio_norm, sampling_rate = load_wav_to_torch(filename, self.sampling_rate)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                "{} {} SR doesn't match target {} SR".format(
                    filename, sampling_rate, self.sampling_rate
                )
            )
        # NOTE: normalize has been achieved by torchaudio
        # audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if self.use_mel_spec_posterior:
            spec_filename = spec_filename.replace(".spec.pt", ".mel.pt")
        try:
            spec = torch.load(spec_filename)
            assert False
        except:
            if self.use_mel_spec_posterior:
                spec = mel_spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.n_mel_channels,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    self.hparams.mel_fmin,
                    self.hparams.mel_fmax,
                    center=False,
                )
            else:
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate_RL:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]), dim=0, descending=True
        )
        emo_audio_embed_syn=[]
        emo_audio_path_list_syn=[]
        goal_audio_list_syn=[]
        input_audio_list_syn=[]
        emo_audio_embed_real=[]
        emo_audio_path_list_real=[]
        input_audio_list_real=[]

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            emo_audio_embed_syn.append(row[1])
            emo_audio_path_list_syn.append(row[2])

            goal_audio_list_syn.append(row[3])
            input_audio_list_syn.append(row[4])
            emo_audio_embed_real.append(row[5])
            emo_audio_path_list_real.append(row[6])
            input_audio_list_real.append(row[7])
            

        max_spec_len = max([x[0].size(1) for x in batch])
        max_wav_len = max([x[8].size(1) for x in batch])
        max_spec_input_len = max([len(x[9]) for x in batch])
        max_wav_input_len = max([x[10].size(1) for x in batch])
        max_spec_realinput_len = max([len(x[11]) for x in batch])
        max_wav_realinput_len = max([x[12].size(1) for x in batch])
        max_spec_emo_real_len = max([len(x[15]) for x in batch])
        max_wav_emo_real_len = max([x[16].size(1) for x in batch])
        

        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        spec_input_lengths = torch.LongTensor(len(batch))
        wav_input_lengths = torch.LongTensor(len(batch))
        spec_realinput_lengths = torch.LongTensor(len(batch))
        wav_realinput_lengths = torch.LongTensor(len(batch))
        spec_emo_real_lengths = torch.LongTensor(len(batch))
        wav_emo_real_lengths = torch.LongTensor(len(batch))
        

        spec_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        spec_input_padded = torch.FloatTensor(len(batch), batch[0][9].size(0), max_spec_input_len)
        wav_input_padded = torch.FloatTensor(len(batch), 1, max_wav_input_len)
        spec_realinput_padded = torch.FloatTensor(len(batch), batch[0][11].size(0), max_spec_realinput_len)
        wav_realinput_padded = torch.FloatTensor(len(batch), 1, max_wav_realinput_len)
        spec_emo_real_padded = torch.FloatTensor(len(batch), batch[0][15].size(0), max_spec_emo_real_len)
        wav_emo_real_padded = torch.FloatTensor(len(batch), 1, max_wav_emo_real_len)


        emo = torch.FloatTensor(len(batch),1024)
        realemo = torch.FloatTensor(len(batch),1024)



        spec_padded.zero_()
        wav_padded.zero_()
        spec_input_padded.zero_()
        wav_input_padded.zero_()
        emo.zero_()
        spec_realinput_padded.zero_()
        wav_realinput_padded.zero_()
        realemo.zero_()
        spec_emo_real_padded.zero_()
        wav_emo_real_padded.zero_()


        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            spec = row[0]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[8]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)
            
            spec_input = row[9]
            if spec_input.size(1) >1025:
                spec_input=spec_input[:,:1025]

            spec_input_padded[i, :, : spec_input.size(1)] = spec_input
            spec_input_lengths[i] = spec_input.size(1)
            
            wav_input = row[10]
            wav_input_padded[i, :, : wav_input.size(1)] = wav_input
            wav_input_lengths[i] = wav_input.size(1)

            spec_real = row[11]
            spec_realinput_padded[i, :, : spec_real.size(1)] = spec_real
            spec_realinput_lengths[i] = spec_real.size(1)

            wav_real = row[12]
            wav_realinput_padded[i, :, : wav_real.size(1)] = wav_real
            wav_realinput_lengths[i] = wav_real.size(1)

            spec_emo_real = row[15]
            spec_emo_real_padded[i, :, : spec_emo_real.size(1)] = spec_emo_real
            spec_emo_real_lengths[i] = spec_emo_real.size(1)

            wav_emo_real = row[16]
            wav_emo_real_padded[i, :, : wav_emo_real.size(1)] = wav_emo_real
            wav_emo_real_lengths[i] = wav_emo_real.size(1)
            # input()

            emo[i, :row[13].size(0)] = row[13]
            emo.squeeze(dim=-1)
            realemo[i, :row[14].size(0)] = row[14]
            realemo.squeeze(dim=-1)

        return (
            emo_audio_embed_syn,
            emo_audio_path_list_syn,
            goal_audio_list_syn,
            input_audio_list_syn,
            emo_audio_embed_real,
            emo_audio_path_list_real,
            input_audio_list_real,
            spec_padded,
            spec_lengths,
            wav_padded,
            wav_lengths,
            spec_input_padded,
            spec_input_lengths,
            wav_input_padded,
            wav_input_lengths,
            spec_realinput_padded,
            spec_realinput_lengths,
            wav_realinput_padded,
            wav_realinput_lengths,
            emo,
            realemo,
            spec_emo_real_padded,
            spec_emo_real_lengths,
            wav_emo_real_padded,
            wav_emo_real_lengths,
        )



# class TextAudioSpeakerCollate_old:
#     """Zero-pads model inputs and targets"""

#     def __init__(self, return_ids=False):
#         self.return_ids = return_ids

#     def __call__(self, batch):
#         """Collate's training batch from normalized text, audio and speaker identities
#         PARAMS
#         ------
#         batch: [text_normalized, spec_normalized, wav_normalized, sid]
#         """
#         # Right zero-pad all one-hot text sequences to max input length
#         _, ids_sorted_decreasing = torch.sort(
#             torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True
#         )

#         max_text_len = max([len(x[0]) for x in batch])
#         max_spec_len = max([x[1].size(1) for x in batch])
#         max_wav_len = max([x[2].size(1) for x in batch])

#         text_lengths = torch.LongTensor(len(batch))
#         spec_lengths = torch.LongTensor(len(batch))
#         wav_lengths = torch.LongTensor(len(batch))
#         sid = torch.LongTensor(len(batch))

#         text_padded = torch.LongTensor(len(batch), max_text_len)
#         tone_padded = torch.LongTensor(len(batch), max_text_len)
#         language_padded = torch.LongTensor(len(batch), max_text_len)
#         bert_padded = torch.FloatTensor(len(batch), 1024, max_text_len)
#         ja_bert_padded = torch.FloatTensor(len(batch), 768, max_text_len)

#         spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
#         wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
#         text_padded.zero_()
#         tone_padded.zero_()
#         language_padded.zero_()
#         spec_padded.zero_()
#         wav_padded.zero_()
#         bert_padded.zero_()
#         ja_bert_padded.zero_()
#         for i in range(len(ids_sorted_decreasing)):
#             row = batch[ids_sorted_decreasing[i]]

#             text = row[0]
#             text_padded[i, : text.size(0)] = text
#             text_lengths[i] = text.size(0)

#             spec = row[1]
#             spec_padded[i, :, : spec.size(1)] = spec
#             spec_lengths[i] = spec.size(1)

#             wav = row[2]
#             wav_padded[i, :, : wav.size(1)] = wav
#             wav_lengths[i] = wav.size(1)

#             sid[i] = row[3]

#             tone = row[4]
#             tone_padded[i, : tone.size(0)] = tone

#             language = row[5]
#             language_padded[i, : language.size(0)] = language

#             bert = row[6]
#             bert_padded[i, :, : bert.size(1)] = bert

#             ja_bert = row[7]
#             ja_bert_padded[i, :, : ja_bert.size(1)] = ja_bert

#         return (
#             text_padded,
#             text_lengths,
#             spec_padded,
#             spec_lengths,
#             wav_padded,
#             wav_lengths,
#             sid,
#             tone_padded,
#             language_padded,
#             bert_padded,
#             ja_bert_padded,
#         )


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
        print('buckets:', self.num_samples_per_bucket)

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        try:
            for i in range(len(buckets) - 1, 0, -1):
                if len(buckets[i]) == 0:
                    buckets.pop(i)
                    self.boundaries.pop(i + 1)
            assert all(len(bucket) > 0 for bucket in buckets)
        # When one bucket is not traversed
        except Exception as e:
            print("Bucket warning ", e)
            for i in range(len(buckets) - 1, -1, -1):
                if len(buckets[i]) == 0:
                    buckets.pop(i)
                    self.boundaries.pop(i + 1)

        num_samples_per_bucket = []

        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            if len_bucket == 0:
                continue
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
