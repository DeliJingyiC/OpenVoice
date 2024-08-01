import os
import numpy as np
import torch
from typing import Any, Dict, Tuple, Generator, List, Optional, Union
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import load_filepaths_and_text
import random
from tqdm import tqdm
from loguru import logger
import datetime
from torch.distributions.normal import Normal


def pad_lastdimension(x: List[np.ndarray], y: torch.Tensor):
    x = [k.squeeze() for k in x]
    assert np.all([k.shape[:-1] == x[0].shape[:-1]
                   for k in x]), f'{[k.shape for k in x]}'
    pad = np.zeros((
        len(x),
        *x[0].shape[:-1],
        max([k.shape[-1] for k in x]),
    ))
    for i, l in enumerate(x):
        pad[i, ..., :l.shape[-1]] = l
        pad[i, ..., l.shape[-1]:] = y[i]
    return pad


class RolloutBuffer(Dataset):

    def __init__(
        self,
        path: Path,
        buffer_size: int = 10,
    ):
        self.loaded = []
        self.buffer = []
        self.folder = path
        self.buffer_size = buffer_size

    def __len__(self):
        # length = sum([len(i["frame"]) for i in self.buffer])
        length = len(self.buffer)
        return length

    # def get_audio_text_speaker_pair(self, audiopath_sid_text):
    #     audiopath, input_audio,emo_label,emo_audio = audiopath_sid_text
    #     spec, wav = self.get_audio(audiopath)
    #     spec_input,wav_input=self.get_audio(input_audio)

    #     emo = torch.FloatTensor(np.load(emo_audio))

    #     # print('emo_getaudiotextspeaker',emo.shape)
    #     return (spec, wav, spec_input,wav_input,emo)

    
    def __getitem__(self, idex: int):
        ###test
        for i in range(len(self.buffer)):
            if idex < len(self.buffer[i]['frame']):
                if idex == len(self.buffer[i]['frame']) - 1:
                    idex = random.randint(0, len(self.buffer[i]['frame']) - 2)
                item_f = self.buffer[i]['frame'][idex]
                item_lob_prob = self.buffer[i]['log_prob'][idex]
                item_lob_probinf = self.buffer[i]['log_probinf'][idex]



                item_model_mean = self.buffer[i]['model_mean'][idex]
                item_model_std = self.buffer[i]['model_std'][idex]
                item_model_stdinf = self.buffer[i]['model_stdinf'][idex]


                step_list=[x+1 for x in range(len(self.buffer[i]['frame']))][-5:]
                return [
                    item_f,
                    item_lob_prob,
                    idex + 1,
                    # len(self.buffer[i]['frame']) - 1,
                    self.buffer[i]['hidrep'],
                    self.buffer[i]['mosscore'],
                    self.buffer[i]["text"],
                    self.buffer[i]['mosscore_update'],
                    self.buffer[i]['duration_target'],
                    self.buffer[i]['speakers'],
                    self.buffer[i]['input_lengths'],
                    self.buffer[i]['output_lengths'],
                    self.buffer[i]['text_org'],
                    item_model_mean,
                    item_model_std,
                    self.buffer[i]['rawtext'],
                    self.buffer[i]['goal_audio'],
                    # item_lob_prob_org,
                    # item_model_mean_org,
                    # item_model_std_org,
                    step_list,
                    item_lob_probinf,
                    item_model_stdinf,
                ]

            else:
                idex = idex - len(self.buffer[i]['frame'])


    def fetch_name(self, **data):
        eps_idx = len(list(self.folder.glob('*.dt')))
        random_idx = np.random.randint(0, 1000)
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        # text = data['text']
        eps_fn = f"{ts}_{eps_idx:09d}_{random_idx:04d}"
        full_path = self.folder / eps_fn

        return full_path

    def save(self, **data):
        file_path = self.fetch_name(**data)
        print('data',data)
        np.savez(
            file_path,
            # hidrep=data['hidrep'],
            # mosscore=data['mosscore'],
            # frame=data['frame'],
            # log_prob=data['log_prob'],
            # text=data['text'],
            **data,
        )
        file_path.with_suffix(".npz").rename(file_path.with_suffix(".dt"))
        print(f"Saved {file_path.with_suffix('.dt')}")
        return file_path

    def load_data(self):
        # print('load_data134')
        for x in sorted(self.folder.glob('*.dt')):
            if x in self.loaded:
                continue
            # print('load_data138')

            kk = np.load(x)
            # print('load_data141')

            self.buffer.append({**kk})
            # print('load_data144')

            self.loaded.append(x)
            # print('self.buffer',self.buffer)
            # print(f"Loaded {x} self.buffer {len(self.buffer)}")


        while (len(self.buffer) > self.buffer_size):
            # print('before', [x['mosscore_update'] for x in self.buffer])
            self.buffer.pop(0)
            # print('after', [x['mosscore_update'] for x in self.buffer])
        return self.buffer

    def add(
        self,
        hidrep,
        mosscore,
        frame,
        log_prob,
        text,
        duration_target,
        speakers,
        input_lengths,
        output_lengths,
        text_org,
        model_mean,
        model_std,
        mosscore_update,
        rawtext,
        goal_audio,
        step_list,
        log_probinf,
        item_model_stdinf,

    ):
        assert len(frame) == len(
            log_prob), f'frame {len(frame)}, log_prob {len(log_prob)}'
        print('mosscore_update buffer', mosscore_update)
        print(
            f"Adding {mosscore_update} to {[x['mosscore_update'] for x in self.buffer]}"
        )
        self.buffer.append({
            'hidrep': hidrep,
            'mosscore': mosscore,
            'frame': frame,
            'log_prob': log_prob,
            'text': text,
            "duration_target": duration_target,
            "speakers": speakers,
            "input_lengths": input_lengths,
            "output_lengths": output_lengths,
            "text_org": text_org,
            "model_mean": model_mean,
            "model_std": model_std,
            'mosscore_update': mosscore_update,
            'rawtext': rawtext,
            'goal_audio': goal_audio,
            'step_list':step_list,
            'log_probinf': log_probinf,
            'item_model_stdinf':item_model_stdinf,


        })
        print()

class BufferCollate_RL:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # print('batch',batch)
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]), dim=0, descending=True
        )
        # print('ids_sorted_decreasing',ids_sorted_decreasing)
        # max_spec_len = max([x[0].size(1) for x in batch])
        # max_wav_len = max([x[1].size(1) for x in batch])
        # max_spec_input_len = max([len(x[2]) for x in batch])
        # max_wav_input_len = max([x[3].size(1) for x in batch])
        # spec_lengths = torch.LongTensor(len(batch))
        # wav_lengths = torch.LongTensor(len(batch))
        # spec_input_lengths = torch.LongTensor(len(batch))
        # wav_input_lengths = torch.LongTensor(len(batch))
        # spec_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max_spec_len)
        # wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        # spec_input_padded = torch.FloatTensor(len(batch), batch[0][2].size(0), max_spec_input_len)
        # wav_input_padded = torch.FloatTensor(len(batch), 1, max_wav_input_len)
        # emo = torch.FloatTensor(len(batch),1024)

        # spec_padded.zero_()
        # wav_padded.zero_()
        # spec_input_padded.zero_()
        # wav_input_padded.zero_()
        # emo.zero_()
        emo_audio_embed=[]
        emo_audio_path_list=[]
        goal_audio_list=[]
        input_audio_list=[]

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            # spec = row[0]

            # spec_padded[i, :, : spec.size(1)] = spec
            # spec_lengths[i] = spec.size(1)

            # wav = row[1]
            # wav_padded[i, :, : wav.size(1)] = wav
            # wav_lengths[i] = wav.size(1)

            
            # spec_input = row[2]
            # if spec_input.size(1) >1025:
            #     spec_input=spec_input[:,:1025]

            # spec_input_padded[i, :, : spec_input.size(1)] = spec_input
            # spec_input_lengths[i] = spec_input.size(1)
            
            # wav = row[3]
            # wav_input_padded[i, :, : wav.size(1)] = wav
            # wav_input_lengths[i] = wav.size(1)

            # emo[i, :row[4].size(0)] = row[4]
            # emo.squeeze(dim=-1)
            emo_audio_embed.append(row[1])
            emo_audio_path_list.append(row[2])

            goal_audio_list.append(row[3])
            input_audio_list.append(row[4])




        return (
            emo_audio_embed,
            emo_audio_path_list,
            goal_audio_list,
            input_audio_list

        )

    def create_dataloader(self, batch_size, num_worker):

        self.load_data()

        def collate_fn(batch: List[Tuple[np.ndarray, float, int, np.ndarray,
                                         float]]):
            rawtext = [x[14] for x in batch]
            # print('batchsize', len(batch))

            model_std = torch.tensor(np.array([x[13] for x in batch]))
            model_std_inf = torch.tensor(np.array([x[18] for x in batch]))


            head = batch[0]

            logprob = torch.tensor(np.array([x[1] for x in batch]))



            logprob_inf = torch.tensor(np.array([x[17] for x in batch]))



            tidx = torch.tensor([x[2] for x in batch])
            step_list = torch.tensor(np.array([x[16] for x in batch]))


            model_mean_lis = [x[12] for x in batch]

            model_mean = torch.tensor(
                pad_lastdimension(
                    model_mean_lis,
                    y=torch.zeros((len(model_mean_lis), )),
                ),
                dtype=torch.float32,
            )


            goal_audio_list = [x[15] for x in batch]
            goal_audio = pad_lastdimension(
                goal_audio_list,
                y=torch.zeros((len(goal_audio_list), )),
            )
            goal_audio = torch.tensor(goal_audio, dtype=torch.float32)

            ##pad frame
            frame_list = [x[0] for x in batch]
            # frame = np.zeros(
            #     (len(batch), max([x.shape[-1] for x in frame_list])))
            # for i, f in enumerate(frame_list):
            #     frame[i, :f.shape[-1]] = f
            paded_frame = pad_lastdimension(
                frame_list,
                y=torch.zeros((len(frame_list), )),
            )
            # assert np.all(
            #     frame.shape == paded_frame.shape
            # ), f'frame.shape {frame.shape}, paded_frame {paded_frame.shape}'

            frame = torch.tensor(paded_frame, dtype=torch.float32)
            # for x in batch:
            #     print("text shape", x[5].shape)
            # exit(0)
            ###pad text
            text_list = [x[5] for x in batch]
            # text = np.zeros(
            #     (len(batch), max([x.shape[-1] for x in text_list])), )
            # for i, t in enumerate(text_list):
            #     text[i, :t.shape[-1]] = t[0]
            paded_text = pad_lastdimension(
                text_list,
                y=torch.zeros((len(text_list), )),
            )
            # assert np.all(
            #     text.shape == paded_text.shape
            # ), f'text.shape {text.shape}, paded_text {paded_text.shape}'

            texts = torch.tensor(paded_text, dtype=torch.long)

            ###pad hq_list
            hp_list = [x[3] for x in batch]
            # hp = np.zeros((
            #     len(hp_list),
            #     hp_list[0].shape[1],
            #     max([x.shape[-1] for x in hp_list]),
            # ))
            # for i, l in enumerate(hp_list):
            #     hp[i, :, :l.shape[-1]] = l[0]
            paded_hp = pad_lastdimension(
                hp_list,
                y=torch.zeros((len(hp_list), )),
            )
            # assert np.all(hp.shape == paded_hp.shape
            #               ), f'hp.shape {hp.shape}, paded_hp {paded_hp.shape}'
            hidrep = torch.tensor(paded_hp)

            ##mosscore
            mos = torch.tensor(np.array([x[4] for x in batch]))


            mosscore_update = torch.tensor(np.array([x[6] for x in batch
                                                     ])).squeeze()


            duration_target = torch.tensor(
                pad_lastdimension(
                    [x[7] for x in batch],
                    y=torch.zeros((len(batch), )),
                ), ).float()
            speakers = torch.tensor(np.array([x[8] for x in batch])).squeeze()
            input_lengths = torch.tensor(np.array([x[9] for x in batch]),
                                         dtype=torch.int).squeeze()
            output_lengths = torch.tensor(np.array([x[10] for x in batch
                                                    ])).squeeze()
            text_org = torch.tensor(
                pad_lastdimension(
                    [x[11] for x in batch],
                    y=torch.zeros((len(batch), )),
                ),
                dtype=torch.long,
            )


            kk = [
                frame,
                logprob,
                tidx,
                hidrep,
                mos,
                texts,
                mosscore_update,
                duration_target,
                speakers,
                input_lengths,
                output_lengths,
                text_org,
                model_mean,
                model_std,
                goal_audio,
                step_list,
                logprob_inf,
            ]
            return kk

        # self.buffer=random.sample(self.buffer,k=batch_size)

        random_index = np.random.choice(np.arange(len(self)),
                                        size=batch_size * 8 * 5)
        # print('random_index', random_index)
        samp = [self[i] for i in random_index]
        # print('samprandom1', self[0])
        # print('samprandom1', self[0][6])
        # print('samprandom2', self[2][6])
        # print('samprandom2', self[22][6])
        # print('samprandom2_logprob', self[0][1], self[2][1], self[22][1])
        # print('samprandom2_model_mean', self[0][13], self[2][13], self[22][13])
        # print('samprandom2_model_std', self[0][14], self[2][14], self[22][14])

        return DataLoader(
            # [self[i] for i in random_index],
            samp,
            # self,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_worker,
            drop_last=True,
            collate_fn=collate_fn)
