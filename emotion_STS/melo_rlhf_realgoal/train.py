#!/home/ubuntu/OpenVoice/.openvoiceenv/bin/python
# flake8: noqa: E402

"""

"""


import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging
from api import STS
import librosa
from utils import load_wav_to_torch_librosa as load_wav_to_torch
from mel_processing import spectrogram_torch, mel_spectrogram_torch
import random
import soundfile

logging.getLogger("numba").setLevel(logging.WARNING)
import commons
import utils
from data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
    TextAudioSpeakerLoader_RL,
    TextAudioSpeakerCollate_RL,
    DistributedBucketSampler,
    # bufferLoader
)
from buffer_jc import RolloutBuffer
from buffer_melo import ReplayBuffer
from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
    DurationDiscriminator,
)
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from mel_processing import mel_spectrogram_torch_RL, spec_to_mel_torch,mel_spectrogram_torch
# from MeloTTS.melo.download_utils import load_pretrain_model
from download_utils import load_pretrain_model
from emotion_reward_model import process_func
from speaker_reward_model import speaker_func
import torchaudio
from datamodule import DataModule
from pathlib import Path
from torch.distributions import Normal

from numpy import dot
from numpy.linalg import norm
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = (
    True  # If encontered training problem,please try to disable TF32.
)
torch.set_float32_matmul_precision("medium")


torch.backends.cudnn.benchmark = True
torch.backends.cuda.sdp_kernel("flash")
torch.backends.cuda.enable_flash_sdp(True)
# torch.backends.cuda.enable_mem_efficient_sdp(
#     True
# )  # Not available if torch version is lower than 2.0
torch.backends.cuda.enable_math_sdp(True)
global_step = 0


def run():
    hps = utils.get_hparams()
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(
        backend="gloo",
        # backend="nccl",
        init_method="env://",  # Due to some training problem,we proposed to use gloo instead of nccl.
        rank=local_rank,
    )  # Use torchrun instead of mp.spawn
    rank = dist.get_rank()
    n_gpus = dist.get_world_size()

    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)
    global global_step

    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    # train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    # train_sampler = DistributedBucketSampler(
    #     train_dataset,
    #     hps.train.batch_size,
    #     [32, 300, 400, 500, 600, 700, 800, 900, 1000],
    #     num_replicas=n_gpus,
    #     rank=rank,
    #     shuffle=True,
    # )

    train_dataset_rl = TextAudioSpeakerLoader_RL(hps.data.training_files, hps.data)
    train_sampler_rl = DistributedBucketSampler(
        train_dataset_rl,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )

    # collate_fn = TextAudioSpeakerCollate()
    collate_fn_rl = TextAudioSpeakerCollate_RL()

    # train_loader = DataLoader(
    #     train_dataset,
    #     num_workers=8,
    #     shuffle=False,
    #     pin_memory=True,
    #     collate_fn=collate_fn,
    #     batch_sampler=train_sampler,
    #     persistent_workers=True,
    #     prefetch_factor=4,
    # )  # DataLoader config could be adjusted.
    
    train_loader_rl = DataLoader(
            train_dataset_rl,
            num_workers=8,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn_rl,
            batch_sampler=train_sampler_rl,
            persistent_workers=True,
            prefetch_factor=4,
        )  # DataLoader config could be adjusted.
    
    # if rank == 0:
        # eval_dataset = TextAudioSpeakerLoader_RL(hps.data.validation_files, hps.data)

        # eval_loader = DataLoader(
        #     eval_dataset,
        #     num_workers=8,
        #     shuffle=False,
        #     batch_size=1,
        #     pin_memory=True,
        #     drop_last=False,
        #     collate_fn=collate_fn_rl,
        # )
        # for index,i in enumerate(eval_loader):


    if (
        "use_noise_scaled_mas" in hps.model.keys()
        and hps.model.use_noise_scaled_mas is True
    ):
        print("Using noise scaled MAS for VITS2")
        mas_noise_scale_initial = 0.01
        noise_scale_delta = 2e-6
    else:
        print("Using normal MAS for VITS1")
        mas_noise_scale_initial = 0.0
        noise_scale_delta = 0.0
    if (
        "use_duration_discriminator" in hps.model.keys()
        and hps.model.use_duration_discriminator is True
    ):
        print("Using duration discriminator for VITS2")
        net_dur_disc = DurationDiscriminator(
            hps.model.hidden_channels,
            hps.model.hidden_channels,
            3,
            0.1,
            gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
        ).cuda(rank)
    if (
        "use_spk_conditioned_encoder" in hps.model.keys()
        and hps.model.use_spk_conditioned_encoder is True
    ):
        if hps.data.n_speakers == 0:
            raise ValueError(
                "n_speakers must be > 0 when using spk conditioned encoder to train multi-speaker model"
            )
    else:
        print("Using normal encoder for VITS1")
    
    replay_buffer=ReplayBuffer(capacity=8)

    net_g = SynthesizerTrn(
        # len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        # n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        # p_dropout=hps.model.p_dropout,
        **hps.model,
    ).cuda(rank)

    net_g_pre = SynthesizerTrn(
        # len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        # n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        # p_dropout=hps.model.p_dropout,
        **hps.model,
    ).cuda(rank)

    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)

    optim_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_g.parameters()),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    if net_dur_disc is not None:
        optim_dur_disc = torch.optim.AdamW(
            net_dur_disc.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
    else:
        optim_dur_disc = None
    net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    net_g_pre = DDP(net_g_pre, device_ids=[rank], find_unused_parameters=True)

    
    hps.pretrain_G = hps.pretrain_G 
    cwd = Path.cwd() 


    if hps.pretrain_G:
        print('load pretrain G')
        utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"),
                net_g,
                None,
                skip_optimizer=True
            )

    if hps.pretrain_D:
        print('load pretrain D')
        utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"),
                net_d,
                None,
                skip_optimizer=True
            )

    if hps.pretrain_rl:
            print('load pretrain G')
            utils.load_checkpoint(
                    utils.RL_pretrain_checkpoint_path(hps.pretrain_rl, "G_*.pth"),
                    net_g_pre,
                    None,
                    skip_optimizer=True
                )
    if net_dur_disc is not None:
        net_dur_disc = DDP(net_dur_disc, device_ids=[rank], find_unused_parameters=True)
        if hps.pretrain_dur:
            utils.load_checkpoint(
                    utils.latest_checkpoint_path(hps.model_dir, "DUR_*.pth"),
                    net_dur_disc,
                    None,
                    skip_optimizer=True
                )
                
    try:
        if net_dur_disc is not None:
            _, _, dur_resume_lr, epoch_str = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "DUR_*.pth"),
                net_dur_disc,
                optim_dur_disc,
                skip_optimizer=hps.train.skip_optimizer
                if "skip_optimizer" in hps.train
                else True,
            )
            _, optim_g, g_resume_lr, epoch_str = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"),
                net_g,
                optim_g,
                skip_optimizer=hps.train.skip_optimizer
                if "skip_optimizer" in hps.train
                else True,
            )
            _, optim_d, d_resume_lr, epoch_str = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"),
                net_d,
                optim_d,
                skip_optimizer=hps.train.skip_optimizer
                if "skip_optimizer" in hps.train
                else True,
            )
            if not optim_g.param_groups[0].get("initial_lr"):
                optim_g.param_groups[0]["initial_lr"] = g_resume_lr
            if not optim_d.param_groups[0].get("initial_lr"):
                optim_d.param_groups[0]["initial_lr"] = d_resume_lr
            if not optim_dur_disc.param_groups[0].get("initial_lr"):
                optim_dur_disc.param_groups[0]["initial_lr"] = dur_resume_lr
        
        global_step_pre=utils.latest_checkpoint_path(hps.model_dir, "D_*.pth")
        global_step_pre=f'{global_step_pre}'
        global_step_pre=global_step_pre.split('_')[-1]
        global_step_pre=global_step_pre.split('.')[0]
        if hps.pretrain_G:
            global_step = int(global_step_pre)

        else:
            global_step = (epoch_str - 1) * len(train_loader_rl)
        epoch_str = max(epoch_str, 1)
        # global_step = (epoch_str - 1) * len(train_loader)
    except Exception as e:
        print(e)
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    if net_dur_disc is not None:
        scheduler_dur_disc = torch.optim.lr_scheduler.ExponentialLR(
            optim_dur_disc, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
        )
    else:
        scheduler_dur_disc = None
    scaler = GradScaler(enabled=hps.train.fp16_run)

    # datamodule = DataModule(cwd / "buffer")
    # datamodule.setup("")
    # buffer=RolloutBuffer(cwd/"buffer")


    for epoch in range(epoch_str, hps.train.epochs + 1):
        try:
            # replay_buffer.clear()

            initial_buffer_RL(
                    rank,
                    epoch,
                    hps,
                    [net_g, net_g_pre],
                    [net_g,net_d,net_dur_disc],
                    [scheduler_g, scheduler_d, scheduler_dur_disc],
                    scaler,
                    [train_loader_rl,train_loader_rl],  # Use buffer data instead of train_loader
                    replay_buffer)

            if len(replay_buffer) < hps.train.batch_size:
                print(f"Buffer not filled yet: {len(replay_buffer)}/{hps.train.batch_size}")
                continue  # Skip training if the buffer is not filled

            # Sample data from the replay buffer
            buffer_data = replay_buffer.sample(hps.train.batch_size)
            print('buffer_length', len(buffer_data))
            if rank == 0:
                train_and_evaluate(
                    rank,
                    epoch,
                    hps,
                    [net_g,net_d,net_dur_disc],
                    [optim_g, optim_d, optim_dur_disc],
                    [scheduler_g, scheduler_d, scheduler_dur_disc],
                    scaler,
                    buffer_data,  # Use buffer data instead of train_loader
                    logger,
                    [writer, writer_eval],
                )
                # train_and_evaluate_syn(
                #     rank,
                #     epoch,
                #     hps,
                #     [net_g,net_d,net_dur_disc],
                #     [optim_g, optim_d, optim_dur_disc],
                #     [scheduler_g, scheduler_d, scheduler_dur_disc],
                #     scaler,
                #     buffer_data,  # Use buffer data instead of train_loader
                #     logger,
                #     [writer, writer_eval],
                # )
            else:
                train_and_evaluate(
                    rank,
                    epoch,
                    hps,
                    [net_g,net_d,net_dur_disc],
                    [optim_g, optim_d, optim_dur_disc],
                    [scheduler_g, scheduler_d, scheduler_dur_disc],
                    scaler,
                    buffer_data,
                    None,
                    None,
                )
                # train_and_evaluate_syn(
                #     rank,
                #     epoch,
                #     hps,
                #     [net_g,net_d,net_dur_disc],
                #     [optim_g, optim_d, optim_dur_disc],
                #     [scheduler_g, scheduler_d, scheduler_dur_disc],
                #     scaler,
                #     buffer_data,
                #     None,
                #     None,
                # )
            replay_buffer.clear()
        except Exception as e:
            print(e)
            torch.cuda.empty_cache()
        scheduler_g.step()
        scheduler_d.step()
        if net_dur_disc is not None:
            scheduler_dur_disc.step()
# Function to pad tensors to the maximum length in the last dimension
def pad_to_max_length(tensor, max_len):
    pad_size = max_len - tensor.shape[-1]
    pad_dims = [0, pad_size]  # Padding for the last dimension
    return F.pad(tensor, pad_dims, mode='constant', value=0)
def compute_log_probability_07222024(x, mu, logs):
        log_prob_list=[]

        # Determine the maximum length in the last dimension
        max_len = max(x[0].shape[-1], mu[0].shape[-1])

        for i in range(len(x)):
            # Pad tensors to the maximum length in the last dimension
            x_padded = pad_to_max_length(x[i], max_len)
            mu_padded = pad_to_max_length(mu[i], max_len)
            logs_padded = pad_to_max_length(logs[i], max_len)
            # Compute the standard deviation from logs

            mask = (x_padded != 0).float()

            mean_logs = logs_padded.mean()
            std_logs = logs_padded.std()
            logs_normalized = (logs_padded - mean_logs) / std_logs

            sigma = torch.exp(logs_normalized)

            # Compute the variance
            var = sigma ** 2

            # Compute the log probability
            log_prob = -0.5 * ((x_padded - mu_padded) ** 2 / var + (torch.log(2 * torch.pi * var)))

            # Apply the mask to ignore padded values
            log_prob = log_prob * mask
            # Sum the log probabilities over the last dimension (feature dimension)
            sample_num = log_prob.size(-1)

            log_prob = log_prob.sum(dim=-1)
            log_prob=log_prob/sample_num

            log_prob_list.append(log_prob)
        
        # Normalize by the number of samples
        # log_prob_tensor = torch.stack(log_prob_list)
        print('log_prob_list',log_prob_list)
        # print('sample_num',sample_num)
        # divided_log_prob_list = [tensor / sample_num for tensor in log_prob_list]
        # print('divided_log_prob_list',divided_log_prob_list)

        log_prob_tensor = torch.stack(log_prob_list)
        # normalized_probs = F.softmax(log_prob_tensor, dim=0)
        #2.Normalize by converting to probability, averaging, and converting back
        # avg_log_prob = total_log_prob - torch.log(torch.tensor(len(x), dtype=torch.float32))

        return log_prob_tensor

def compute_log_probability(x, mu, logs):
        log_prob_list=[]

        for i in range(len(x)):


            variance = torch.exp(logs[i])
            # print(f"Variance: {variance}")
            variance = torch.clamp(variance, min=1e-6)
            var = variance
            # Compute the log probability
            log_prob = -0.5 * (torch.log(2 * torch.pi * var) + ((x[i] - mu[i]) ** 2 / var))
            # print(f"Log Probability: {log_prob}")       
            # Apply the mask to ignore padded values
            # Sum the log probabilities over the last dimension (feature dimension)
            sample_num = log_prob.size(-1)
            # print(f"Sample Num: {sample_num}")
            log_prob = log_prob.sum(dim=-1)
            log_prob=log_prob.sum(dim=-1)/sample_num

            log_prob_list.append(log_prob)
        
        # Normalize by the number of samples
        # log_prob_tensor = torch.stack(log_prob_list)
        # print('log_prob_list',log_prob_list)
        # print('sample_num',sample_num)
        # divided_log_prob_list = [tensor / sample_num for tensor in log_prob_list]
        # print('divided_log_prob_list',divided_log_prob_list)

        log_prob_tensor = torch.stack(log_prob_list)
        # normalized_probs = F.softmax(log_prob_tensor, dim=0)
        #2.Normalize by converting to probability, averaging, and converting back
        # avg_log_prob = total_log_prob - torch.log(torch.tensor(len(x), dtype=torch.float32))

        return log_prob_tensor
        
        


def pad_tensors(tensors, pad_value=0):
    # Find the maximum length in the last dimension
    max_len = max(tensor.size(-1) for tensor in tensors)
    
    # Create a list to store the padded tensors
    padded_tensors = []
    
    for tensor in tensors:
        # Calculate the amount of padding needed
        pad_len = max_len - tensor.size(-1)
        # Pad the tensor
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_len), value=pad_value)
        padded_tensors.append(padded_tensor)
    
    # Stack the padded tensors
    stacked_tensors = torch.stack(padded_tensors)
    return stacked_tensors

# def pad_emo_audio_embed_real(emo_audio_embed_real, embedding_dim=1024):
#     """
#     Pads the emotion audio embeddings in the batch.

#     PARAMS
#     ------
#     emo_audio_embed_real: list of torch.FloatTensor
#         List of emotion audio embeddings, one per batch item.
#     embedding_dim: int
#         The dimension of the emotion audio embeddings.

#     RETURNS
#     -------
#     emo_audio_embed_real_padded: torch.FloatTensor
#         Padded emotion audio embeddings.
#     """
#     batch_size = len(emo_audio_embed_real)

#     # Initialize tensor for padded embeddings
#     emo_audio_embed_real_padded = torch.FloatTensor(batch_size, embedding_dim)
#     emo_audio_embed_real_padded.zero_()

#     # Pad each emotion audio embedding
#     for i, emo in enumerate(emo_audio_embed_real):
#         length = min(emo.size(0), embedding_dim)
#         emo_audio_embed_real_padded[i, :length] = emo[:length]

#     return emo_audio_embed_real_padded
def batchwise_emotion_regularizer(S_z, S_y):
    print('enter batchwise_emotion_regularizer')
    # Reduce ties and boost relative representation of infrequent labels
    # Get unique batch elements based on S_y
    unique_batch = torch.unique(S_y, dim=1)
    
    # Create separate lists for storing losses
    emo_losses = []

    for i in range(S_z.shape[0]):
        pred = S_z[i]
        target = S_y[i]
        unique_item = unique_batch[i]
        
        # Sample indices based on the unique items
        indices = torch.stack([random.choice((target==i).nonzero()[0]) for i in unique_item])
        unique_pred = pred[indices]

        # Calculate emotion feature loss for this set
        emo_loss = F.mse_loss(unique_pred, unique_item)
        emo_losses.append(emo_loss)

    # Return the losses for each set
    return torch.tensor(emo_losses, device=S_z.device)

    # unique_batch = torch.unique(targets, dim=1)
    # unique_pred = []
    # for pred, target, unique_item in zip(features, targets, unique_batch):
    #     indices = torch.stack([random.choice((target==i).nonzero()[0]) for i in unique_item])
    #     unique_pred.append(pred[indices])
    # emotion_feature = torch.stack(unique_pred)
    # emo_loss = F.mse_loss(emotion_feature, unique_batch)
    # return emo_loss

def initial_buffer_RL(
    rank, epoch,hps, nets, optims, schedulers, scaler, loaders, buffer
):
    with torch.no_grad():
        net_g, net_g_pre = nets
        optim_g, optim_d, optim_dur_disc = optims
        scheduler_g, scheduler_d, scheduler_dur_disc = schedulers
        train_loader, eval_loader = loaders
        train_loader.batch_sampler.set_epoch(epoch)
        global global_step
        net_g.eval()
        net_g_pre.eval()

        for batch_idx, (
            emo_audio_embed_syn,
            emo_audio_path_list_syn,
            goal_audio_list_syn,
            input_audio_list_syn,
            emo_audio_embed_real,
            emo_audio_path_list_real,
            input_audio_list_real,
            goal_syn_spec,
            goal_syn_spec_len,
            goal_syn_wav,
            goal_syn_wav_len,
            input_syn_spec,
            input_syn_spec_len,
            input_syn_wav,
            input_syn_wav_len,
            input_real_spec,
            input_real_spec_len,
            input_real_wav,
            input_real_wav_len,
            emo_e,
            emo_real_e, 
            spec_emo_real_padded,
            spec_emo_real_lengths,
            wav_emo_real_padded,
            wav_emo_real_lengths,
            spec_realgoal_padded,
            spec_realgoal_lengths,
            wav_realgoal_padded,
            wav_realgoal_lengths,
            real_goal_list
          

        ) in enumerate(tqdm(train_loader)):
            print('enter initial_buffer_RL')

            goal_syn_spec, goal_syn_spec_len = goal_syn_spec.cuda(rank, non_blocking=True), goal_syn_spec_len.cuda(
            rank, non_blocking=True
        )
            goal_syn_wav, goal_syn_wav_len = goal_syn_wav.cuda(rank, non_blocking=True), goal_syn_wav_len.cuda(
            rank, non_blocking=True
        )
            input_syn_spec, input_syn_spec_len = input_syn_spec.cuda(rank, non_blocking=True), input_syn_spec_len.cuda(
            rank, non_blocking=True
        )
            input_syn_wav, input_syn_wav_len = input_syn_wav.cuda(rank, non_blocking=True), input_syn_wav_len.cuda(
            rank, non_blocking=True
        )
            input_real_spec, input_real_spec_len = input_real_spec.cuda(rank, non_blocking=True), input_real_spec_len.cuda(
            rank, non_blocking=True
        )
            input_real_wav, input_real_wav_len = input_real_wav.cuda(rank, non_blocking=True), input_real_wav_len.cuda(
            rank, non_blocking=True
        )
            emo_e, emo_real_e = emo_e.cuda(rank, non_blocking=True), emo_real_e.cuda(
            rank, non_blocking=True
        )
            spec_emo_real_padded, spec_emo_real_lengths = spec_emo_real_padded.cuda(rank, non_blocking=True), spec_emo_real_lengths.cuda(
            rank, non_blocking=True
        )
            wav_emo_real_padded, wav_emo_real_lengths = wav_emo_real_padded.cuda(rank, non_blocking=True), wav_emo_real_lengths.cuda(
            rank, non_blocking=True
        )
            spec_realgoal_padded, spec_realgoal_lengths = spec_realgoal_padded.cuda(rank, non_blocking=True), spec_realgoal_lengths.cuda(
            rank, non_blocking=True
        )
            wav_realgoal_padded, wav_realgoal_lengths = wav_realgoal_padded.cuda(rank, non_blocking=True), wav_realgoal_lengths.cuda(
            rank, non_blocking=True
        )


            if buffer.is_full():
                break
            if net_g.module.use_noise_scaled_mas:
                current_mas_noise_scale = (
                    net_g.module.mas_noise_scale_initial
                    - net_g.module.noise_scale_delta * global_step
                )
                net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)



            num_samples = len(emo_audio_embed_real)
            log_p_real = torch.empty(num_samples, device=rank)
            log_p_pre_real = torch.empty(num_samples, device=rank)
            log_p_pre_syn = torch.empty(num_samples, device=rank)
            emotion_reward_list = torch.empty(num_samples, device=rank)
            speaker_reward_list = torch.empty(num_samples, device=rank)
            # l1_losses = torch.empty(num_samples, device=rank)


            # print('emo_audio_embed_real',emo_audio_embed_real)
            # print('emo_audio_embed_syn',emo_audio_embed_syn)
            emo_audio_embed_real = [torch.FloatTensor(np.load(path)).to(rank) for path in emo_audio_embed_real]
            emo_audio_embed_real = torch.stack(emo_audio_embed_real)
            emo_audio_embed_syn = [torch.FloatTensor(np.load(path_syn)).to(rank) for path_syn in emo_audio_embed_syn]
            emo_audio_embed_syn = torch.stack(emo_audio_embed_syn)
            print('enter infer_sts_RL')



            (
                y_hat,
                mu, logs,y_mask,
                (z,z_p, m_q, logs_q)
            ) = net_g.module.infer_sts_RL(
                input_audio_list_real,
                emo_audio_embed_real,
                hps.data.sampling_rate,
                hps.data.filter_length,
                hps.data.hop_length,
                hps.data.win_length,

            )
            # base_dir = os.path.join(hps.model_dir, "train_audio")
            # if not os.path.exists(base_dir):
            #     os.makedirs(base_dir)
            # for audio_i in range(len(y_hat)):
            #     output_path_real=os.path.join(base_dir, "real_{}_{}.wav".format(global_step,audio_i))
            #     soundfile.write(output_path_real, y_hat[audio_i][0,0].data.cpu().float().numpy(), hps.data.sampling_rate)
            # print('line722')
            
            (
                y_hat_syn_inf,
                mu_syn_inf, logs_syn_inf,y_mask_syn_inf,
                (z_syn_inf,z_p_syn_inf, m_q_syn_inf, logs_q_syn_inf)
            ) = net_g.module.infer_sts_RL(
                input_audio_list_syn,
                emo_audio_embed_syn,
                hps.data.sampling_rate,
                hps.data.filter_length,
                hps.data.hop_length,
                hps.data.win_length,

            )
            # for audio_i_syn in range(len(y_hat_syn_inf)):
            #     output_path_real_syn=os.path.join(base_dir, "syn_{}_{}.wav".format(global_step,audio_i_syn))
            #     soundfile.write(output_path_real_syn, y_hat_syn_inf[audio_i_syn][0,0].data.cpu().float().numpy(), hps.data.sampling_rate)
            

            with torch.no_grad():
                (
                    y_hat_forward,l_length_forward,attn_forward, ids_slice_forward, x_mask_forward,z_mask_forward,
                    (z_forward,z_p_forward,m_p_forward,logs_p_forward, m_q_forward, logs_q_forward),(hidden_x_forward, logw_forward, logw_),
                (z_forward,z_p_forward, m_q_forward, logs_q_forward),mu_forward,logs_forward
                ) = net_g.module.forward(
                    spec_realgoal_padded,
                    spec_realgoal_lengths,
                    input_real_spec,
                    input_real_spec_len,
                    emo_real_e,


                )
                
                print('after net_g.module.forward')

                (
                    y_hat_pre,l_length_pre,attn_pre, ids_slice_pre, x_mask_pre,z_mask_pre,
                    (z_pre,z_p_pre,m_p_pre,logs_p_pre, m_q_pre, logs_q_pre),(hidden_x_pre, logw_pre, logw_),
                (z_pre,z_p_pre, m_q_pre, logs_q_pre),mu_pre,logs_pre
                ) = net_g_pre.module.forward(
                    spec_realgoal_padded,
                    spec_realgoal_lengths,
                    input_real_spec,
                    input_real_spec_len,
                    emo_real_e,
    
                )
            print('outside of infer_sts_RL')
            log_p_real = compute_log_probability(y_hat_forward, mu_forward, logs_forward)
            log_p_pre_real = compute_log_probability(y_hat_pre, mu_pre, logs_pre)
            log_p_real=torch.tensor(log_p_real, requires_grad=True).to(rank)
            log_p_pre_real=torch.tensor(log_p_pre_real, requires_grad=True).to(rank)
            KL=log_p_real-log_p_pre_real
            print('outside of infer_sts_RL641')

            # ###kl_mel
            # print('enter kl mel')
            # mel = spec_to_mel_torch(
            # input_real_spec,
            # hps.data.filter_length,
            # hps.data.n_mel_channels,
            # hps.data.sampling_rate,
            # hps.data.mel_fmin,
            # hps.data.mel_fmax,
            # )



            # y_mel = commons.slice_segments(
            #     mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            # )

            
            # y_hat_mel = mel_spectrogram_torch_RL(
            #     o_l1,
            #     hps.data.filter_length,
            #     hps.data.n_mel_channels,
            #     hps.data.sampling_rate,
            #     hps.data.hop_length,
            #     # 2065,
            #     hps.data.win_length,
            #     hps.data.mel_fmin,
            #     hps.data.mel_fmax,
            # )
            # # print('initial buffer rl y_mel',y_mel.shape)
            # # print('initial buffer rl y_hat_mel',y_hat_mel.shape)
            # l1_losses = []

            # for l1_i in range(y_mel.shape[0]):
            #     # Compute L1 loss without reduction for each sample
            #     l1_loss = F.l1_loss(y_mel[l1_i], y_hat_mel[l1_i], reduction='mean') * hps.train.c_mel
            #     l1_losses.append(l1_loss)

            # # Convert the list of losses to a tensor
            # l1_losses = torch.stack(l1_losses)
            # # print('l1_loss1',l1_losses)
            # l1_losses=l1_losses.to(y_mel.device)


            ###emotion reward real
            wav_real=[]
            for path in emo_audio_path_list_real:
                wav_real.append(librosa.load(path, 16000)[0])
            wav_real_cuda=[wavform for wavform in wav_real]

            wav_emb_real = [process_func(np.expand_dims(wav, 0), 16000, embeddings=True)[0] for wav in wav_real_cuda]
            wav_emb_va = [process_func(np.expand_dims(wav, 0), 16000, embeddings=False)[0] for wav in wav_real_cuda]

            wav_emb_real = torch.tensor(wav_emb_real).to(rank)
            wav_emb_va = torch.tensor(wav_emb_va).to(rank)

            y_hat_emb = [process_func(np.expand_dims(y_hat_i[0][0].cpu(), 0), 16000, embeddings=True)[0] for y_hat_i in y_hat]
            y_hat_va = [process_func(np.expand_dims(y_hat_i[0][0].cpu(), 0), 16000, embeddings=False)[0] for y_hat_i in y_hat]

            y_hat_emb = torch.tensor(y_hat_emb).to(rank)
            y_hat_va = torch.tensor(y_hat_va).to(rank)
            
            # cos_sims = dot(wav_emb_real, y_hat_emb) / (norm(wav_emb_real) * norm(y_hat_emb))
            cos_sims = F.cosine_similarity(wav_emb_real, y_hat_emb, dim=-1) 
            cos_sims=cos_sims.to(rank)

            # emotion_reward_list = (cos_sims + 1.0) / 2 

            emotion_reward_list=emotion_reward_list.to(rank)
            # Calculate Euclidean Distance
            # Ensure that both tensors are on the same device and flattened
            wav_emb_real_flat = wav_emb_real.view(wav_emb_real.size(0), -1)  # Flatten embeddings
            y_hat_emb_flat = y_hat_emb.view(y_hat_emb.size(0), -1)  # Flatten embeddings

            # Compute pairwise Euclidean distances
            euclidean_distances = 1-torch.cdist(wav_emb_va, y_hat_va, p=2)
            euclidean_distances=euclidean_distances.to(rank)

            euclidean_distances_emowemo = 1-torch.cdist(wav_emb_real_flat, wav_emb_real_flat, p=2)
            latent_sim=wav_emb_real_flat @ y_hat_emb_flat.T
            euclidean_distances_emowemo=euclidean_distances_emowemo.to(rank)
            print('euclidean_distances_syn',euclidean_distances.shape)
            print('latent_sim_syn',latent_sim.shape)
            emotion_loss=batchwise_emotion_regularizer(euclidean_distances,latent_sim)
            print('emotion_loss',emotion_loss)

            ###emotion reward syn
            wav_syn=[]
            for path_syn in emo_audio_path_list_syn:
                wav_syn.append(librosa.load(path_syn, 16000)[0])
            wav_syn_cuda=[wavform_syn for wavform_syn in wav_syn]
            wav_emb_syn = [process_func(np.expand_dims(wav_syn, 0), 16000, embeddings=True)[0] for wav_syn in wav_syn_cuda]
            wav_emb_va_syn = [process_func(np.expand_dims(wav_syn, 0), 16000, embeddings=False)[0] for wav_syn in wav_syn_cuda]

            wav_emb_syn = torch.tensor(wav_emb_syn).to(rank)
            wav_emb_va_syn = torch.tensor(wav_emb_va_syn).to(rank)

            y_hat_syn_inf_emb = [process_func(np.expand_dims(y_hat_syn_inf_i[0][0].cpu(), 0), 16000, embeddings=True)[0] for y_hat_syn_inf_i in y_hat_syn_inf]
            y_hat_syn_inf_va = [process_func(np.expand_dims(y_hat_syn_inf_i[0][0].cpu(), 0), 16000, embeddings=False)[0] for y_hat_syn_inf_i in y_hat_syn_inf]
            
            y_hat_syn_inf_emb = torch.tensor(y_hat_syn_inf_emb).to(rank)
            y_hat_syn_inf_va = torch.tensor(y_hat_syn_inf_va).to(rank)

            # cos_sims = dot(wav_emb_real, y_hat_emb) / (norm(wav_emb_real) * norm(y_hat_emb))
            cos_sims_syn_inf = F.cosine_similarity(wav_emb_syn, y_hat_syn_inf_emb, dim=-1) 
            # emotion_reward_list_syn = (cos_sims_syn_inf + 1.0) / 2 
            cos_sims_syn_inf=cos_sims_syn_inf.to(rank)
            # Calculate Euclidean Distance
            # Ensure that both tensors are on the same device and flattened
            wav_emb_syn_flat = wav_emb_syn.view(wav_emb_syn.size(0), -1)  # Flatten embeddings
            y_hat_syn_inf_emb_flat = y_hat_syn_inf_emb.view(y_hat_syn_inf_emb.size(0), -1)  # Flatten embeddings

            # Compute pairwise Euclidean distances
            euclidean_distances_syn = 1-torch.cdist(wav_emb_va_syn, y_hat_syn_inf_va, p=2)
            euclidean_distances_syn=euclidean_distances_syn.to(rank)
            latent_sim_syn=wav_emb_syn_flat @ y_hat_syn_inf_emb_flat.T


            euclidean_distances_syn_emowemo = 1-torch.cdist(wav_emb_syn_flat, wav_emb_syn_flat, p=2)
            euclidean_distances_syn_emowemo=euclidean_distances_syn_emowemo.to(rank)
            print('euclidean_distances_syn',euclidean_distances_syn.shape)
            print('latent_sim_syn',latent_sim_syn.shape)

            emotion_loss_syn=batchwise_emotion_regularizer(euclidean_distances_syn,latent_sim_syn)
            print('emotion_loss_syn',emotion_loss_syn)
            ###speaker reward real
            pcm_list = [torchaudio.load(path, normalize=False)[0] for path in input_audio_list_real]
            pcm2_list = [y_hat[i] for i in range(len(pcm_list))]
            # speaker_reward_list = [speaker_func(pcm.to(rank), pcm2.to(rank), hps.data.sampling_rate)[0] for pcm, pcm2 in zip(pcm_list, pcm2_list)]
            # speaker_reward_list = torch.tensor(speaker_reward_list).to(rank)
            emb1 = [speaker_func(pcm.to(rank), pcm2.to(rank), hps.data.sampling_rate)[1] for pcm, pcm2 in zip(pcm_list, pcm2_list)]
            emb2 = [speaker_func(pcm.to(rank), pcm2.to(rank), hps.data.sampling_rate)[2] for pcm, pcm2 in zip(pcm_list, pcm2_list)]
            emb1 = torch.stack(emb1).to(rank)
            emb2 = torch.stack(emb2).to(rank)
            emb1 = emb1.view(emb1.size(0), -1)  # Flatten embeddings
            emb2 = emb2.view(emb2.size(0), -1)  # Flatten embeddings

            speaker_reward_list = 1-torch.cdist(emb1, emb2, p=2)
            speaker_reward_list = torch.tensor(speaker_reward_list).to(rank)
            # print('speaker_reward_list',speaker_reward_list.shape)
            # print('euclidean_distances',euclidean_distances.shape)

            emotion_loss=batchwise_emotion_regularizer(euclidean_distances,speaker_reward_list)
            print('emotion_loss',emotion_loss)
            ###speaker reward syn
            pcm_list_syn = [torchaudio.load(path_syn, normalize=False)[0] for path_syn in input_audio_list_syn]
            pcm2_list_syn = [y_hat_syn_inf[i] for i in range(len(pcm_list_syn))]
            # speaker_reward_list_syn = [speaker_func(pcm_syn.to(rank), pcm2_syn.to(rank), hps.data.sampling_rate) for pcm_syn, pcm2_syn in zip(pcm_list_syn, pcm2_list_syn)]
            # speaker_reward_list_syn = torch.tensor(speaker_reward_list_syn).to(rank)
            emb1_syn = [speaker_func(pcm.to(rank), pcm2.to(rank), hps.data.sampling_rate)[1] for pcm, pcm2 in zip(pcm_list_syn, pcm2_list_syn)]
            emb2_syn = [speaker_func(pcm.to(rank), pcm2.to(rank), hps.data.sampling_rate)[2] for pcm, pcm2 in zip(pcm_list_syn, pcm2_list_syn)]
            emb1_syn = torch.stack(emb1_syn).to(rank)
            emb2_syn = torch.stack(emb2_syn).to(rank)
            emb1_syn = emb1_syn.view(emb1_syn.size(0), -1)  # Flatten embeddings
            emb2_syn = emb2_syn.view(emb2_syn.size(0), -1)  # Flatten embeddings
            speaker_reward_list_syn = 1-torch.cdist(emb1_syn, emb2_syn, p=2)
            speaker_reward_list_syn = torch.tensor(speaker_reward_list_syn).to(rank)
            emotion_loss_syn=batchwise_emotion_regularizer(euclidean_distances_syn,speaker_reward_list_syn)
            print('enter line 701')


           
                

            # emo_audio_embed_syn = [torch.FloatTensor(np.load(path_syn)).to(rank) for path_syn in emo_audio_embed_syn]

            # emo_audio_embed_syn = torch.stack(emo_audio_embed_syn)

            y_hat_pre_syn,l_length_pre_syn,attn_pre_syn, ids_slice_pre_syn, x_mask_pre_syn,z_mask_pre_syn,(z_pre_syn,z_p_pre_syn,m_p_pre_syn,logs_p_pre_syn, m_q_pre_syn, logs_q_pre_syn),(hidden_x_pre_syn, logw_pre_syn, logw_),(z_pre_syn,z_p_pre_syn, m_q_pre_syn, logs_q_pre_syn),mu_pre_syn,logs_pre_syn= net_g_pre.module.forward(goal_syn_spec,
                goal_syn_spec_len,
                input_syn_spec,
                input_syn_spec_len,
                emo_e,

            )
            # print('y_hat_pre_syn',y_hat_pre_syn.shape)
            y_hat_cur_syn,l_length_cur_syn,attn_cur_syn, ids_slice_cur_syn, x_mask_cur_syn,z_mask_cur_syn,(z_cur_syn,z_p_cur_syn,m_p_cur_syn,logs_p_cur_syn, m_q_cur_syn, logs_q_cur_syn),(hidden_x_cur_syn, logw_cur_syn, logw_),(z_cur_syn,z_p_cur_syn, m_q_cur_syn, logs_q_cur_syn),mu_cur_syn,logs_cur_syn= net_g_pre.module.forward(
                goal_syn_spec,
                goal_syn_spec_len,
                input_syn_spec,
                input_syn_spec_len,
                emo_e,
            )
            # print('mu_cur_syn',mu_cur_syn.shape)
            # print('logs_cur_syn',logs_cur_syn.shape)

            log_p_pre_syn = compute_log_probability(y_hat_pre_syn, mu_cur_syn, logs_cur_syn)
            # print('log_p_pre_syn737',log_p_pre_syn)

            log_p_pre_syn = torch.tensor(log_p_pre_syn, device=rank, requires_grad=True)


            print()
            buffer.push({
                # 'emo_audio_embed_syn': emo_audio_embed_syn,
                # 'emo_audio_path_list_syn': emo_audio_path_list_syn,
                # 'goal_audio_list_syn': goal_audio_list_syn,
                # 'input_audio_list_syn': input_audio_list_syn,
                # 'log_p_real': log_p_real.cpu().numpy(),
                # 'log_p_pre_real': log_p_pre_real.cpu().numpy(),
                'speaker_reward_list': speaker_reward_list.cpu().numpy(),
                'emotion_reward_list': cos_sims.cpu().numpy(),
                'log_p_pre_syn':log_p_pre_syn.cpu().numpy(),
                'goal_syn_spec':goal_syn_spec.cpu().numpy(),
                'goal_syn_spec_len':goal_syn_spec_len.cpu().numpy(),
                'goal_syn_wav':goal_syn_wav.cpu().numpy(),
                'goal_syn_wav_len':goal_syn_wav_len.cpu().numpy(),
                'input_syn_spec':input_syn_spec.cpu().numpy(),
                'input_syn_spec_len':input_syn_spec_len.cpu().numpy(),
                'input_syn_wav':input_syn_wav.cpu().numpy(),
                'input_syn_wav_len':input_syn_wav_len.cpu().numpy(),
                'input_real_spec':input_real_spec.cpu().numpy(),
                'input_real_spec_len':input_real_spec_len.cpu().numpy(),
                'input_real_wav':input_real_wav.cpu().numpy(),
                'input_real_wav_len':input_real_wav_len.cpu().numpy(),
                'emo_e':emo_e.cpu().numpy(),
                'emo_real_e':emo_real_e.cpu().numpy(), 
                # 'l1_losses':l1_losses.cpu().numpy(),
                'y_hat':torch.tensor(y_hat_forward).cpu().numpy(),
                'kl':KL.cpu().numpy(),
                'spec_emo_real_padded':spec_emo_real_padded.cpu().numpy(),
                'spec_emo_real_lengths':spec_emo_real_lengths.cpu().numpy(),
                'wav_emo_real_padded':wav_emo_real_padded.cpu().numpy(),
                'wav_emo_real_lengths':wav_emo_real_lengths.cpu().numpy(),
                'speaker_reward_list_syn': speaker_reward_list_syn.cpu().numpy(),
                'emotion_reward_list_syn': cos_sims_syn_inf.cpu().numpy(),
                'emotion_loss':emotion_loss.cpu().numpy(),
                'emotion_loss_syn':emotion_loss_syn.cpu().numpy(),
                'spec_realgoal_padded':spec_realgoal_padded.cpu().numpy(),
                'spec_realgoal_lengths':spec_realgoal_lengths.cpu().numpy(),
                'wav_realgoal_padded':wav_realgoal_padded.cpu().numpy(),
                'wav_realgoal_lengths':wav_realgoal_lengths.cpu().numpy(),
            })
            
            # if global_step % hps.train.log_interval == 0:
            #     traindata=buffer.load_data()
            #     train_and_evaluate_RL(rank, epoch,hps, nets, optims, schedulers, scaler, loaders, logger, writers,traindata)

def initial_buffer_RL_07222024(
    rank, epoch,hps, nets, optims, schedulers, scaler, loaders, buffer
):
    with torch.no_grad():
        net_g, net_g_pre = nets
        optim_g, optim_d, optim_dur_disc = optims
        scheduler_g, scheduler_d, scheduler_dur_disc = schedulers
        train_loader, eval_loader = loaders
        train_loader.batch_sampler.set_epoch(epoch)
        global global_step
        net_g.eval()
        net_g_pre.eval()
        # net_g.train()
        # net_g_pre.train()
        # if net_dur_disc is not None:
        #     net_dur_disc.train()

        for batch_idx, (
            emo_audio_embed_syn,
            emo_audio_path_list_syn,
            goal_audio_list_syn,
            input_audio_list_syn,
            emo_audio_embed_real,
            emo_audio_path_list_real,
            input_audio_list_real,
            goal_syn_spec,
            goal_syn_spec_len,
            goal_syn_wav,
            goal_syn_wav_len,
            input_syn_spec,
            input_syn_spec_len,
            input_syn_wav,
            input_syn_wav_len,
            input_real_spec,
            input_real_spec_len,
            input_real_wav,
            input_real_wav_len,
            emo_e,
            emo_real_e,        

        ) in enumerate(tqdm(train_loader)):
            print('enter initial_buffer_RL')

            goal_syn_spec, goal_syn_spec_len = goal_syn_spec.cuda(rank, non_blocking=True), goal_syn_spec_len.cuda(
            rank, non_blocking=True
        )
            goal_syn_wav, goal_syn_wav_len = goal_syn_wav.cuda(rank, non_blocking=True), goal_syn_wav_len.cuda(
            rank, non_blocking=True
        )
            input_syn_spec, input_syn_spec_len = input_syn_spec.cuda(rank, non_blocking=True), input_syn_spec_len.cuda(
            rank, non_blocking=True
        )
            input_syn_wav, input_syn_wav_len = input_syn_wav.cuda(rank, non_blocking=True), input_syn_wav_len.cuda(
            rank, non_blocking=True
        )
            input_real_spec, input_real_spec_len = input_real_spec.cuda(rank, non_blocking=True), input_real_spec_len.cuda(
            rank, non_blocking=True
        )
            input_real_wav, input_real_wav_len = input_real_wav.cuda(rank, non_blocking=True), input_real_wav_len.cuda(
            rank, non_blocking=True
        )
            emo_e, emo_real_e = emo_e.cuda(rank, non_blocking=True), emo_real_e.cuda(
            rank, non_blocking=True
        )

            if buffer.is_full():
                break
            if net_g.module.use_noise_scaled_mas:
                current_mas_noise_scale = (
                    net_g.module.mas_noise_scale_initial
                    - net_g.module.noise_scale_delta * global_step
                )
                net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)


            # log_p_real=[]
            # log_p_pre_real=[]
            # log_p_pre_syn=[]
            # emotion_reward_list=[]
            # speaker_reward_list=[]

            # l1_losses = []
            num_samples = len(emo_audio_embed_real)
            log_p_real = torch.empty(num_samples, device=rank)
            log_p_pre_real = torch.empty(num_samples, device=rank)
            log_p_pre_syn = torch.empty(num_samples, device=rank)
            emotion_reward_list = torch.empty(num_samples, device=rank)
            speaker_reward_list = torch.empty(num_samples, device=rank)
            # l1_losses = torch.empty(num_samples, device=rank)

            ###inference real audio
            # for i in range(len(emo_audio_embed_real)):
            # for i in range(num_samples):
            #     # emo_emb_path=emo_audio_embed_real[i]
            #     # emo = torch.FloatTensor(np.load(emo_emb_path)).to(rank)
            # print('emo_audio_embed_real',emo_audio_embed_real)
            emo_audio_embed_real = [torch.FloatTensor(np.load(path)).to(rank) for path in emo_audio_embed_real]
        #     input_audio_list_real = [audio, sample_rate = librosa.load(audio_src_path, sr=sampling_rate)
        # audio = torch.tensor(audio).float().to(rank) for path in input_audio_list_real]
            
            # Stack tensors for batch processing
            # input_audio_list_real = torch.stack(input_audio_list_real)
            emo_audio_embed_real = torch.stack(emo_audio_embed_real)
            print('enter infer_sts_RL')
            # print('input_audio_list_real',input_audio_list_real)
            # print('emo_audio_embed_real',emo_audio_embed_real)


            (
                y_hat,
                mu, logs,y_mask,
                (z,z_p, m_q, logs_q)
            ) = net_g.module.infer_sts_RL(
                input_audio_list_real,
                emo_audio_embed_real,
                hps.data.sampling_rate,
                hps.data.filter_length,
                hps.data.hop_length,
                hps.data.win_length,

            )
            with torch.no_grad():
                (
                    y_hat_forward,l_length_forward,attn_forward, ids_slice_forward, x_mask_forward,z_mask_forward,
                    (z_forward,z_p_forward,m_p_forward,logs_p_forward, m_q_forward, logs_q_forward),(hidden_x_forward, logw_forward, logw_),
                (z_forward,z_p_forward, m_q_forward, logs_q_forward),mu_forward,logs_forward
                ) = net_g.module.forward(
                    input_real_spec,
                    input_real_spec_len,
                    input_real_spec,
                    input_real_spec_len,
                    emo_real_e,


                )
                
                print('after net_g.module.forward')

                (
                    y_hat_pre,l_length_pre,attn_pre, ids_slice_pre, x_mask_pre,z_mask_pre,
                    (z_pre,z_p_pre,m_p_pre,logs_p_pre, m_q_pre, logs_q_pre),(hidden_x_pre, logw_pre, logw_),
                (z_pre,z_p_pre, m_q_pre, logs_q_pre),mu_pre,logs_pre
                ) = net_g_pre.module.forward(
                    input_real_spec,
                    input_real_spec_len,
                    input_real_spec,
                    input_real_spec_len,
                    emo_real_e,
    
                )
            print('outside of infer_sts_RL')
            log_p_real = compute_log_probability(y_hat_forward, mu_forward, logs_forward)
            log_p_pre_real = compute_log_probability(y_hat_pre, mu_pre, logs_pre)
            log_p_real=torch.tensor(log_p_real, requires_grad=True).to(rank)
            log_p_pre_real=torch.tensor(log_p_pre_real, requires_grad=True).to(rank)
            KL=log_p_real-log_p_pre_real
            print('outside of infer_sts_RL641')

            # ###kl_mel
            # print('enter kl mel')
            # mel = spec_to_mel_torch(
            # input_real_spec,
            # hps.data.filter_length,
            # hps.data.n_mel_channels,
            # hps.data.sampling_rate,
            # hps.data.mel_fmin,
            # hps.data.mel_fmax,
            # )



            # y_mel = commons.slice_segments(
            #     mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            # )

            
            # y_hat_mel = mel_spectrogram_torch_RL(
            #     o_l1,
            #     hps.data.filter_length,
            #     hps.data.n_mel_channels,
            #     hps.data.sampling_rate,
            #     hps.data.hop_length,
            #     # 2065,
            #     hps.data.win_length,
            #     hps.data.mel_fmin,
            #     hps.data.mel_fmax,
            # )
            # # print('initial buffer rl y_mel',y_mel.shape)
            # # print('initial buffer rl y_hat_mel',y_hat_mel.shape)
            # l1_losses = []

            # for l1_i in range(y_mel.shape[0]):
            #     # Compute L1 loss without reduction for each sample
            #     l1_loss = F.l1_loss(y_mel[l1_i], y_hat_mel[l1_i], reduction='mean') * hps.train.c_mel
            #     l1_losses.append(l1_loss)

            # # Convert the list of losses to a tensor
            # l1_losses = torch.stack(l1_losses)
            # # print('l1_loss1',l1_losses)
            # l1_losses=l1_losses.to(y_mel.device)


            ###emotion reward
            wav_real=[]
            for path in emo_audio_path_list_real:
                wav_real.append(librosa.load(path, 16000)[0])
            wav_real_cuda=[wavform for wavform in wav_real]

            wav_emb_real = [process_func(np.expand_dims(wav, 0), 16000, embeddings=True)[0] for wav in wav_real_cuda]

            wav_emb_real = torch.tensor(wav_emb_real).to(rank)
            print('outside of infer_sts_RL696')


            

            y_hat_emb = [process_func(np.expand_dims(y_hat_i[0][0].cpu(), 0), 16000, embeddings=True)[0] for y_hat_i in y_hat]
            print('outside of infer_sts_RL672')

            
            y_hat_emb = torch.tensor(y_hat_emb).to(rank)
            

            # cos_sims = dot(wav_emb_real, y_hat_emb) / (norm(wav_emb_real) * norm(y_hat_emb))
            cos_sims = F.cosine_similarity(wav_emb_real, y_hat_emb, dim=1) 
            print('outside of infer_sts_RL710')


            emotion_reward_list = (cos_sims + 1.0) / 2 
            print('outside of infer_sts_RL714')

            emotion_reward_list=emotion_reward_list.to(rank)
            # Calculate Euclidean Distance
            # Ensure that both tensors are on the same device and flattened
            wav_emb_real_flat = wav_emb_real.view(wav_emb_real.size(0), -1)  # Flatten embeddings
            y_hat_emb_flat = y_hat_emb.view(y_hat_emb.size(0), -1)  # Flatten embeddings
            print('outside of infer_sts_RL721')

            # Compute pairwise Euclidean distances
            euclidean_distances = torch.cdist(wav_emb_real_flat, y_hat_emb_flat, p=2)
            euclidean_distances = -euclidean_distances
            euclidean_distances=euclidean_distances.to(rank)
            print('outside of infer_sts_RL727')


            ###speaker reward
            pcm_list = [torchaudio.load(path, normalize=False)[0] for path in input_audio_list_real]
            print('outside of infer_sts_RL728')
            print('y_hat',len(y_hat))
            print('pcm_list',len(pcm_list))
            pcm2_list = [y_hat[i] for i in range(len(pcm_list))]
            # for pcm, pcm2 in zip(pcm_list, pcm2_list):
                # print('pcm.to(rank)',pcm.to(rank))
                # print('pcm.to(rank)',pcm2.to(rank))
            print('outside of infer_sts_RL738')

            speaker_reward_list = [speaker_func(pcm.to(rank), pcm2.to(rank), hps.data.sampling_rate)[0] for pcm, pcm2 in zip(pcm_list, pcm2_list)]
            print('outside of speaker_reward_list')
            # print('initial_buffer speaker_reward_list',speaker_reward_list)
            speaker_reward_list = torch.tensor(speaker_reward_list).to(rank)

            # speaker_reward_list=torch.tensor(speaker_reward_list).to(rank)
            # emotion_reward_list=torch.tensor(emotion_reward_list).to(rank)
            # log_p_real=torch.tensor(log_p_real).to(rank)
            # log_p_pre_real=torch.tensor(log_p_pre_real).to(rank)
            # l1_losses=torch.tensor(l1_losses).to(rank)

            # log_p_real = log_p_real.cpu()
            # log_p_pre_real = log_p_pre_real.cpu()
            #   # This line has been modified for the tensor initialization without a for-loop
            # emotion_reward_list = torch.tensor(emotion_reward_list, device=rank)
            # speaker_reward_list = torch.tensor(speaker_reward_list, device=rank)
            # l1_losses = l1_losses.cpu()

            print('enter line 701')

            ###inference syn audio
            # emo_audio_embed_syn = [torch.FloatTensor(np.load(path)).to(rank) for path in emo_audio_embed_syn]
            # input_audio_list_syn = [torch.FloatTensor(np.load(path,allow_pickle=True)).to(rank) for path in input_audio_list_syn]
            
            # Stack tensors for batch processing
            # input_audio_list_syn = torch.stack(input_audio_list_syn)
            # emo_audio_embed_syn = torch.stack(emo_audio_embed_syn)
           
                

            emo_audio_embed_syn = [torch.FloatTensor(np.load(path_syn)).to(rank) for path_syn in emo_audio_embed_syn]

            emo_audio_embed_syn = torch.stack(emo_audio_embed_syn)

            y_hat_pre_syn,l_length_pre_syn,attn_pre_syn, ids_slice_pre_syn, x_mask_pre_syn,z_mask_pre_syn,(z_pre_syn,z_p_pre_syn,m_p_pre_syn,logs_p_pre_syn, m_q_pre_syn, logs_q_pre_syn),(hidden_x_pre_syn, logw_pre_syn, logw_),(z_pre_syn,z_p_pre_syn, m_q_pre_syn, logs_q_pre_syn),mu_pre_syn,logs_pre_syn= net_g_pre.module.forward(goal_syn_spec,
                goal_syn_spec_len,
                input_syn_spec,
                input_syn_spec_len,
                emo_e,

            )
            # print('y_hat_pre_syn',y_hat_pre_syn.shape)
            y_hat_cur_syn,l_length_cur_syn,attn_cur_syn, ids_slice_cur_syn, x_mask_cur_syn,z_mask_cur_syn,(z_cur_syn,z_p_cur_syn,m_p_cur_syn,logs_p_cur_syn, m_q_cur_syn, logs_q_cur_syn),(hidden_x_cur_syn, logw_cur_syn, logw_),(z_cur_syn,z_p_cur_syn, m_q_cur_syn, logs_q_cur_syn),mu_cur_syn,logs_cur_syn= net_g_pre.module.forward(
                goal_syn_spec,
                goal_syn_spec_len,
                input_syn_spec,
                input_syn_spec_len,
                emo_e,
            )
            # print('mu_cur_syn',mu_cur_syn.shape)
            # print('logs_cur_syn',logs_cur_syn.shape)

            log_p_pre_syn = compute_log_probability(y_hat_pre_syn, mu_cur_syn, logs_cur_syn)
            # print('log_p_pre_syn737',log_p_pre_syn)

            log_p_pre_syn = torch.tensor(log_p_pre_syn, device=rank, requires_grad=True)


            print()
            buffer.push({
                # 'emo_audio_embed_syn': emo_audio_embed_syn,
                # 'emo_audio_path_list_syn': emo_audio_path_list_syn,
                # 'goal_audio_list_syn': goal_audio_list_syn,
                # 'input_audio_list_syn': input_audio_list_syn,
                # 'log_p_real': log_p_real.cpu().numpy(),
                # 'log_p_pre_real': log_p_pre_real.cpu().numpy(),
                'speaker_reward_list': speaker_reward_list.cpu().numpy(),
                'emotion_reward_list': emotion_reward_list.cpu().numpy(),
                'log_p_pre_syn':log_p_pre_syn.cpu().numpy(),
                'goal_syn_spec':goal_syn_spec.cpu().numpy(),
                'goal_syn_spec_len':goal_syn_spec_len.cpu().numpy(),
                'goal_syn_wav':goal_syn_wav.cpu().numpy(),
                'goal_syn_wav_len':goal_syn_wav_len.cpu().numpy(),
                'input_syn_spec':input_syn_spec.cpu().numpy(),
                'input_syn_spec_len':input_syn_spec_len.cpu().numpy(),
                'input_syn_wav':input_syn_wav.cpu().numpy(),
                'input_syn_wav_len':input_syn_wav_len.cpu().numpy(),
                'input_real_spec':input_real_spec.cpu().numpy(),
                'input_real_spec_len':input_real_spec_len.cpu().numpy(),
                'input_real_wav':input_real_wav.cpu().numpy(),
                'input_real_wav_len':input_real_wav_len.cpu().numpy(),
                'emo_e':emo_e.cpu().numpy(),
                'emo_real_e':emo_real_e.cpu().numpy(), 
                # 'l1_losses':l1_losses.cpu().numpy(),
                'y_hat':torch.tensor(y_hat_forward).cpu().numpy(),
                'kl':KL.cpu().numpy(),
                
            })
            
            # if global_step % hps.train.log_interval == 0:
            #     traindata=buffer.load_data()
            #     train_and_evaluate_RL(rank, epoch,hps, nets, optims, schedulers, scaler, loaders, logger, writers,traindata)


def initial_buffer_RL_old(
    rank, epoch,hps, nets, optims, schedulers, scaler, loaders, buffer
):
    with torch.no_grad():
        net_g, net_g_pre = nets
        optim_g, optim_d, optim_dur_disc = optims
        scheduler_g, scheduler_d, scheduler_dur_disc = schedulers
        train_loader, eval_loader = loaders
        train_loader.batch_sampler.set_epoch(epoch)
        global global_step

        # net_g.train()
        # net_g_pre.train()
        # if net_dur_disc is not None:
        #     net_dur_disc.train()

        for batch_idx, (
            emo_audio_embed_syn,
            emo_audio_path_list_syn,
            goal_audio_list_syn,
            input_audio_list_syn,
            emo_audio_embed_real,
            emo_audio_path_list_real,
            input_audio_list_real,
            goal_syn_spec,
            goal_syn_spec_len,
            goal_syn_wav,
            goal_syn_wav_len,
            input_syn_spec,
            input_syn_spec_len,
            input_syn_wav,
            input_syn_wav_len,
            input_real_spec,
            input_real_spec_len,
            input_real_wav,
            input_real_wav_len,
            emo_e,
            emo_real_e,        

        ) in enumerate(tqdm(train_loader)):
            print('enter initial_buffer_RL')

            
            goal_syn_spec, goal_syn_spec_len = goal_syn_spec.cuda(rank, non_blocking=True), goal_syn_spec_len.cuda(
            rank, non_blocking=True
        )
            goal_syn_wav, goal_syn_wav_len = goal_syn_wav.cuda(rank, non_blocking=True), goal_syn_wav_len.cuda(
            rank, non_blocking=True
        )
            input_syn_spec, input_syn_spec_len = input_syn_spec.cuda(rank, non_blocking=True), input_syn_spec_len.cuda(
            rank, non_blocking=True
        )
            input_syn_wav, input_syn_wav_len = input_syn_wav.cuda(rank, non_blocking=True), input_syn_wav_len.cuda(
            rank, non_blocking=True
        )
            input_real_spec, input_real_spec_len = input_real_spec.cuda(rank, non_blocking=True), input_real_spec_len.cuda(
            rank, non_blocking=True
        )
            input_real_wav, input_real_wav_len = input_real_wav.cuda(rank, non_blocking=True), input_real_wav_len.cuda(
            rank, non_blocking=True
        )
            emo_e, emo_real_e = emo_e.cuda(rank, non_blocking=True), emo_real_e.cuda(
            rank, non_blocking=True
        )


            if buffer.is_full():
                break
            if net_g.module.use_noise_scaled_mas:
                current_mas_noise_scale = (
                    net_g.module.mas_noise_scale_initial
                    - net_g.module.noise_scale_delta * global_step
                )
                net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)
            # emo_audio_embed=emo_audio_embed
            # emo_audio_path_list=emo_audio_path_list
            # goal_audio_list=goal_audio_list
            # input_audio_list=input_audio_list


            # log_p_real=[]
            # log_p_pre_real=[]
            # log_p_pre_syn=[]
            # emotion_reward_list=[]
            # speaker_reward_list=[]

            # l1_losses = []
            num_samples = len(emo_audio_embed_real)
            log_p_real = torch.empty(num_samples, device=rank)
            log_p_pre_real = torch.empty(num_samples, device=rank)
            log_p_pre_syn = torch.empty(num_samples, device=rank)
            emotion_reward_list = torch.empty(num_samples, device=rank)
            speaker_reward_list = torch.empty(num_samples, device=rank)
            # l1_losses = torch.empty(num_samples, device=rank)

            ###inference real audio
            # for i in range(len(emo_audio_embed_real)):
            for i in range(num_samples):
                emo_emb_path=emo_audio_embed_real[i]
                emo = torch.FloatTensor(np.load(emo_emb_path)).to(rank)

                (
                    y_hat,
                    mu, logs,y_mask,
                    (z,z_p, m_q, logs_q),
                    o_l1,ids_slice
                ) = net_g.module.infer_sts_RL(
                    input_audio_list_real[i],
                    emo,
                    hps.data.sampling_rate,
                    hps.data.filter_length,
                    hps.data.hop_length,
                    hps.data.win_length,

                )


                (
                    y_hat_pre,
                    mu_pre, logs_pre,y_mask_pre,
                    (z_pre,z_p_pre, m_q_pre, logs_q_pre),
                    o_l1_pre,ids_slice_pre
                ) = net_g_pre.module.infer_sts_RL(
                    input_audio_list_real[i],
                    emo,
                    hps.data.sampling_rate,
                    hps.data.filter_length,
                    hps.data.hop_length,
                    hps.data.win_length,
                )

                # log_p_real.append(compute_log_probability(y_hat,mu,logs).cpu().item())
                # log_p_pre_real.append(compute_log_probability(y_hat_pre,mu_pre,logs_pre).cpu().item())
                log_p_real[i] = compute_log_probability(y_hat, mu, logs).cpu().item()
                log_p_pre_real[i] = compute_log_probability(y_hat_pre, mu_pre, logs_pre).cpu().item()
                
                ###kl_mel
                print('enter kl mel')

                mel = spec_to_mel_torch(
                input_real_spec[i].unsqueeze(0).to(rank),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
                ).to(rank)
                print('mel',mel.device)


                y_mel = commons.slice_segments(
                    mel, ids_slice, hps.train.segment_size // hps.data.hop_length
                ).to(rank)

                y_hat_mel = mel_spectrogram_torch(
                    o_l1.squeeze(1).to(rank),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    # 2065,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                ).to(rank)
                print('y_mel',y_mel.device)
                print('y_hat_mel',y_hat_mel.device)
                # l1_losses[i] = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                # print('l1_losses',l1_losses)

                ###emotion reward
                wav, sr = librosa.load(emo_audio_path_list_real[i], 16000)

                y_hat_i=y_hat[0][0].cpu()

                wav_e = process_func(np.expand_dims(wav, 0), sr, embeddings=True)[0]

                y_hat_e = process_func(np.expand_dims(y_hat_i, 0), sr, embeddings=True)[0]

                cos_sim = dot(wav_e, y_hat_e)/(norm(wav_e)*norm(y_hat_e))

                # cos_sim=(cos_sim+1.0)/2

                # emotion_reward_list.append(cos_sim)
                emotion_reward_list[i] = cos_sim


                ###speaker reward
                pcm, sample_rate = torchaudio.load(input_audio_list_real[i],normalize=False)

                
                # wav_spk, sr = librosa.load(input_audio_list[i], 16000)
                # wav_spk=torch.from_numpy(wav_spk)
                pcm2=y_hat[0]


                pcm2=pcm2.to(rank)


                # speaker_reward_list.append(speaker_func(pcm,pcm2,sr))
                speaker_reward_list[i] = speaker_func(pcm, pcm2, sr)


            speaker_reward_list=torch.tensor(speaker_reward_list).to(rank)
            emotion_reward_list=torch.tensor(emotion_reward_list).to(rank)
            log_p_real=torch.tensor(log_p_real).to(rank)
            log_p_pre_real=torch.tensor(log_p_pre_real).to(rank)
            # l1_losses=torch.tensor(l1_losses).to(rank)





            ###inference syn audio
            for i in range(len(emo_audio_embed_real)):
                emo_emb_path=emo_audio_embed_syn[i]
                emo = torch.FloatTensor(np.load(emo_emb_path)).to(rank)
                (
                    y_hat_pre_syn,
                    mu_pre_syn, logs_pre_syn,y_mask_pre_syn,
                    (z_pre_syn,z_p_pre_syn, m_q_pre_syn, logs_q_pre_syn)
                ) = net_g_pre.module.infer_sts_RL(
                    input_audio_list_syn[i],
                    emo,
                    hps.data.sampling_rate,
                    hps.data.filter_length,
                    hps.data.hop_length,
                    hps.data.win_length,
                )
                log_p_pre_syn.append(compute_log_probability(y_hat_pre_syn,mu_pre_syn,logs_pre_syn).cpu().item())
            # log_p_pre_syn=torch.tensor(log_p_pre_syn).to(rank)
            log_p_pre_syn[i] = compute_log_probability(y_hat_pre_syn, mu_pre_syn, logs_pre_syn).cpu().item()

            print('speaker_reward_list',speaker_reward_list)
            print('emotion_reward_list',emotion_reward_list)




            buffer.push({
                'emo_audio_embed_syn': emo_audio_embed_syn,
                'emo_audio_path_list_syn': emo_audio_path_list_syn,
                'goal_audio_list_syn': goal_audio_list_syn,
                'input_audio_list_syn': input_audio_list_syn,
                'log_p_real': log_p_real.cpu().numpy(),
                'log_p_pre_real': log_p_pre_real.cpu().numpy(),
                'speaker_reward_list': speaker_reward_list.cpu().numpy(),
                'emotion_reward_list': emotion_reward_list.cpu().numpy(),
                'log_p_pre_syn':log_p_pre_syn.cpu().numpy(),
                'goal_syn_spec':goal_syn_spec.cpu().numpy(),
                'goal_syn_spec_len':goal_syn_spec_len.cpu().numpy(),
                'goal_syn_wav':goal_syn_wav.cpu().numpy(),
                'goal_syn_wav_len':goal_syn_wav_len.cpu().numpy(),
                'input_syn_spec':input_syn_spec.cpu().numpy(),
                'input_syn_spec_len':input_syn_spec_len.cpu().numpy(),
                'input_syn_wav':input_syn_wav.cpu().numpy(),
                'input_syn_wav_len':input_syn_wav_len.cpu().numpy(),
                'input_real_spec':input_real_spec.cpu().numpy(),
                'input_real_spec_len':input_real_spec_len.cpu().numpy(),
                'input_real_wav':input_real_wav.cpu().numpy(),
                'input_real_wav_len':input_real_wav_len.cpu().numpy(),
                'emo_e':emo_e.cpu().numpy(),
                'emo_real_e':emo_real_e.cpu().numpy(), 
                # 'l1_losses':l1_losses.cpu().numpy(),
            })
            
            # if global_step % hps.train.log_interval == 0:
            #     traindata=buffer.load_data()
            #     train_and_evaluate_RL(rank, epoch,hps, nets, optims, schedulers, scaler, loaders, logger, writers,traindata)
def train_and_evaluate_RL(
    rank, epoch,hps, nets, optims, schedulers, scaler, loaders, logger, writers,
):
    print('enter train_and_evaluate_RL')

    net_g, net_d, net_dur_disc = nets
    optim_g, optim_d, optim_dur_disc = optims
    scheduler_g, scheduler_d, scheduler_dur_disc = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers


    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    if net_dur_disc is not None:
        net_dur_disc.train()

    for batch_idx, (
        spec_goal,
        spec_lengths_goal,
        y,
        y_lengths,
        # speakers,
        spec_input,
        spec_input_lengths,
        y_input,
        y_input_lengths,
        emo,
    ) in enumerate(tqdm(train_loader)):
        print('enter_train_loader_RL')
        if net_g.module.use_noise_scaled_mas:
            current_mas_noise_scale = (
                net_g.module.mas_noise_scale_initial
                - net_g.module.noise_scale_delta * global_step
            )
            net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)

        spec_goal, spec_lengths_goal = spec_goal.cuda(rank, non_blocking=True), spec_lengths_goal.cuda(
            rank, non_blocking=True
        )
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(
            rank, non_blocking=True
        )
        spec_input, spec_input_lengths = spec_input.cuda(rank, non_blocking=True), spec_input_lengths.cuda(
            rank, non_blocking=True
        )

        emo = emo.cuda(rank, non_blocking=True)


        with autocast(enabled=hps.train.fp16_run):
            (
                y_hat,
                l_length,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                (hidden_x, logw, logw_),
                (z,z_p, m_q, logs_q)
            ) = net_g(
                spec_goal,
                spec_lengths_goal,
                spec_input,
                spec_input_lengths,
                # speakers,
                emo,
            )

            mel = spec_to_mel_torch(
                spec_goal,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )


            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )


            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            
            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            
            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc

            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                    hidden_x.detach(), x_mask.detach(), logw.detach(), logw_.detach()
                )
                with autocast(enabled=False):
                    # TODO: I think need to mean using the mask, but for now, just mean all
                    (
                        loss_dur_disc,
                        losses_dur_disc_r,
                        losses_dur_disc_g,
                    ) = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                    loss_dur_disc_all = loss_dur_disc
                optim_dur_disc.zero_grad()
                scaler.scale(loss_dur_disc_all).backward()
                scaler.unscale_(optim_dur_disc)
                commons.clip_grad_value_(net_dur_disc.parameters(), None)
                scaler.step(optim_dur_disc)

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw, logw_)
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                # print('loss_mel.shape',loss_mel)
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
                # loss_gen_all = loss_gen + loss_fm + loss_mel 

                if net_dur_disc is not None:
                    loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
                    loss_gen_all += loss_dur_gen
                print('loss_gen_all',loss_gen_all)
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

       
        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
                # losses = [loss_disc, loss_gen, loss_fm, loss_mel]

                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/dur": loss_dur,
                        "loss/g/kl": loss_kl,
                    }
                )
                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                )
                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                )

                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()
                    ),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()
                    ),
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        mel[0].data.cpu().numpy()
                    ),
                    "all/attn": utils.plot_alignment_to_numpy(
                        attn[0, 0].data.cpu().numpy()
                    ),
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )

            if global_step % hps.train.eval_interval == 0:
                print(f'saving checkpoint to {hps.model_dir}')
                evaluate(hps, net_g, eval_loader, writer_eval)
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
                )
                utils.save_checkpoint(
                    net_d,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
                )
                if net_dur_disc is not None:
                    utils.save_checkpoint(
                        net_dur_disc,
                        optim_dur_disc,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, "DUR_{}.pth".format(global_step)),
                    )

                keep_ckpts = getattr(hps.train, "keep_ckpts", 5)
                if keep_ckpts > 0:
                    utils.clean_checkpoints(
                        path_to_models=hps.model_dir,
                        n_ckpts_to_keep=keep_ckpts,
                        sort_by_time=True,
                    )

        global_step += 1
    print('train_and_eval finish')
   
    if rank == 0:
        logger.info("====> Epoch: {}".format(epoch))
    torch.cuda.empty_cache()

def train_and_evaluate_RL_old(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers
):
   
    net_g, net_g_pre, net_d, net_dur_disc = nets
    optim_g, optim_d, optim_dur_disc = optims
    scheduler_g, scheduler_d, scheduler_dur_disc = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers


    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    if net_dur_disc is not None:
        net_dur_disc.train()


    for batch_idx, (
        spec_goal,
        spec_lengths_goal,
        y,#goal
        y_lengths,#goal
        spec_input,
        spec_input_lengths,
        y_input,
        y_input_lengths,
        emo,
        emo_audio,
        goal_audio,
        input_audio,
    ) in enumerate(tqdm(train_loader)):
        print('enter_train_loader')
        if net_g.module.use_noise_scaled_mas:
            current_mas_noise_scale = (
                net_g.module.mas_noise_scale_initial
                - net_g.module.noise_scale_delta * global_step
            )
            net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)

        spec_goal, spec_lengths_goal = spec_goal.cuda(rank, non_blocking=True), spec_lengths_goal.cuda(
            rank, non_blocking=True
        )
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(
            rank, non_blocking=True
        )
        spec_input, spec_input_lengths = spec_input.cuda(rank, non_blocking=True), spec_input_lengths.cuda(
            rank, non_blocking=True
        )

        emo = emo.cuda(rank, non_blocking=True)
        # emo_audio = emo_audio.cuda(rank, non_blocking=True)
        # goal_audio = goal_audio.cuda(rank, non_blocking=True)
        # input_audio = input_audio.cuda(rank, non_blocking=True)







        with autocast(enabled=hps.train.fp16_run):
            (
                y_hat,
                mu, logs,y_mask,
                (z,z_p, m_q, logs_q)
            ) = net_g.module.infer_sts_RL(
                spec_input,
                spec_input_lengths,
                y_input,
                y_input_lengths,
                # speakers,
                emo,
                hps.data.sampling_rate,
                hps.data.filter_length,
                hps.data.hop_length,
                hps.data.win_length,
            )

            (
                y_hat_pre,
                mu_pre, logs_pre,y_mask_pre,
                (z_pre,z_p_pre, m_q_pre, logs_q_pre)
            ) = net_g_pre.module.infer_sts_RL(
                spec_input,
                spec_input_lengths,
                y_input,
                y_input_lengths,
                # speakers,
                emo,
                hps.data.sampling_rate,
                hps.data.filter_length,
                hps.data.hop_length,
                hps.data.win_length,
            )
            
            log_p=compute_log_probability(y_hat,mu,logs)
            log_p_pre=compute_log_probability(y_hat_pre,mu_pre,logs_pre)
            # print('log_p',log_p)
            # print('log_p_pre',log_p_pre)

            ###emotion reward
            emotion_list=[]
            for i in range(y_hat.shape[0]):
                wav, sr = librosa.load(emo_audio[i], 16000)
                y_hat_i=y_hat[i][0].cpu()
                wav_e = process_func(np.expand_dims(wav, 0), sr, embeddings=True)[0]
                y_hat_e = process_func(np.expand_dims(y_hat_i, 0), sr, embeddings=True)[0]
                
                cos_sim = dot(wav_e, y_hat_e)/(norm(wav_e)*norm(y_hat_e))
                emotion_list.append(cos_sim)
            ##turn the emo list into a tensor with the same shape of log_p
            shape = log_p.shape
            device=log_p.device
            emotion_tensor=torch.tensor(emotion_list,dtype=torch.float32)
            emotion_tensor=emotion_tensor.view(shape)
            emotion_tensor=emotion_tensor.to(device)

            # print('y_input',y_input.shape)
            # print('y_hat',y_hat.shape)
            speaker_smi=speaker_func(y_input,y_hat)
            # print('cos_sim',cos_sim)
            # print('speaker_smi',speaker_smi)

            RolloutBuffer.save(

                    log_p=log_p.cpu().numpy(),
                    log_p_pre=log_p_pre.cpu().numpy(),
                    cos_sim=cos_sim.cpu.numpy(),
                    speaker_smi=speaker_smi.cpu.numpy(),
                    spec_goal=spec_goal.cpu.numpy(),
                    spec_lengths_goal=spec_lengths_goal.cpu.numpy(),
                    y_goal=y.cpu.numpy(),#goal
                    y_goal_lengths=y_lengths.cpu.numpy(),#goal
                    spec_input=spec_input.cpu.numpy(),
                    spec_input_lengths=spec_input_lengths.cpu().numpy(),
                    y_input=y_input.cpu().numpy(),
                    y_input_lengths=y_input_lengths.cpu().numpy(),
                    emo=emo.cpu.numpy(),
                )

def train_and_evaluate_original(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers
):
    net_g, net_d, net_dur_disc = nets
    optim_g, optim_d, optim_dur_disc = optims
    scheduler_g, scheduler_d, scheduler_dur_disc = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers
    # print('train_loader_train&eval',len(train_loader))
    # print('eval_loader_train&eval',len(eval_loader))

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    if net_dur_disc is not None:
        net_dur_disc.train()
    # print('train_loader',len(train_loader))
    for batch_idx, (
        spec_goal,
        spec_lengths_goal,
        y,
        y_lengths,
        # speakers,
        spec_input,
        spec_input_lengths,
        y_input,
        y_input_lengths,
        emo,
    ) in enumerate(tqdm(train_loader)):
        print('enter_train_loader')
        if net_g.module.use_noise_scaled_mas:
            current_mas_noise_scale = (
                net_g.module.mas_noise_scale_initial
                - net_g.module.noise_scale_delta * global_step
            )
            net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)

        spec_goal, spec_lengths_goal = spec_goal.cuda(rank, non_blocking=True), spec_lengths_goal.cuda(
            rank, non_blocking=True
        )
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(
            rank, non_blocking=True
        )
        spec_input, spec_input_lengths = spec_input.cuda(rank, non_blocking=True), spec_input_lengths.cuda(
            rank, non_blocking=True
        )

        emo = emo.cuda(rank, non_blocking=True)

        with autocast(enabled=hps.train.fp16_run):
            (
                y_hat,
                l_length,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                (hidden_x, logw, logw_),
                (z,z_p, m_q, logs_q)
            ) = net_g(
                spec_goal,
                spec_lengths_goal,
                spec_input,
                spec_input_lengths,
                # speakers,
                emo,
            )

            mel = spec_to_mel_torch(
                spec_goal,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )

            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            
            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice
            
            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                    hidden_x.detach(), x_mask.detach(), logw.detach(), logw_.detach()
                )
                with autocast(enabled=False):
                    # TODO: I think need to mean using the mask, but for now, just mean all
                    (
                        loss_dur_disc,
                        losses_dur_disc_r,
                        losses_dur_disc_g,
                    ) = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                    loss_dur_disc_all = loss_dur_disc
                optim_dur_disc.zero_grad()
                scaler.scale(loss_dur_disc_all).backward()
                scaler.unscale_(optim_dur_disc)
                commons.clip_grad_value_(net_dur_disc.parameters(), None)
                scaler.step(optim_dur_disc)

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw, logw_)
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
                # loss_gen_all = loss_gen + loss_fm + loss_mel 

                if net_dur_disc is not None:
                    loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
                    loss_gen_all += loss_dur_gen
                print('loss_gen_all',loss_gen_all)
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()
       
        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
                # losses = [loss_disc, loss_gen, loss_fm, loss_mel]

                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/dur": loss_dur,
                        "loss/g/kl": loss_kl,
                    }
                )
                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                )
                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                )

                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()
                    ),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()
                    ),
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        mel[0].data.cpu().numpy()
                    ),
                    "all/attn": utils.plot_alignment_to_numpy(
                        attn[0, 0].data.cpu().numpy()
                    ),
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )

            if global_step % hps.train.eval_interval == 0:
                print(f'saving checkpoint to {hps.model_dir}')
                evaluate(hps, net_g, eval_loader, writer_eval)
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
                )
                utils.save_checkpoint(
                    net_d,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
                )
                if net_dur_disc is not None:
                    utils.save_checkpoint(
                        net_dur_disc,
                        optim_dur_disc,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, "DUR_{}.pth".format(global_step)),
                    )

                keep_ckpts = getattr(hps.train, "keep_ckpts", 5)
                if keep_ckpts > 0:
                    utils.clean_checkpoints(
                        path_to_models=hps.model_dir,
                        n_ckpts_to_keep=keep_ckpts,
                        sort_by_time=True,
                    )

        global_step += 1
    print('train_and_eval finish')
   
    if rank == 0:
        logger.info("====> Epoch: {}".format(epoch))
    torch.cuda.empty_cache()


def train_and_evaluate(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers
):
    # print('enter train_and_eval111')
    net_g, net_d, net_dur_disc = nets
    optim_g, optim_d, optim_dur_disc = optims
    scheduler_g, scheduler_d, scheduler_dur_disc= schedulers
    train_loader = loaders


    if writers is not None:

        writer, writer_eval = writers

    
    global global_step

    net_g.train()
    net_d.train()
    if net_dur_disc is not None:
        net_dur_disc.train()

    for batch_idx, data in enumerate(tqdm(train_loader)):
        print('enter train and eval train loader')

        speaker_reward_list = torch.tensor(data['speaker_reward_list']).cuda(rank, non_blocking=True)
        emotion_reward_list = torch.tensor(data['emotion_reward_list']).cuda(rank, non_blocking=True)
        log_p_pre_syn = torch.tensor(data['log_p_pre_syn'], requires_grad=True).cuda(rank, non_blocking=True)
        goal_syn_spec = torch.tensor(data['goal_syn_spec']).cuda(rank, non_blocking=True)
        goal_syn_spec_len = torch.tensor(data['goal_syn_spec_len']).cuda(rank, non_blocking=True)
        goal_syn_wav = torch.tensor(data['goal_syn_wav']).cuda(rank, non_blocking=True)
        goal_syn_wav_len = torch.tensor(data['goal_syn_wav_len']).cuda(rank, non_blocking=True)
        input_syn_spec = torch.tensor(data['input_syn_spec']).cuda(rank, non_blocking=True)
        input_syn_spec_len = torch.tensor(data['input_syn_spec_len']).cuda(rank, non_blocking=True)
        input_syn_wav = torch.tensor(data['input_syn_wav']).cuda(rank, non_blocking=True)
        input_syn_wav_len = torch.tensor(data['input_syn_wav_len']).cuda(rank, non_blocking=True)
        input_real_spec = torch.tensor(data['input_real_spec']).cuda(rank, non_blocking=True)
        input_real_spec_len = torch.tensor(data['input_real_spec_len']).cuda(rank, non_blocking=True)
        input_real_wav = torch.tensor(data['input_real_wav']).cuda(rank, non_blocking=True)
        input_real_wav_len = torch.tensor(data['input_real_wav_len']).cuda(rank, non_blocking=True)
        emo_e = torch.tensor(data['emo_e']).cuda(rank, non_blocking=True)
        emo_real_e = torch.tensor(data['emo_real_e']).cuda(rank, non_blocking=True)
        y_hat_reward=torch.tensor(data['y_hat']).cuda(rank, non_blocking=True)
        KL_=torch.tensor(data['kl']).cuda(rank, non_blocking=True)
        spec_emo_real_padded=torch.tensor(data['spec_emo_real_padded']).cuda(rank, non_blocking=True)
        spec_emo_real_lengths=torch.tensor(data['spec_emo_real_lengths']).cuda(rank, non_blocking=True)
        wav_emo_real_padded=torch.tensor(data['wav_emo_real_padded']).cuda(rank, non_blocking=True)
        wav_emo_real_lengths=torch.tensor(data['wav_emo_real_lengths']).cuda(rank, non_blocking=True)
        speaker_reward_list_syn = torch.tensor(data['speaker_reward_list_syn']).cuda(rank, non_blocking=True)
        emotion_reward_list_syn = torch.tensor(data['emotion_reward_list_syn']).cuda(rank, non_blocking=True)
        emotion_loss = torch.tensor(data['emotion_loss']).cuda(rank, non_blocking=True)
        emotion_loss_syn = torch.tensor(data['emotion_loss_syn']).cuda(rank, non_blocking=True)
        spec_realgoal_padded = torch.tensor(data['spec_realgoal_padded']).cuda(rank, non_blocking=True)
        spec_realgoal_lengths = torch.tensor(data['spec_realgoal_lengths']).cuda(rank, non_blocking=True)
        wav_realgoal_padded = torch.tensor(data['wav_realgoal_padded']).cuda(rank, non_blocking=True)
        wav_realgoal_lengths = torch.tensor(data['wav_realgoal_lengths']).cuda(rank, non_blocking=True)
        # print('spec in train_and_Eval',input_syn_spec.shape)
        # print('spec_lengths in train_and_Eval',input_syn_spec_len.shape)
        # print('spec in train_and_Eval',goal_syn_spec.shape)
        # print('spec_lengths in train_and_Eval',goal_syn_spec_len.shape)
        # print('emo_audio_embed_real in train_and_Eval',emo_real_e.shape)
        # print('y_hat_reward in train_and_Eval',y_hat_reward.shape)
        # print('KL_ in train_and_Eval',KL_.shape)
        # print('speaker_reward_list',speaker_reward_list)
        # print('emotion_reward_list',emotion_reward_list)
        # print('input_real_wav',input_real_wav.shape)

        if net_g.module.use_noise_scaled_mas:
            current_mas_noise_scale = (
                net_g.module.mas_noise_scale_initial
                - net_g.module.noise_scale_delta * global_step
            )
            net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)
        with autocast(enabled=hps.train.fp16_run):
            print('enter train and eval train loader1629')
            (
                y_hat,
                l_length,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                (hidden_x, logw, logw_),
                (z,z_p, m_q, logs_q),mu,logs
            ) = net_g(
                spec_realgoal_padded,
                spec_realgoal_lengths,
                input_real_spec,
                input_real_spec_len,
                # speakers,
                emo_real_e,
            )
            print('outside of SynthesizerTrn')
            # print('y_hat_reward',y_hat_reward.shape)
            # print('input_real_spec',input_real_spec)
            # print('input_real_spec_len',input_real_spec_len)

            # print('mu',mu)
            # print('logs',logs)
            # print('y_hat',y_hat)


            log_p_reward = compute_log_probability(y_hat_reward, mu, logs)
            log_p_reward_yhat = compute_log_probability(y_hat, mu, logs)
            # print('logs_p',logs_p)


            # log_p_reward = torch.cat(log_p_reward, dim=0)
            log_p_reward = log_p_reward.view(-1)
            log_p_reward_yhat = log_p_reward_yhat.view(-1)

            
            mel = spec_to_mel_torch(
                spec_realgoal_padded,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )

            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            
            y = commons.slice_segments(
                wav_realgoal_padded, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            
            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                    hidden_x.detach(), x_mask.detach(), logw.detach(), logw_.detach()
                )
                with autocast(enabled=False):
                    # TODO: I think need to mean using the mask, but for now, just mean all
                    (
                        loss_dur_disc,
                        losses_dur_disc_r,
                        losses_dur_disc_g,
                    ) = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                    loss_dur_disc_all = loss_dur_disc
                optim_dur_disc.zero_grad()
                scaler.scale(loss_dur_disc_all).backward()
                scaler.unscale_(optim_dur_disc)
                commons.clip_grad_value_(net_dur_disc.parameters(), None)
                scaler.step(optim_dur_disc)

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)


        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw, logw_)
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                # print('y_mel',y_mel.shape)
                # print('y_hat_mel',y_hat_mel.shape)

                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
                # loss_gen_all = loss_gen + loss_mel + loss_dur + loss_kl

                if net_dur_disc is not None:
                    loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
                    loss_gen_all += loss_dur_gen
                print('emotion_reward_list',emotion_reward_list)
                print('loss_gen_all',loss_gen_all)
                print('log_p_reward_yhat',log_p_reward_yhat)
                print('speaker_reward_list',speaker_reward_list)

                reward=1*emotion_reward_list -(0.02*loss_gen_all-1)
                # reward=0*speaker_reward_list+1*emotion_reward_list -1*loss_gen_all

                # reward=-emotion_loss-loss_gen_all

                # total_loss=loss_gen_all
                base_reward=loss_gen_all
                log_p_pre_syn = log_p_pre_syn.squeeze()
                print('log_p_pre_syn',log_p_pre_syn)
                print('KL_',KL_)


                total_loss=-(log_p_reward_yhat*reward + (0*KL_)) -0*log_p_pre_syn 
                total_loss_=loss_gen_all
                # print('total_loss',total_loss)
                total_loss_reduced = total_loss.mean(dim=0)
                # print('total_loss_reduced',total_loss_reduced)
                loss_gen_all=torch.mean(total_loss_reduced)
                # print('loss_gen_all',loss_gen_all)

                print('outside of loss_gen_all')
                
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()


        # print('log_p_real',log_p_reward)
        ##July 22
        # print('reward',reward)
        # print('speaker_reward_list',speaker_reward_list)
        # print('emotion_reward_list',emotion_reward_list)

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                # losses = [total_loss]
                # losses = [loss_disc, loss_gen, loss_fm, loss_mel]
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                # logger.info([x.item() for x in losses] + [global_step, lr])
                # logger.info([total_loss] + [global_step, lr])
                ##July 22
                logger.info(f'total_loss = {loss_gen_all} + at global step {global_step}, leanring rate is {lr}')
                ##July 22
                log_tensor_values(reward.tolist(), 'reward',logger,global_step, lr)


                log_tensor_values(speaker_reward_list.tolist(), 'speaker_reward_list',logger,global_step, lr)


                log_tensor_values(emotion_reward_list.tolist(), 'emotion_reward_list',logger,global_step, lr)

                # log_tensor_values(l1_losses.tolist(), 'l1_losses',logger,global_step, lr)

                log_tensor_values(log_p_reward_yhat.tolist(), 'log_p_reward_yhat',logger,global_step, lr)

                log_tensor_values(KL_.tolist(), 'KL_loss',logger,global_step, lr)

                log_tensor_values(log_p_pre_syn.tolist(), 'log_p_pre_syn',logger,global_step, lr)

                ##July 22
                print('reward_for_tensorboard',reward.shape)
                print('reward_for_tensorboard',reward)

                reward_for_tensorboard=torch.mean(reward,dim=0).item()
                print('reward_for_tensorboard',reward_for_tensorboard)
                emotion_reward_list=torch.mean(emotion_reward_list,dim=0).item()
                print('emotion_loss',emotion_loss)

                # speaker_reward_list=torch.mean(speaker_reward_list,dim=0).item()
                # emotion_reward_list=torch.mean(emotion_reward_list,dim=0).item()
                # l1_losses=torch.mean(l1_losses,dim=0).item()
                KL_loss=torch.mean(KL_,dim=0).item()
                print('reward_for_tensorboard2365')

                scalar_dict = {
                    # "loss/g/total": loss_gen_all,
                    # "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    # "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                    "loss": loss_gen_all,
                    ##July 22
                    "total_reward": reward_for_tensorboard,
                    'emotion_loss':emotion_reward_list,
                    'base_reward':base_reward,
                    # "speaker_reward": speaker_reward_list,
                    # "emotion_reward": emotion_reward_list,
                    # "l1_losses": l1_losses,
                    "KL(cur,pre)": KL_loss,
                }
                print('reward_for_tensorboard2381')


                # scalar_dict.update(
                #     {
                #         "loss/g/fm": loss_fm,
                #         "loss/g/mel": loss_mel,
                #         "loss/g/dur": loss_dur,
                #         "loss/g/kl": loss_kl,
                #     }
                # )

                ##July 22
                print('total_loss',total_loss)
                scalar_dict.update(
                    # {"loss/g/{}".format(total_loss)}
                    {"loss/g/{}".format(i): v.item() for i, v in enumerate(total_loss)}
                )

                print('reward_for_tensorboard2399')


                # scalar_dict.update(
                #     {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                # )
                # scalar_dict.update(
                #     {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                # )

                # image_dict = {
                #     "slice/mel_org": utils.plot_spectrogram_to_numpy(
                #         y_mel[0].data.cpu().numpy()
                #     ),
                #     "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                #         y_hat_mel[0].data.cpu().numpy()
                #     ),
                #     "all/mel": utils.plot_spectrogram_to_numpy(
                #         mel[0].data.cpu().numpy()
                #     ),
                #     "all/attn": utils.plot_alignment_to_numpy(
                #         attn[0, 0].data.cpu().numpy()
                #     ),
                # }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    # images=image_dict,
                    scalars=scalar_dict,
                )
                print('line1268')

            if global_step % hps.train.eval_interval == 0:
                # evaluate(hps, net_g, eval_loader, writer_eval)
                # train_and_evaluate_syn(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers)
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
                )
                # utils.save_checkpoint(
                #     net_g,
                #     optim_g,
                #     hps.train.learning_rate,
                #     epoch,
                #     os.path.join(hps.pretrained_RL_dir, "G_{}.pth".format(global_step)),
                # )
                utils.save_checkpoint(
                    net_d,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
                )
                if net_dur_disc is not None:
                    utils.save_checkpoint(
                        net_dur_disc,
                        optim_dur_disc,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, "DUR_{}.pth".format(global_step)),
                    )

                keep_ckpts = getattr(hps.train, "keep_ckpts", 5)
                if keep_ckpts > 0:
                    utils.clean_checkpoints(
                        path_to_models=hps.model_dir,
                        n_ckpts_to_keep=keep_ckpts,
                        sort_by_time=True,
                    )
                    # utils.clean_checkpoints(
                    #     path_to_models=hps.pretrained_RL_dir,
                    #     n_ckpts_to_keep=keep_ckpts,
                    #     sort_by_time=True,
                    # )
        # print('line1303 global_step',global_step)
        global_step += 1
        # print('line1305 global_step',global_step)

    print('train_and_eval finish')
   
    if rank == 0:
        logger.info("====> Epoch: {}".format(epoch))
    torch.cuda.empty_cache()

def train_and_evaluate_syn(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers
):
    # print('enter train_and_eval111')
    net_g, net_d, net_dur_disc = nets
    optim_g, optim_d, optim_dur_disc = optims
    scheduler_g, scheduler_d, scheduler_dur_disc= schedulers
    train_loader = loaders


    if writers is not None:

        writer, writer_eval = writers

    
    global global_step

    net_g.train()
    net_d.train()
    if net_dur_disc is not None:
        net_dur_disc.train()

    for batch_idx, data in enumerate(tqdm(train_loader)):
        print('enter train and eval train loader')

        speaker_reward_list = torch.tensor(data['speaker_reward_list']).cuda(rank, non_blocking=True)
        emotion_reward_list = torch.tensor(data['emotion_reward_list']).cuda(rank, non_blocking=True)
        log_p_pre_syn = torch.tensor(data['log_p_pre_syn'], requires_grad=True).cuda(rank, non_blocking=True)
        goal_syn_spec = torch.tensor(data['goal_syn_spec']).cuda(rank, non_blocking=True)
        goal_syn_spec_len = torch.tensor(data['goal_syn_spec_len']).cuda(rank, non_blocking=True)
        goal_syn_wav = torch.tensor(data['goal_syn_wav']).cuda(rank, non_blocking=True)
        goal_syn_wav_len = torch.tensor(data['goal_syn_wav_len']).cuda(rank, non_blocking=True)
        input_syn_spec = torch.tensor(data['input_syn_spec']).cuda(rank, non_blocking=True)
        input_syn_spec_len = torch.tensor(data['input_syn_spec_len']).cuda(rank, non_blocking=True)
        input_syn_wav = torch.tensor(data['input_syn_wav']).cuda(rank, non_blocking=True)
        input_syn_wav_len = torch.tensor(data['input_syn_wav_len']).cuda(rank, non_blocking=True)
        input_real_spec = torch.tensor(data['input_real_spec']).cuda(rank, non_blocking=True)
        input_real_spec_len = torch.tensor(data['input_real_spec_len']).cuda(rank, non_blocking=True)
        input_real_wav = torch.tensor(data['input_real_wav']).cuda(rank, non_blocking=True)
        input_real_wav_len = torch.tensor(data['input_real_wav_len']).cuda(rank, non_blocking=True)
        emo_e = torch.tensor(data['emo_e']).cuda(rank, non_blocking=True)
        emo_real_e = torch.tensor(data['emo_real_e']).cuda(rank, non_blocking=True)
        y_hat_reward=torch.tensor(data['y_hat']).cuda(rank, non_blocking=True)
        KL_=torch.tensor(data['kl']).cuda(rank, non_blocking=True)
        spec_emo_real_padded=torch.tensor(data['spec_emo_real_padded']).cuda(rank, non_blocking=True)
        spec_emo_real_lengths=torch.tensor(data['spec_emo_real_lengths']).cuda(rank, non_blocking=True)
        wav_emo_real_padded=torch.tensor(data['wav_emo_real_padded']).cuda(rank, non_blocking=True)
        wav_emo_real_lengths=torch.tensor(data['wav_emo_real_lengths']).cuda(rank, non_blocking=True)
        speaker_reward_list_syn = torch.tensor(data['speaker_reward_list_syn']).cuda(rank, non_blocking=True)
        emotion_reward_list_syn = torch.tensor(data['emotion_reward_list_syn']).cuda(rank, non_blocking=True)
        spec_realgoal_padded = torch.tensor(data['spec_realgoal_padded']).cuda(rank, non_blocking=True)
        spec_realgoal_lengths = torch.tensor(data['spec_realgoal_lengths']).cuda(rank, non_blocking=True)
        wav_realgoal_padded = torch.tensor(data['wav_realgoal_padded']).cuda(rank, non_blocking=True)
        wav_realgoal_lengths = torch.tensor(data['wav_realgoal_lengths']).cuda(rank, non_blocking=True)
        emotion_loss = torch.tensor(data['emotion_loss']).cuda(rank, non_blocking=True)
        emotion_loss_syn = torch.tensor(data['emotion_loss_syn']).cuda(rank, non_blocking=True)

        # print('spec in train_and_Eval',input_syn_spec.shape)
        
        if net_g.module.use_noise_scaled_mas:
            current_mas_noise_scale = (
                net_g.module.mas_noise_scale_initial
                - net_g.module.noise_scale_delta * global_step
            )
            net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)
        with autocast(enabled=hps.train.fp16_run):
            (
                y_hat,
                l_length,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                (hidden_x, logw, logw_),
                (z,z_p, m_q, logs_q),mu,logs
            ) = net_g(
                goal_syn_spec,
                goal_syn_spec_len,
                input_syn_spec,
                input_syn_spec_len,
                # speakers,
                emo_e,
            )


            log_p_reward = compute_log_probability(y_hat_reward, mu, logs)
            log_p_reward_yhat = compute_log_probability(y_hat, mu, logs)
            # print('logs_p',logs_p)


            # log_p_reward = torch.cat(log_p_reward, dim=0)
            log_p_reward = log_p_reward.view(-1)
            log_p_reward_yhat = log_p_reward_yhat.view(-1)

            
            mel = spec_to_mel_torch(
                goal_syn_spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )


            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            
            y = commons.slice_segments(
                goal_syn_wav, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            
            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                    hidden_x.detach(), x_mask.detach(), logw.detach(), logw_.detach()
                )
                with autocast(enabled=False):
                    # TODO: I think need to mean using the mask, but for now, just mean all
                    (
                        loss_dur_disc,
                        losses_dur_disc_r,
                        losses_dur_disc_g,
                    ) = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                    loss_dur_disc_all = loss_dur_disc
                optim_dur_disc.zero_grad()
                scaler.scale(loss_dur_disc_all).backward()
                scaler.unscale_(optim_dur_disc)
                commons.clip_grad_value_(net_dur_disc.parameters(), None)
                scaler.step(optim_dur_disc)

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)


        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw, logw_)
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())

                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
                if net_dur_disc is not None:
                    loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
                    loss_gen_all += loss_dur_gen
                reward=-0.1*emotion_loss_syn -1*loss_gen_all
                log_p_pre_syn = log_p_pre_syn.squeeze()
                total_loss=-(log_p_reward_yhat*reward + (0*KL_)) -0*log_p_pre_syn 
                total_loss_=loss_gen_all
                total_loss_reduced = total_loss.mean(dim=0)
                loss_gen_all=torch.mean(total_loss_reduced)
                
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()


        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                # losses = [total_loss]
                # losses = [loss_disc, loss_gen, loss_fm, loss_mel]
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                # logger.info([x.item() for x in losses] + [global_step, lr])
                # logger.info([total_loss] + [global_step, lr])
                ##July 22
                logger.info(f'total_loss in syn = {loss_gen_all} + at global step {global_step}, leanring rate is {lr}')
                ##July 22
                log_tensor_values(reward.tolist(), 'reward',logger,global_step, lr)


                log_tensor_values(speaker_reward_list_syn.tolist(), 'speaker_reward_list in syn',logger,global_step, lr)


                log_tensor_values(emotion_reward_list_syn.tolist(), 'emotion_reward_list in syn',logger,global_step, lr)

                # log_tensor_values(l1_losses.tolist(), 'l1_losses',logger,global_step, lr)

                log_tensor_values(log_p_reward_yhat.tolist(), 'log_p_reward in syn',logger,global_step, lr)

                log_tensor_values(KL_.tolist(), 'KL_loss in syn',logger,global_step, lr)

                log_tensor_values(log_p_pre_syn.tolist(), 'log_p_pre_syn in syn',logger,global_step, lr)

                ##July 22
                # print('reward_for_tensorboard',reward.shape)
                reward_for_tensorboard=torch.mean(reward,dim=0).item()
                # print('reward_for_tensorboard',reward_for_tensorboard)

                speaker_reward_list=torch.mean(speaker_reward_list,dim=0).item()
                emotion_reward_list=torch.mean(emotion_reward_list,dim=0).item()
                # l1_losses=torch.mean(l1_losses,dim=0).item()
                KL_loss=torch.mean(KL_,dim=0).item()
                # print('reward_for_tensorboard2365')

                scalar_dict = {
                    # "loss/g/total": loss_gen_all,
                    # "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    # "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                    "loss": loss_gen_all,
                    ##July 22
                    "total_reward": reward_for_tensorboard,
                    "speaker_reward": speaker_reward_list,
                    "emotion_reward": emotion_reward_list,
                    # "l1_losses": l1_losses,
                    "KL(cur,pre)": KL_loss,
                }
                # print('reward_for_tensorboard2381')


                # scalar_dict.update(
                #     {
                #         "loss/g/fm": loss_fm,
                #         "loss/g/mel": loss_mel,
                #         "loss/g/dur": loss_dur,
                #         "loss/g/kl": loss_kl,
                #     }
                # )

                ##July 22
                # print('total_loss',total_loss)
                scalar_dict.update(
                    # {"loss/g/{}".format(total_loss)}
                    {"loss/g/{}".format(i): v.item() for i, v in enumerate(total_loss)}
                )

                # print('reward_for_tensorboard2399')


                # scalar_dict.update(
                #     {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                # )
                # scalar_dict.update(
                #     {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                # )

                # image_dict = {
                #     "slice/mel_org": utils.plot_spectrogram_to_numpy(
                #         y_mel[0].data.cpu().numpy()
                #     ),
                #     "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                #         y_hat_mel[0].data.cpu().numpy()
                #     ),
                #     "all/mel": utils.plot_spectrogram_to_numpy(
                #         mel[0].data.cpu().numpy()
                #     ),
                #     "all/attn": utils.plot_alignment_to_numpy(
                #         attn[0, 0].data.cpu().numpy()
                #     ),
                # }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    # images=image_dict,
                    scalars=scalar_dict,
                )
                # print('line1268')

            if global_step % hps.train.eval_interval == 0:
                # evaluate(hps, net_g, eval_loader, writer_eval)
                # train_and_evaluate_syn(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers)
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
                )
                # utils.save_checkpoint(
                #     net_g,
                #     optim_g,
                #     hps.train.learning_rate,
                #     epoch,
                #     os.path.join(hps.pretrained_RL_dir, "G_{}.pth".format(global_step)),
                # )
                utils.save_checkpoint(
                    net_d,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
                )
                if net_dur_disc is not None:
                    utils.save_checkpoint(
                        net_dur_disc,
                        optim_dur_disc,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, "DUR_{}.pth".format(global_step)),
                    )

                keep_ckpts = getattr(hps.train, "keep_ckpts", 5)
                if keep_ckpts > 0:
                    utils.clean_checkpoints(
                        path_to_models=hps.model_dir,
                        n_ckpts_to_keep=keep_ckpts,
                        sort_by_time=True,
                    )
                    # utils.clean_checkpoints(
                    #     path_to_models=hps.pretrained_RL_dir,
                    #     n_ckpts_to_keep=keep_ckpts,
                    #     sort_by_time=True,
                    # )
        # print('line1303 global_step',global_step)
        global_step += 1
        # print('line1305 global_step',global_step)

    print('train_and_eval finish')
   
    if rank == 0:
        logger.info("====> Epoch: {}".format(epoch))
    torch.cuda.empty_cache()



def train_and_evaluate07222024(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers
):
    print('enter train_and_eval111')
    net_g, net_d, net_dur_disc = nets
    # net_g= nets[0]
    # optim_g= optims[0]
    optim_g, optim_d, optim_dur_disc = optims
    scheduler_g, scheduler_d, scheduler_dur_disc= schedulers

    # train_loader, eval_loader = loaders
    train_loader = loaders


    if writers is not None:

        writer, writer_eval = writers

    
    global global_step

    net_g.train()
    net_d.train()
    if net_dur_disc is not None:
        net_dur_disc.train()
    ## Freeze net_d and net_dur_disc
    # net_d.eval()
    # for param in net_d.parameters():
    #     param.requires_grad = False
    # if net_dur_disc is not None:
    #     net_dur_disc.eval()
    #     for param in net_dur_disc.parameters():
    #         param.requires_grad = False

    # for name, param in net_g.named_parameters():
    #     if not param.requires_grad:
    #         print(f"Parameter {name} does not require gradient!")


    for batch_idx, data in enumerate(tqdm(train_loader)):
        print('enter train and eval train loader')

        # if data_list is None:
        #     print(f"Warning: data at batch {batch_idx} is None")
        #     continue
        # # for data in data_list:

        # emo_audio_embed_syn=data['emo_audio_embed_syn']
        # emo_audio_path_list_syn=data['emo_audio_path_list_syn']
        # goal_audio_list_syn=data['goal_audio_list_syn']
        # input_audio_list_syn=data['input_audio_list_syn']
        # log_p_real = torch.tensor(data['log_p_real'], requires_grad=True).cuda(rank, non_blocking=True)
        # log_p_pre_real = torch.tensor(data['log_p_pre_real'], requires_grad=True).cuda(rank, non_blocking=True)
        speaker_reward_list = torch.tensor(data['speaker_reward_list']).cuda(rank, non_blocking=True)
        emotion_reward_list = torch.tensor(data['emotion_reward_list']).cuda(rank, non_blocking=True)
        log_p_pre_syn = torch.tensor(data['log_p_pre_syn'], requires_grad=True).cuda(rank, non_blocking=True)
        # # log_p_pre_syn = log_p_pre_syn.view(-1)
        # goal_syn_spec = torch.tensor(data['goal_syn_spec']).cuda(rank, non_blocking=True)
        # goal_syn_spec_len = torch.tensor(data['goal_syn_spec_len']).cuda(rank, non_blocking=True)
        # goal_syn_wav = torch.tensor(data['goal_syn_wav']).cuda(rank, non_blocking=True)
        # goal_syn_wav_len = torch.tensor(data['goal_syn_wav_len']).cuda(rank, non_blocking=True)
        # input_syn_spec = torch.tensor(data['input_syn_spec']).cuda(rank, non_blocking=True)
        # input_syn_spec_len = torch.tensor(data['input_syn_spec_len']).cuda(rank, non_blocking=True)
        # input_syn_wav = torch.tensor(data['input_syn_wav']).cuda(rank, non_blocking=True)
        # input_syn_wav_len = torch.tensor(data['input_syn_wav_len']).cuda(rank, non_blocking=True)
        # input_real_spec = torch.tensor(data['input_real_spec']).cuda(rank, non_blocking=True)
        # input_real_spec_len = torch.tensor(data['input_real_spec_len']).cuda(rank, non_blocking=True)
        input_real_wav = torch.tensor(data['input_real_wav']).cuda(rank, non_blocking=True)
        # input_real_wav_len = torch.tensor(data['input_real_wav_len']).cuda(rank, non_blocking=True)
        # emo_e = torch.tensor(data['emo_e']).cuda(rank, non_blocking=True)
        # emo_real_e = torch.tensor(data['emo_real_e']).cuda(rank, non_blocking=True)
        # l1_losses=torch.tensor(data['l1_losses']).cuda(rank, non_blocking=True)
        spec=torch.tensor(data['spec']).cuda(rank, non_blocking=True)
        spec = spec.squeeze(1)
        spec_lengths=torch.tensor(data['spec_lengths']).cuda(rank, non_blocking=True)
        spec_lengths = spec_lengths.view(-1)
        emo_audio_embed_real=torch.tensor(data['emo_audio_embed_real']).cuda(rank, non_blocking=True)
        y_hat_reward=torch.tensor(data['y_hat']).cuda(rank, non_blocking=True)
        KL_=torch.tensor(data['kl']).cuda(rank, non_blocking=True)

        print('spec in train_and_Eval',spec.shape)
        print('spec_lengths in train_and_Eval',spec_lengths)
        print('emo_audio_embed_real in train_and_Eval',emo_audio_embed_real)
        print('y_hat_reward in train_and_Eval',y_hat_reward.shape)
        print('KL_ in train_and_Eval',KL_.shape)









        

        # print('emo_audio_embed_syn',emo_audio_embed_syn)
        # print('emo_audio_path_list_syn',emo_audio_path_list_syn)
        # print('goal_audio_list_syn',goal_audio_list_syn)
        # print('input_audio_list_syn',input_audio_list_syn)
        # print('log_p_real',log_p_real)
        # print('log_p_pre_real',log_p_pre_real)
        print('speaker_reward_list',speaker_reward_list)
        print('emotion_reward_list',emotion_reward_list)
        # print('log_p_pre_syn',log_p_pre_syn)
        # print('goal_syn_spec',goal_syn_spec)
        # print('goal_syn_spec_len',goal_syn_spec_len)
        # print('goal_syn_wav',goal_syn_wav)
        # print('goal_syn_wav_len',goal_syn_wav_len)
        # print('input_syn_spec',input_syn_spec)
        # print('input_syn_spec_len',input_syn_spec_len)
        # print('input_syn_wav',input_syn_wav)
        # print('input_syn_wav_len',input_syn_wav_len)
        # print('input_real_spec',input_real_spec)
        # print('input_real_spec_len',input_real_spec_len)
        # print('input_real_wav',input_real_wav)
        # print('input_real_wav_len',input_real_wav_len)
        # print('emo_e',emo_e)
        # print('emo_real_e',emo_real_e)




        if net_g.module.use_noise_scaled_mas:
            current_mas_noise_scale = (
                net_g.module.mas_noise_scale_initial
                - net_g.module.noise_scale_delta * global_step
            )
            net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)
    

        
        
        with autocast(enabled=hps.train.fp16_run):
            print('enter train and eval train loader1629')
            (
                y_hat,
                l_length,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                (hidden_x, logw, logw_),
                (z,z_p, m_q, logs_q),mu,logs
            ) = net_g(
                spec,
                spec_lengths,
                spec,
                spec_lengths,
                # speakers,
                emo_audio_embed_real,
            )
            print('outside of SynthesizerTrn')

            log_p_reward = compute_log_probability(y_hat_reward, mu, logs)
            print('outside of compute_log_probability',log_p_reward)
            print('outside of compute_log_probability',log_p_reward.shape)
            print('emotion_reward_list',emotion_reward_list.shape)

            # log_p_reward = torch.cat(log_p_reward, dim=0)
            log_p_reward = log_p_reward.view(-1)
            
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            print('spec',spec.shape)
            print('mel',mel.shape)

            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )


            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            
            y = commons.slice_segments(
                input_real_wav, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            
            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                    hidden_x.detach(), x_mask.detach(), logw.detach(), logw_.detach()
                )
                with autocast(enabled=False):
                    # TODO: I think need to mean using the mask, but for now, just mean all
                    (
                        loss_dur_disc,
                        losses_dur_disc_r,
                        losses_dur_disc_g,
                    ) = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                    loss_dur_disc_all = loss_dur_disc
                optim_dur_disc.zero_grad()
                scaler.scale(loss_dur_disc_all).backward()
                scaler.unscale_(optim_dur_disc)
                commons.clip_grad_value_(net_dur_disc.parameters(), None)
                scaler.step(optim_dur_disc)

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)


        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw, logw_)
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                print('y_mel',y_mel.shape)
                print('y_hat_mel',y_hat_mel.shape)

                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                # print('trainline1727')
                # print('log_p_reward',log_p_reward)
                # print('KL_',KL_)
                # print('KL_',KL_.shape)
                # print('reward',reward)
                # print('reward',reward.shape)
                # print('log_p_pre_syn',log_p_pre_syn)
                # print('log_p_pre_syn',log_p_pre_syn.shape)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
                reward=0*speaker_reward_list+0.01*emotion_reward_list -loss_gen_all
                total_loss=-(log_p_reward*reward + (0*KL_)) -0*log_p_pre_syn 


                loss_gen_all=torch.mean(total_loss)
                # loss_gen_all = loss_gen + loss_fm + loss_mel 


                # if net_dur_disc is not None:
                #     loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
                #     loss_gen_all += loss_dur_gen
                print('loss_gen_all',loss_gen_all)
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()
       

        # KL_loss=log_p_real-log_p_pre_real

        # reward=0*speaker_reward_list+1*emotion_reward_list-0.01*l1_losses

        # total_loss=-(log_p_real*reward + (0*KL_)) -0*log_p_pre_syn 
        # total_loss=-(log_p_real*reward) -log_p_pre_syn 

        print('log_p_real',log_p_reward)
        ##July 22
        print('reward',reward)
        print('speaker_reward_list',speaker_reward_list)
        print('emotion_reward_list',emotion_reward_list)

        ##July 22
        # if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
        #     print("Loss is NaN or Inf")
        # else:
        #     print(f"Total loss: {total_loss}")

        # total_losses=[torch.mean(loss.unsqueeze(0)) for loss in total_loss]
        # print('total_losses',total_losses)
        

        # stacked_total_losses = torch.stack(total_losses)
        # print('stacked_total_losses',stacked_total_losses)
        # total_loss = torch.mean(stacked_total_losses, dim=0)
        # print('total_loss',total_loss)

        # optim_g.zero_grad()
        # print('Before backward:')
        # for name, param in net_g.named_parameters():
        #     if param.grad is None:
        #         print(f"{name} has no gradient before backward")

        # scaler.scale(total_loss).backward()
        # print('After backward:')
        # for name, param in net_g.named_parameters():
        #     if param.grad is None:
        #         print(f"{name} has no gradient after backward")
        #     else:
        #         print(f"{name}: {param.grad}")
        # scaler.unscale_(optim_g)
        # grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        # scaler.step(optim_g)
        # scaler.update()

        # for param in net_d.parameters():
        #     print('dis param',param)
        #     print('dis param.grad',param.grad)
        #     param.requires_grad = False
        #     print('dis param.grad',param.grad)
        # for param in net_dur_disc.parameters():
        #     print('dur param',param)
        #     print('dur param.grad',param.grad)
        #     param.requires_grad = False
        #     print('dur param.grad',param.grad)
        # for param in net_g.parameters():
        #     print('gene param',param)
        #     print('gene param.grad',param.grad)



        


        if rank == 0:

            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                # losses = [total_loss]
                # losses = [loss_disc, loss_gen, loss_fm, loss_mel]
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                # logger.info([x.item() for x in losses] + [global_step, lr])
                # logger.info([total_loss] + [global_step, lr])
                ##July 22
                logger.info(f'total_loss = {total_loss} + at global step {global_step}, leanring rate is {lr}')
                ##July 22
                log_tensor_values(reward.tolist(), 'reward',logger,global_step, lr)


                log_tensor_values(speaker_reward_list.tolist(), 'speaker_reward_list',logger,global_step, lr)


                log_tensor_values(emotion_reward_list.tolist(), 'emotion_reward_list',logger,global_step, lr)

                # log_tensor_values(l1_losses.tolist(), 'l1_losses',logger,global_step, lr)

                log_tensor_values(log_p_reward.tolist(), 'log_p_reward',logger,global_step, lr)

                log_tensor_values(KL_.tolist(), 'KL_loss',logger,global_step, lr)

                log_tensor_values(log_p_pre_syn.tolist(), 'log_p_pre_syn',logger,global_step, lr)

                ##July 22
                reward_for_tensorboard=torch.mean(reward,dim=0).item()
                speaker_reward_list=torch.mean(speaker_reward_list,dim=0).item()
                emotion_reward_list=torch.mean(emotion_reward_list,dim=0).item()
                # l1_losses=torch.mean(l1_losses,dim=0).item()
                KL_loss=torch.mean(KL_,dim=0).item()


                scalar_dict = {
                    # "loss/g/total": loss_gen_all,
                    # "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    # "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                    "loss": loss_gen_all,
                    ##July 22
                    # "total_reward": reward_for_tensorboard,
                    "speaker_reward": speaker_reward_list,
                    "emotion_reward": emotion_reward_list,
                    # "l1_losses": l1_losses,
                    "KL(cur,pre)": KL_loss,
                }


                # scalar_dict.update(
                #     {
                #         "loss/g/fm": loss_fm,
                #         "loss/g/mel": loss_mel,
                #         "loss/g/dur": loss_dur,
                #         "loss/g/kl": loss_kl,
                #     }
                # )

                ##July 22
                # scalar_dict.update(
                #     # {"loss/g/{}".format(total_loss)}
                #     {"loss/g/{}".format(i): v.item() for i, v in enumerate(total_loss)}
                # )



                # scalar_dict.update(
                #     {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                # )
                # scalar_dict.update(
                #     {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                # )

                # image_dict = {
                #     "slice/mel_org": utils.plot_spectrogram_to_numpy(
                #         y_mel[0].data.cpu().numpy()
                #     ),
                #     "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                #         y_hat_mel[0].data.cpu().numpy()
                #     ),
                #     "all/mel": utils.plot_spectrogram_to_numpy(
                #         mel[0].data.cpu().numpy()
                #     ),
                #     "all/attn": utils.plot_alignment_to_numpy(
                #         attn[0, 0].data.cpu().numpy()
                #     ),
                # }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    # images=image_dict,
                    scalars=scalar_dict,
                )
                print('line1268')

            if global_step % hps.train.eval_interval == 0:
                net_d.train()
                for param in net_d.parameters():
                    param.requires_grad = True
                if net_dur_disc is not None:
                    net_dur_disc.train()
                    for param in net_dur_disc.parameters():
                        param.requires_grad = True
                print(f'saving checkpoint to {hps.model_dir}')
                # evaluate(hps, net_g, eval_loader, writer_eval)
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
                )
                # utils.save_checkpoint(
                #     net_g,
                #     optim_g,
                #     hps.train.learning_rate,
                #     epoch,
                #     os.path.join(hps.pretrained_RL_dir, "G_{}.pth".format(global_step)),
                # )
                utils.save_checkpoint(
                    net_d,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
                )
                if net_dur_disc is not None:
                    utils.save_checkpoint(
                        net_dur_disc,
                        optim_dur_disc,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, "DUR_{}.pth".format(global_step)),
                    )

                keep_ckpts = getattr(hps.train, "keep_ckpts", 5)
                if keep_ckpts > 0:
                    utils.clean_checkpoints(
                        path_to_models=hps.model_dir,
                        n_ckpts_to_keep=keep_ckpts,
                        sort_by_time=True,
                    )
                    # utils.clean_checkpoints(
                    #     path_to_models=hps.pretrained_RL_dir,
                    #     n_ckpts_to_keep=keep_ckpts,
                    #     sort_by_time=True,
                    # )
                # net_d.eval()
                # for param in net_d.parameters():
                #     param.requires_grad = False
                # if net_dur_disc is not None:
                #     net_dur_disc.eval()
                #     for param in net_dur_disc.parameters():
                #         param.requires_grad = False
        print('line1303 global_step',global_step)
        global_step += 1
        print('line1305 global_step',global_step)

    print('train_and_eval finish')
   
    if rank == 0:
        logger.info("====> Epoch: {}".format(epoch))
    torch.cuda.empty_cache()
def log_tensor_values(tensor, tensor_name,logger,global_step, lr):
    for i, value in enumerate(tensor):
        print('log_tensor_values value',value)
        print('log_tensor_values global_step',global_step)
        print('log_tensor_values lr',lr)

        logger.info(f'{tensor_name}[{i}] = {value} + at global step {global_step}, leanring rate is {lr}')           

"""        with autocast(enabled=hps.train.fp16_run):
            (
                y_hat,
                l_length,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                (hidden_x, logw, logw_),
                (z,z_p, m_q, logs_q)
            ) = net_g(
                spec_goal,
                spec_lengths_goal,
                spec_input,
                spec_input_lengths,
                # speakers,
                emo,
            )

            mel = spec_to_mel_torch(
                spec_goal,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            # print('mel.shape',mel.shape)

            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            # print('y_mel.shape',mel.shape)

            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            # print('y_hat_mel.shape',y_hat_mel.shape)
            
            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice
            # print(y.shape,'y.shape')
            # print(y_hat.shape,'y_hat.shape')
            
            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc
                # print('loss_disc_all',loss_disc_all)
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                    hidden_x.detach(), x_mask.detach(), logw.detach(), logw_.detach()
                )
                with autocast(enabled=False):
                    # TODO: I think need to mean using the mask, but for now, just mean all
                    (
                        loss_dur_disc,
                        losses_dur_disc_r,
                        losses_dur_disc_g,
                    ) = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                    loss_dur_disc_all = loss_dur_disc
                optim_dur_disc.zero_grad()
                scaler.scale(loss_dur_disc_all).backward()
                scaler.unscale_(optim_dur_disc)
                commons.clip_grad_value_(net_dur_disc.parameters(), None)
                scaler.step(optim_dur_disc)

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw, logw_)
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                # print('loss_mel.shape',loss_mel)
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
                # loss_gen_all = loss_gen + loss_fm + loss_mel 

                if net_dur_disc is not None:
                    loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
                    loss_gen_all += loss_dur_gen
                print('loss_gen_all',loss_gen_all)
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

       
        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
                # losses = [loss_disc, loss_gen, loss_fm, loss_mel]

                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/dur": loss_dur,
                        "loss/g/kl": loss_kl,
                    }
                )
                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                )
                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                )

                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()
                    ),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()
                    ),
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        mel[0].data.cpu().numpy()
                    ),
                    "all/attn": utils.plot_alignment_to_numpy(
                        attn[0, 0].data.cpu().numpy()
                    ),
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )

            # if global_step % hps.train.eval_interval == 0:
            #     print(f'saving checkpoint to {hps.model_dir}')
            #     evaluate(hps, net_g, eval_loader, writer_eval)
            #     utils.save_checkpoint(
            #         net_g,
            #         optim_g,
            #         hps.train.learning_rate,
            #         epoch,
            #         os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
            #     )
            #     utils.save_checkpoint(
            #         net_d,
            #         optim_d,
            #         hps.train.learning_rate,
            #         epoch,
            #         os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
            #     )
            #     if net_dur_disc is not None:
            #         utils.save_checkpoint(
            #             net_dur_disc,
            #             optim_dur_disc,
            #             hps.train.learning_rate,
            #             epoch,
            #             os.path.join(hps.model_dir, "DUR_{}.pth".format(global_step)),
            #         )

            #     keep_ckpts = getattr(hps.train, "keep_ckpts", 5)
            #     if keep_ckpts > 0:
            #         utils.clean_checkpoints(
            #             path_to_models=hps.model_dir,
            #             n_ckpts_to_keep=keep_ckpts,
            #             sort_by_time=True,
            #         )

        global_step += 1
    print('train_and_eval finish')
   
    if rank == 0:
        logger.info("====> Epoch: {}".format(epoch))
    torch.cuda.empty_cache()

"""
def train_and_evaluate_old(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers
):
    print('enter train_and_eval')
    
    net_g, net_d, net_dur_disc = nets
    #print('line1005')
    optim_g, optim_d, optim_dur_disc = optims
    #print('line1007')
    scheduler_g, scheduler_d, scheduler_dur_disc = schedulers
    #print('line1009')
    train_loader, eval_loader = loaders
    #print('line1011')
    if writers is not None:
        #print('line1013')
        writer, writer_eval = writers
    #print('line1015')
    # #print('train_loader_train&eval',len(train_loader))
    # #print('eval_loader_train&eval',len(eval_loader))
    train_loader.batch_sampler.set_epoch(epoch)
    #print('line1019')

    global global_step

    net_g.train()
    net_d.train()
    if net_dur_disc is not None:
        net_dur_disc.train()
    # #print('train_loader',len(train_loader))
    for batch_idx, (
        spec_goal,
        spec_lengths_goal,
        y,
        y_lengths,
        # speakers,
        spec_input,
        spec_input_lengths,
        y_input,
        y_input_lengths,
        emo,
    ) in enumerate(tqdm(train_loader)):
        #print('enter_train_eval_loader')
        if net_g.module.use_noise_scaled_mas:
            current_mas_noise_scale = (
                net_g.module.mas_noise_scale_initial
                - net_g.module.noise_scale_delta * global_step
            )
            net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)
        #print('line1047')

        spec_goal, spec_lengths_goal = spec_goal.cuda(rank, non_blocking=True), spec_lengths_goal.cuda(
            rank, non_blocking=True
        )
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(
            rank, non_blocking=True
        )
        spec_input, spec_input_lengths = spec_input.cuda(rank, non_blocking=True), spec_input_lengths.cuda(
            rank, non_blocking=True
        )

        emo = emo.cuda(rank, non_blocking=True)





        with autocast(enabled=hps.train.fp16_run):
            (
                y_hat,
                l_length,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                (hidden_x, logw, logw_),
                (z,z_p, m_q, logs_q)
            ) = net_g(
                spec_goal,
                spec_lengths_goal,
                spec_input,
                spec_input_lengths,
                # speakers,
                emo,
            )

            mel = spec_to_mel_torch(
                spec_goal,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            # #print('mel.shape',mel.shape)

            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            # #print('y_mel.shape',mel.shape)

            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            # print('y_hat_mel.shape',y_hat_mel.shape)
            
            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice
            # print(y.shape,'y.shape')
            # print(y_hat.shape,'y_hat.shape')
            
            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc
                # print('loss_disc_all',loss_disc_all)
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                    hidden_x.detach(), x_mask.detach(), logw.detach(), logw_.detach()
                )
                with autocast(enabled=False):
                    # TODO: I think need to mean using the mask, but for now, just mean all
                    (
                        loss_dur_disc,
                        losses_dur_disc_r,
                        losses_dur_disc_g,
                    ) = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                    loss_dur_disc_all = loss_dur_disc
                optim_dur_disc.zero_grad()
                scaler.scale(loss_dur_disc_all).backward()
                scaler.unscale_(optim_dur_disc)
                commons.clip_grad_value_(net_dur_disc.parameters(), None)
                scaler.step(optim_dur_disc)

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw, logw_)
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                # print('loss_mel.shape',loss_mel)
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
                # loss_gen_all = loss_gen + loss_fm + loss_mel 

                if net_dur_disc is not None:
                    loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
                    loss_gen_all += loss_dur_gen
                print('loss_gen_all',loss_gen_all)
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

       
        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
                # losses = [loss_disc, loss_gen, loss_fm, loss_mel]

                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/dur": loss_dur,
                        "loss/g/kl": loss_kl,
                    }
                )
                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                )
                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                )

                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()
                    ),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()
                    ),
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        mel[0].data.cpu().numpy()
                    ),
                    "all/attn": utils.plot_alignment_to_numpy(
                        attn[0, 0].data.cpu().numpy()
                    ),
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )

            if global_step % hps.train.eval_interval == 0:
                print(f'saving checkpoint to {hps.model_dir}')
                evaluate(hps, net_g, eval_loader, writer_eval)
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
                )
                utils.save_checkpoint(
                    net_d,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
                )
                if net_dur_disc is not None:
                    utils.save_checkpoint(
                        net_dur_disc,
                        optim_dur_disc,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, "DUR_{}.pth".format(global_step)),
                    )

                keep_ckpts = getattr(hps.train, "keep_ckpts", 5)
                if keep_ckpts > 0:
                    utils.clean_checkpoints(
                        path_to_models=hps.model_dir,
                        n_ckpts_to_keep=keep_ckpts,
                        sort_by_time=True,
                    )

        global_step += 1
    print('train_and_eval finish')
   
    if rank == 0:
        logger.info("====> Epoch: {}".format(epoch))
    torch.cuda.empty_cache()

#
# def train_and_evaluate_old(
#     rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers
# ):
#     net_g, net_d, net_dur_disc = nets
#     optim_g, optim_d, optim_dur_disc = optims
#     scheduler_g, scheduler_d, scheduler_dur_disc = schedulers
#     train_loader, eval_loader = loaders
#     if writers is not None:
#         writer, writer_eval = writers

#     train_loader.batch_sampler.set_epoch(epoch)
#     global global_step

#     net_g.train()
#     net_d.train()
#     if net_dur_disc is not None:
#         net_dur_disc.train()
        
#     for batch_idx, (
#         x,
#         x_lengths,
#         spec,
#         spec_lengths,
#         y,
#         y_lengths,
#         speakers,
#         tone,
#         language,
#         bert,
#         ja_bert,
#         emo,
#     ) in enumerate(tqdm(train_loader)):
#         if net_g.module.use_noise_scaled_mas:
#             current_mas_noise_scale = (
#                 net_g.module.mas_noise_scale_initial
#                 - net_g.module.noise_scale_delta * global_step
#             )
#             net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)
#         x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(
#             rank, non_blocking=True
#         )
#         spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(
#             rank, non_blocking=True
#         )
#         y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(
#             rank, non_blocking=True
#         )
#         speakers = speakers.cuda(rank, non_blocking=True)
#         tone = tone.cuda(rank, non_blocking=True)
#         language = language.cuda(rank, non_blocking=True)
#         bert = bert.cuda(rank, non_blocking=True)
#         ja_bert = ja_bert.cuda(rank, non_blocking=True)
#         emo = emo.cuda(rank, non_blocking=True)
        

#         with autocast(enabled=hps.train.fp16_run):
#             (
#                 y_hat,
#                 l_length,
#                 attn,
#                 ids_slice,
#                 x_mask,
#                 z_mask,
#                 (z, z_p, m_p, logs_p, m_q, logs_q),
#                 (hidden_x, logw, logw_),
#             ) = net_g(
#                 x,
#                 x_lengths,
#                 spec,
#                 spec_lengths,
#                 speakers,
#                 tone,
#                 language,
#                 bert,
#                 ja_bert,
#                 emo,
#             )
#             mel = spec_to_mel_torch(
#                 spec,
#                 hps.data.filter_length,
#                 hps.data.n_mel_channels,
#                 hps.data.sampling_rate,
#                 hps.data.mel_fmin,
#                 hps.data.mel_fmax,
#             )
#             y_mel = commons.slice_segments(
#                 mel, ids_slice, hps.train.segment_size // hps.data.hop_length
#             )
#             y_hat_mel = mel_spectrogram_torch(
#                 y_hat.squeeze(1),
#                 hps.data.filter_length,
#                 hps.data.n_mel_channels,
#                 hps.data.sampling_rate,
#                 hps.data.hop_length,
#                 hps.data.win_length,
#                 hps.data.mel_fmin,
#                 hps.data.mel_fmax,
#             )

#             y = commons.slice_segments(
#                 y, ids_slice * hps.data.hop_length, hps.train.segment_size
#             )  # slice

#             # Discriminator
#             y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
#             with autocast(enabled=False):
#                 loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
#                     y_d_hat_r, y_d_hat_g
#                 )
#                 loss_disc_all = loss_disc
#             if net_dur_disc is not None:
#                 y_dur_hat_r, y_dur_hat_g = net_dur_disc(
#                     hidden_x.detach(), x_mask.detach(), logw.detach(), logw_.detach()
#                 )
#                 with autocast(enabled=False):
#                     # TODO: I think need to mean using the mask, but for now, just mean all
#                     (
#                         loss_dur_disc,
#                         losses_dur_disc_r,
#                         losses_dur_disc_g,
#                     ) = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
#                     loss_dur_disc_all = loss_dur_disc
#                 optim_dur_disc.zero_grad()
#                 scaler.scale(loss_dur_disc_all).backward()
#                 scaler.unscale_(optim_dur_disc)
#                 commons.clip_grad_value_(net_dur_disc.parameters(), None)
#                 scaler.step(optim_dur_disc)

#         optim_d.zero_grad()
#         scaler.scale(loss_disc_all).backward()
#         scaler.unscale_(optim_d)
#         grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
#         scaler.step(optim_d)

#         with autocast(enabled=hps.train.fp16_run):
#             # Generator
#             y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
#             if net_dur_disc is not None:
#                 y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw, logw_)
#             with autocast(enabled=False):
#                 loss_dur = torch.sum(l_length.float())
#                 loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
#                 loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

#                 loss_fm = feature_loss(fmap_r, fmap_g)
#                 loss_gen, losses_gen = generator_loss(y_d_hat_g)
#                 loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
#                 if net_dur_disc is not None:
#                     loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
#                     loss_gen_all += loss_dur_gen
#         optim_g.zero_grad()
#         scaler.scale(loss_gen_all).backward()
#         scaler.unscale_(optim_g)
#         grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
#         scaler.step(optim_g)
#         scaler.update()

#         if rank == 0:
#             if global_step % hps.train.log_interval == 0:
#                 lr = optim_g.param_groups[0]["lr"]
#                 losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
#                 logger.info(
#                     "Train Epoch: {} [{:.0f}%]".format(
#                         epoch, 100.0 * batch_idx / len(train_loader)
#                     )
#                 )
#                 logger.info([x.item() for x in losses] + [global_step, lr])

#                 scalar_dict = {
#                     "loss/g/total": loss_gen_all,
#                     "loss/d/total": loss_disc_all,
#                     "learning_rate": lr,
#                     "grad_norm_d": grad_norm_d,
#                     "grad_norm_g": grad_norm_g,
#                 }
#                 scalar_dict.update(
#                     {
#                         "loss/g/fm": loss_fm,
#                         "loss/g/mel": loss_mel,
#                         "loss/g/dur": loss_dur,
#                         "loss/g/kl": loss_kl,
#                     }
#                 )
#                 scalar_dict.update(
#                     {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
#                 )
#                 scalar_dict.update(
#                     {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
#                 )
#                 scalar_dict.update(
#                     {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
#                 )

#                 image_dict = {
#                     "slice/mel_org": utils.plot_spectrogram_to_numpy(
#                         y_mel[0].data.cpu().numpy()
#                     ),
#                     "slice/mel_gen": utils.plot_spectrogram_to_numpy(
#                         y_hat_mel[0].data.cpu().numpy()
#                     ),
#                     "all/mel": utils.plot_spectrogram_to_numpy(
#                         mel[0].data.cpu().numpy()
#                     ),
#                     "all/attn": utils.plot_alignment_to_numpy(
#                         attn[0, 0].data.cpu().numpy()
#                     ),
#                 }
#                 utils.summarize(
#                     writer=writer,
#                     global_step=global_step,
#                     images=image_dict,
#                     scalars=scalar_dict,
#                 )

#             if global_step % hps.train.eval_interval == 0:
#                 evaluate(hps, net_g, eval_loader, writer_eval)
#                 utils.save_checkpoint(
#                     net_g,
#                     optim_g,
#                     hps.train.learning_rate,
#                     epoch,
#                     os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
#                 )
#                 utils.save_checkpoint(
#                     net_d,
#                     optim_d,
#                     hps.train.learning_rate,
#                     epoch,
#                     os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
#                 )
#                 if net_dur_disc is not None:
#                     utils.save_checkpoint(
#                         net_dur_disc,
#                         optim_dur_disc,
#                         hps.train.learning_rate,
#                         epoch,
#                         os.path.join(hps.model_dir, "DUR_{}.pth".format(global_step)),
#                     )
#                 keep_ckpts = getattr(hps.train, "keep_ckpts", 5)
#                 if keep_ckpts > 0:
#                     utils.clean_checkpoints(
#                         path_to_models=hps.model_dir,
#                         n_ckpts_to_keep=keep_ckpts,
#                         sort_by_time=True,
#                     )

#         global_step += 1

#     if rank == 0:
#         logger.info("====> Epoch: {}".format(epoch))
#     torch.cuda.empty_cache()



def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    image_dict = {}
    audio_dict = {}
    print("Evaluating ...")
    with torch.no_grad():
        # print('eval_loader',len(eval_loader))
        # print('eval_loader',eval_loader)

        # print('eval_loader')
        # for i in eval_loader:
        #     print('i_evalloader',i)
            # input()
        for batch_idx, (
        spec_goal,
        spec_lengths_goal,
        y,
        y_lengths,
        # speakers,
        spec_input,
        spec_input_lengths,
        y_input,
        y_input_lengths,
        emo,
        emo_audio,
        goal_audio,
        input_audio,
        ) in enumerate(tqdm(eval_loader)):
            # print('batch_idx',batch_idx)
            # input()
            # x, x_lengths = x.cuda(), x_lengths.cuda()
            spec_goal, spec_lengths_goal = spec_goal.cuda(), spec_lengths_goal.cuda()
            y, y_lengths = y.cuda(), y_lengths.cuda()
            # speakers = speakers.cuda()
            # bert = bert.cuda()
            # ja_bert = ja_bert.cuda()
            # tone = tone.cuda()
            spec_input,spec_input_lengths =spec_input.cuda(),spec_input_lengths.cuda()
            y_input,y_input_lengths =y_input.cuda(),y_input_lengths.cuda()
            # language = language.cuda()
            emo = emo.cuda()
            # print('evaluate line 824')

            for use_sdp in [True, False]:
                # print('evaluate line 827')
                y_hat, attn,mask, *_ = generator.module.infer(
                    spec_input,
                    spec_input_lengths,
                    # speakers,
                    emo,
                    max_len=1000,
                    sdp_ratio=0.0 if not use_sdp else 1.0,
                    y=spec_goal,
                )
                # print('evaluate915_y_hat',y_hat.shape)
                # print('evaluate916_mask',mask.shape)

                y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length
                # print('evaluate919_y_hat_lengths',y_hat_lengths.shape)

                mel = spec_to_mel_torch(
                    spec_goal,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                # print('evaluate929_mel',mel.shape)

                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1).float(),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                # print('evaluate941_mel',y_hat_mel.shape)

                image_dict.update(
                    {
                        f"gen/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                            y_hat_mel[0].cpu().numpy()
                        )
                    }
                )
                # print('y_hat_lengths[0]',y_hat_lengths.shape)
                # print('line950')
                audio_dict.update(
                    {
                        f"gen/audio_{batch_idx}_{use_sdp}": y_hat[
                            0, :, : y_hat_lengths[0]
                        ]
                    }
                )
                # print('line958')

                image_dict.update(
                    {
                        f"gt/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                            mel[0].cpu().numpy()
                        )
                    }
                )
                # print('line967')

                audio_dict.update({f"gt/audio_{batch_idx}": y[0, :, : y_lengths[0]]})
    # print('Evauate line965')

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )
    # print('Evauate line974')

    generator.train()
    print('Evauate done')
    torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
