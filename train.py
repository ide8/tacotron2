# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

# Default imports
import os
import time
import json
import shutil
import argparse
import importlib
import numpy as np
from datetime import datetime
from contextlib import contextmanager

# Torch
import torch
import torch.distributed as dist
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Distributed + AMP
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
amp.lists.functional_overrides.FP32_FUNCS.remove('softmax')
amp.lists.functional_overrides.FP16_FUNCS.append('softmax')

# Parse args
parser = argparse.ArgumentParser(description='PyTorch Tacotron 2 Training')
parser.add_argument('--exp', type=str, default=None, required=True, help='Name of an experiment for configs setting.')
parser.add_argument('--rank', default=0, type=int, help='Rank of the process, do not set! Done by multiproc module')
parser.add_argument('--world-size', default=1, type=int, help='Number of processes, do not set! Done by multiproc module')
args = parser.parse_args()

# Prepare config
shutil.copyfile(os.path.join('configs', 'experiments', args.exp + '.py'), os.path.join('configs', '__init__.py'))

# Reload Config
configs = importlib.import_module('configs')
configs = importlib.reload(configs)
Config = configs.Config

# Config dependent imports
from tacotron2.text import text_to_sequence
from router import models, loss_functions, data_functions


def reduce_tensor(tensor, num_gpus):
    """

    :param tensor:
    :param num_gpus:
    :return:
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= num_gpus
    return rt


def init_distributed(dist_backend, dist_url, world_size, rank, group_name):
    """

    :param dist_backend:
    :param dist_url:
    :param world_size:
    :param rank:
    :param group_name:
    :return:
    """
    assert torch.cuda.is_available(), 'Distributed mode requires CUDA.'
    print('Initializing Distributed')

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=dist_backend, init_method=dist_url,
        world_size=world_size, rank=rank, group_name=group_name)

    print('Done initializing distributed')


def restore_checkpoint(restore_path, model_name):
    """

    :param restore_path:
    :param model_name:
    :return:
    """
    checkpoint = torch.load(restore_path, map_location='cpu')
    start_epoch = checkpoint['epoch'] + 1

    print('Restoring from `{}` checkpoint'.format(restore_path))

    model_config = checkpoint['config']
    model = models.get_model(model_name, model_config, to_cuda=True)

    # Unwrap distributed
    model_dict = {}
    for key, value in checkpoint['state_dict'].items():
        new_key = key.replace('module.1.', '')
        new_key = new_key.replace('module.', '')
        model_dict[new_key] = value

    model.load_state_dict(model_dict)

    return model, model_config, checkpoint, start_epoch


def save_checkpoint(model, epoch, config, optimizer, filepath):
    """

    :param model:
    :param epoch:
    :param config:
    :param optimizer:
    :param filepath:
    :return:
    """
    print('Saving model and optimizer state at epoch {} to {}'.format(
        epoch, filepath))
    torch.save({'epoch': epoch,
                'config': config,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, filepath)


def save_sample(model_name, model_path, tacotron2_path, waveglow_path, phrases):
    """

    :param model_name:
    :param model_path:
    :param tacotron2_path:
    :param waveglow_path:
    :param phrases:
    :return:
    """
    if model_name == 'Tacotron2':
        assert waveglow_path is not None, 'WaveGlow checkpoint path is missing, could not generate sample'
        tacotron2_path = model_path

    elif model_name == 'WaveGlow':
        assert tacotron2_path is not None, 'Tacotron2 checkpoint path is missing, could not generate sample'
        waveglow_path = model_path

    else:
        raise NotImplementedError('Unknown model requested: {}'.format(model_name))

    t2, _, _, _ = restore_checkpoint(tacotron2_path, 'Tacotron2')
    wg, _, _, _ = restore_checkpoint(waveglow_path, 'WaveGlow')

    with evaluating(t2), evaluating(wg), torch.no_grad():
        ref_audios = phrases['ref_audios'] if Config.gst_use else (None,)

        for speaker_id in phrases['speaker_ids']:
            for text in phrases['texts']:
                for ref_audio in ref_audios:
                    inputs = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
                    inputs = torch.from_numpy(inputs).to(device='cuda', dtype=torch.int64)

                    s_id = torch.IntTensor([speaker_id]).cuda().long()

                    ref_mel = ref_audio

                    _, mel, _, _ = t2.infer(inputs, s_id, ref_mel=ref_mel)
                    audio = wg.infer(mel)

                    audio_numpy = audio[0].data.cpu().numpy()

                    yield speaker_id, audio_numpy, ref_mel


# adapted from: https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998/3
# Following snippet is licensed under MIT license


@contextmanager
def evaluating(model):
    """
    Temporarily switch to evaluation mode.

    :param model:
    :return:
    """
    istrain = model.training
    try:
        model.eval()
        yield model
    finally:
        if istrain:
            model.train()


def validate(model, criterion, valset, batch_size, world_size, collate_fn, distributed_run, batch_to_gpu):
    """
    Handles all the validation scoring and printing

    :param model:
    :param criterion:
    :param valset:
    :param batch_size:
    :param world_size:
    :param collate_fn:
    :param distributed_run:
    :param batch_to_gpu:
    :return:
    """
    with evaluating(model), torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, num_workers=1, shuffle=False, sampler=val_sampler,
                                batch_size=batch_size, pin_memory=False, collate_fn=collate_fn)
        val_loss = 0.0

        for i, batch in enumerate(val_loader):
            x, y, len_x = batch_to_gpu(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)

            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, world_size).item()
            else:
                reduced_val_loss = loss.item()

            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    return val_loss


def adjust_learning_rate(epoch, optimizer, learning_rate, anneal_steps, anneal_factor):
    """

    :param epoch:
    :param optimizer:
    :param learning_rate:
    :param anneal_steps:
    :param anneal_factor:
    :return:
    """
    p = 0
    if anneal_steps is not None:
        for i, a_step in enumerate(anneal_steps):
            if epoch >= int(a_step):
                p = p+1

    if anneal_factor == 0.3:
        lr = learning_rate*((0.1 ** (p//2))*(1.0 if p % 2 == 0 else 0.3))
    else:
        lr = learning_rate*(anneal_factor ** p)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    # Experiment dates
    str_date, str_time = datetime.now().strftime("%d-%m-%yT%H-%M-%S").split('T')

    # Directories paths
    main_directory = os.path.join(Config.output_directory, args.exp, str_date, str_time)
    tf_directory = os.path.join(main_directory, 'tf_events')
    checkpoint_directory = os.path.join(main_directory, 'checkpoints')
    print('Experiment path: `{}`'.format(main_directory))

    # Directories check
    if not os.path.exists(main_directory):
        os.makedirs(main_directory)

    if not os.path.exists(checkpoint_directory):
        os.makedirs(checkpoint_directory)

    if not os.path.exists(tf_directory):
        os.makedirs(tf_directory)

    # Experiment files set up
    tensorboard_writer = tensorboard.SummaryWriter(log_dir=tf_directory)
    shutil.copy2('configs/__init__.py', os.path.join(main_directory, 'config.py'))
    with open(os.path.join(main_directory, 'args.json'), 'w') as fl:
        json.dump(vars(args), fl, indent=4)

    # Enble cuda
    torch.backends.cudnn.enabled = Config.cudnn_enabled
    torch.backends.cudnn.benchmark = Config.cudnn_benchmark

    # Set model name & Init distributed
    model_name = Config.model_name
    distributed_run = args.world_size > 1

    if distributed_run:
        init_distributed(dist_backend=Config.dist_backend,
                         dist_url=Config.dist_url,
                         world_size=args.world_size,
                         rank=args.rank,
                         group_name=Config.group_name)

    # Restore training from checkpoint
    if Config.restore_from:
        model, model_config, checkpoint, start_epoch = restore_checkpoint(Config.restore_from, model_name)
    else:
        checkpoint, start_epoch = None, 0

        model_config = models.get_model_config(model_name)
        model = models.get_model(model_name, model_config, to_cuda=True)

    # Distributed run
    if not Config.amp_run and distributed_run:
        model = DDP(model)

    # Define Optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=Config.learning_rate,
                                 weight_decay=Config.weight_decay)

    # Restore optimizer state
    if checkpoint and 'optimizer_state_dict' in checkpoint:
        print('Restoring optimizer state')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # FP16 option
    if Config.amp_run:  # TODO: test if FP16 actually works
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        if distributed_run:
            model = DDP(model)

    # Set sigma for WaveGlow loss
    try:
        sigma = model_config['sigma']
    except KeyError:
        sigma = None

    # Set criterion
    criterion = loss_functions.get_loss_function(model_name, sigma)

    # Set amount of frames per decoder step
    try:  # TODO: make it working with n > 1
        n_frames_per_step = model_config['n_frames_per_step']
    except KeyError:
        n_frames_per_step = None


    # Set dataloaders
    collate_fn = data_functions.get_collate_function(model_name, n_frames_per_step)
    trainset = data_functions.get_data_loader(model_name=model_name, audiopaths_and_text=Config.training_files)
    train_sampler = DistributedSampler(trainset) if distributed_run else None
    train_loader = DataLoader(trainset,
                              num_workers=1,
                              shuffle=False,
                              sampler=train_sampler,
                              batch_size=Config.batch_size,
                              pin_memory=False,
                              drop_last=True,
                              collate_fn=collate_fn)
    valset = data_functions.get_data_loader(model_name=model_name, audiopaths_and_text=Config.validation_files)
    batch_to_gpu = data_functions.get_batch_to_gpu(model_name)

    # Iteration inside of the epoch
    iteration = 0

    # Set model into training mode
    model.train()

    # Training loop
    if start_epoch >= Config.epochs:
        print('Checkpoint epoch {} >= total epochs {}'.format(start_epoch, Config.epochs))
    else:
        for epoch in range(start_epoch, Config.epochs):

            epoch_start_time = time.time()

            # Used to calculate avg items/sec over epoch
            reduced_num_items_epoch = 0

            # Used to calculate avg loss over epoch
            train_epoch_avg_loss = 0.0
            train_epoch_avg_items_per_sec = 0.0
            num_iters = 0

            for i, batch in enumerate(train_loader):
                print('Batch: {}/{} epoch {}'.format(i, len(train_loader), epoch))

                iter_start_time = time.time()
                adjust_learning_rate(epoch, optimizer, learning_rate=Config.learning_rate, anneal_steps=Config.anneal_steps, anneal_factor=Config.anneal_factor)
                model.zero_grad()
                x, y, num_items = batch_to_gpu(batch)
                y_pred = model(x)
                loss = criterion(y_pred, y)

                if distributed_run:
                    reduced_loss = reduce_tensor(loss.data, args.world_size).item()
                    reduced_num_items = reduce_tensor(num_items.data, 1).item()
                else:
                    reduced_loss = loss.item()
                    reduced_num_items = num_items.item()

                if np.isnan(reduced_loss):
                    raise Exception('loss is NaN')

                train_epoch_avg_loss += reduced_loss
                num_iters += 1

                # Accumulate number of items processed in this epoch
                reduced_num_items_epoch += reduced_num_items

                if Config.amp_run:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()

                iteration += 1

                iter_stop_time = time.time()
                iter_time = iter_stop_time - iter_start_time
                items_per_sec = reduced_num_items/iter_time
                train_epoch_avg_items_per_sec += items_per_sec

            epoch_stop_time = time.time()

            epoch_time = epoch_stop_time - epoch_start_time
            train_epoch_items_per_sec = reduced_num_items_epoch / epoch_time
            train_epoch_avg_items_per_sec = train_epoch_avg_items_per_sec / num_iters if num_iters > 0 else 0.0
            train_epoch_avg_loss = train_epoch_avg_loss / num_iters if num_iters > 0 else 0.0
            epoch_val_loss = validate(model, criterion, valset, Config.batch_size, args.world_size,
                                      collate_fn, distributed_run, batch_to_gpu)

            tensorboard_writer.add_scalar(tag='train_stats/epoch_items_per_sec',
                                          scalar_value=train_epoch_items_per_sec,
                                          global_step=epoch)
            tensorboard_writer.add_scalar(tag='train_stats/epoch_avg_items_per_sec',
                                          scalar_value=train_epoch_avg_items_per_sec,
                                          global_step=epoch)
            tensorboard_writer.add_scalar(tag='train_stats/epoch_time',
                                          scalar_value=epoch_time,
                                          global_step=epoch)
            tensorboard_writer.add_scalar(tag='epoch_avg_loss/train',
                                          scalar_value=train_epoch_avg_loss,
                                          global_step=epoch)
            tensorboard_writer.add_scalar(tag='epoch_avg_loss/val',
                                          scalar_value=epoch_val_loss,
                                          global_step=epoch)



            if (epoch % Config.epochs_per_checkpoint == 0) and args.rank == 0:
                checkpoint_path = os.path.join(checkpoint_directory, 'checkpoint_{}'.format(epoch))
                save_checkpoint(model, epoch, model_config, optimizer, checkpoint_path)

                # Save test audio files to tensorboard
                for i, (speaker_id, sample, ref_mel) in enumerate(save_sample(model_name,
                                                        checkpoint_path,
                                                        Config.tacotron2_checkpoint,
                                                        Config.waveglow_checkpoint,
                                                        Config.phrases)):

                    if Config.gst_use:
                        tensorboard_writer.add_audio(tag='epoch_{}/ref:speaker_{}_sample_{}'.format(epoch, speaker_id, i),
                                                     snd_tensor=ref_mel,
                                                     sample_rate=Config.sampling_rate)

                    tensorboard_writer.add_audio(tag='epoch_{}/infer:speaker_{}_sample_{}'.format(epoch, speaker_id, i),
                                                 snd_tensor=sample,
                                                 sample_rate=Config.sampling_rate)
    tensorboard_writer.close()


if __name__ == '__main__':
    main()
