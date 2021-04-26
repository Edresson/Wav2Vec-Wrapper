# -*- coding: utf-8 -*-
# ToDo: Add support of multi-gpu training
import os
import re
import json
import random
import argparse
import torch
import pandas as pd

import torchaudio
import librosa
import numpy as np

from shutil import copyfile
from utils.dataset import Dataset, DataColletor
from utils.generic_utils import load_config, load_vocab
from utils.generic_utils import save_best_checkpoint, calculate_wer
from utils.tensorboard import TensorboardWriter

import transformers
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer

from torch.utils.data import DataLoader
from torch.optim import AdamW

transformers.logging.set_verbosity_info()


def evaluation(model, processor, devel_dataset, epoch, global_step, config, tensorboard, USE_CUDA):
    model.eval()
    steps = 0
    tot_loss = 0
    tot_wer = 0
    with torch.no_grad():
        for batch in devel_dataset:
            input_values, attention_mask, labels = batch['input_values'], batch['attention_mask'], batch['labels']
            if USE_CUDA:
                input_values = input_values.cuda(non_blocking=True)
                attention_mask = attention_mask.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            outputs = model(input_values, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            # compute metrics 
            pred_ids = np.argmax(outputs.logits.cpu().detach().numpy(), axis=-1)
            tot_wer += calculate_wer(pred_ids, labels.cpu().detach().numpy(), processor)
            tot_loss += loss.item()
            steps += 1

    # calculate avg of metrics
    avg_loss = tot_loss/steps
    avg_wer = tot_wer/steps

    # tensorboard log
    tensorboard.log_evaluation(avg_loss, avg_wer, global_step)

    print("\n\n --> EVAL PERFORMANCE\n")
    print("     | > : CTC Loss ({:.5f})\n".format(avg_loss))
    print("     | > :   WER    ({:.5f})\n".format(avg_wer))

    return avg_loss



def train(model, optimizer, lr_scheduler, scaler, train_dataset, gpu_audio_augmentation, epoch, global_step, config, tensorboard, USE_CUDA):
    model.train()
    batch_n_iter = int(len(train_dataset.dataset) / config.batch_size)
    step = 0
    for batch in train_dataset:
        input_values, attention_mask, labels = batch['input_values'], batch['attention_mask'], batch['labels']
        if USE_CUDA:
            input_values = input_values.cuda(non_blocking=True)
            attention_mask = attention_mask.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=config.mixed_precision):
            # apply noise data augmentation
            if gpu_audio_augmentation is not None:
                input_values = gpu_audio_augmentation(input_values.unsqueeze(1), sample_rate=config.sampling_rate).squeeze(1)
            
            outputs = model(input_values, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # check nan loss
        if torch.isnan(loss).any():
            raise RuntimeError(f"> Detected NaN loss at step {global_step}.")

        if config.mixed_precision:
            scaler.scale(loss).backward()
            # scaler.unscale_(optimizer)
            # update learning rate
            scaler.step(optimizer)
            scaler.update()
        else:
            # back propag
            loss.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        model.zero_grad()

        step += 1
        global_step += 1
        # console log
        if global_step % config.print_step == 0:
            print("  --> STEP: {}/{} -- GLOBAL_STEP: {}\n".format(step, batch_n_iter, global_step))
            print("     | > : CTC Loss ({:.5f})\n".format(loss.item()))
        # Tensorboard logs
        if global_step % config.plot_step == 0:
            tensorboard.log_training(loss.item(), global_step)

    return global_step

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, required=True,
                        help="json file with configurations")
    parser.add_argument('--checkpoint_path', type=str, default='facebook/wav2vec2-large-xlsr-53',
                        help="path of checkpoint pt file, for continue training")
    args = parser.parse_args()

    config = load_config(args.config_path)
    
    OUTPUT_DIR = config['output_path']
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # tensorboard logs
    tensorboard = TensorboardWriter(os.path.join(OUTPUT_DIR, 'tensorboard'))

    vocab = load_vocab(config.vocab['vocab_path'])

    dataset = Dataset(config, vocab)

    # preprocess and normalise datasets
    dataset.preprocess_datasets()

    processor = dataset.processor

    # save vocab
    with open(os.path.join(OUTPUT_DIR, 'vocab.json'), "w", encoding="utf-8") as vocab_file:
        json.dump(vocab, vocab_file)

    # save config train
    copyfile(args.config_path, os.path.join(OUTPUT_DIR, 'config_train.json'))

    # save the feature_extractor and the tokenizer
    processor.save_pretrained(OUTPUT_DIR)
    # create data colletor
    data_collator = DataColletor(processor=processor, padding=True)

    # load model
    model = Wav2Vec2ForCTC.from_pretrained(
        args.checkpoint_path, 
        attention_dropout=config['attention_dropout'],
        hidden_dropout=config['hidden_dropout'],
        feat_proj_dropout=config['feat_proj_dropout'],
        mask_time_prob=config['mask_time_prob'],
        layerdrop=config['layerdrop'],
        gradient_checkpointing=config['gradient_checkpointing'], 
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ctc_zero_infinity=True
    )

    # freeze feature extractor
    if config['freeze_feature_extractor']:
        model.freeze_feature_extractor()

    # Use CUDA
    USE_CUDA = torch.cuda.is_available()
    if USE_CUDA:
        print("> Using CUDA")
        model = model.cuda()
    else:
        print("> CUDA is not available")
        model = model.cpu()

    # export model with transformers for save the config    
    model.save_pretrained(OUTPUT_DIR)

    train_dataset = DataLoader(dataset=dataset.train_dataset,
                          batch_size=config['batch_size'],
                          collate_fn=data_collator,
                          shuffle=True, num_workers=config['num_loader_workers'])

    devel_dataset = DataLoader(dataset=dataset.devel_dataset,
                          batch_size=config['batch_size'],
                          collate_fn=data_collator,
                          shuffle=False, num_workers=config['num_loader_workers'])

    # Define Optimizer
    optimizer = AdamW(model.parameters(), lr=config['lr'])
    
    global_step = 0
    restore_epoch = None
    lr_scheduler = None
    
    # scalers for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None

    # Learning rate scheduler
    if "lr_scheduler" in config:
        lr_scheduler = getattr(torch.optim.lr_scheduler, config.lr_scheduler)
        lr_scheduler = lr_scheduler(optimizer, **config.lr_scheduler_params)
    
    # restore optimizer
    optimizer_checkpoint_path = os.path.join(args.checkpoint_path, 'optimizer.pt')
    if os.path.isfile(optimizer_checkpoint_path):
        if USE_CUDA:
            map_location=lambda storage, loc: storage.cuda()
        else:
            map_location='cpu'

        try:
            state_dict = torch.load(optimizer_checkpoint_path, map_location=map_location)
            optimizer.load_state_dict(state_dict['optimizer'])
            global_step = state_dict['step']
            restore_epoch = state_dict['epoch']
            if lr_scheduler is not None:
                try:
                    lr_scheduler.load_state_dict(state_dict["scheduler"])
                    lr_scheduler.optimizer = optimizer
                    print("> Scheduler Loaded")
                except:
                    print("> Scheduler Load failed !")

            if "scaler" in state_dict and config.mixed_precision:
                print(" > Restoring AMP Scaler...")
                scaler.load_state_dict(state_dict["scaler"])

            print("> Optimizer Loaded")
        except:
            print("> Optimizer exist but is not possible load !")
    
    # GPU Audio Data augmentation
    if 'gpu_audio_augmentation' in config.keys(): 
        from torch_audiomentations import Compose, Gain, AddBackgroundNoise, ApplyImpulseResponse
        # ToDo: Implement Time mask and  
        gpu_audio_augmentation = Compose(
            transforms=[
                    AddBackgroundNoise(**config.gpu_audio_augmentation['additive']),
                    ApplyImpulseResponse(**config.gpu_audio_augmentation['rir']),
                    Gain(**config.gpu_audio_augmentation['gain']),
            ]
        )
    else:
        gpu_audio_augmentation = None
    best_loss = float('inf')
    early_epochs = 0

    max_epoch = config.epochs
    start_epoch = restore_epoch if restore_epoch else 0 

    for epoch in range(start_epoch, max_epoch):
        # evaluation when  epoch start, if continue training it's useful for save  really the best checkpoint
        devel_loss = evaluation(model, processor, devel_dataset, epoch, global_step, config, tensorboard, USE_CUDA)
        best_loss, early_epochs = save_best_checkpoint(OUTPUT_DIR, model, optimizer, lr_scheduler, scaler, global_step, epoch, devel_loss, best_loss, early_epochs)
        print("\n > EPOCH: {}/{}".format(epoch, max_epoch), flush=True)
        global_step = train(model, optimizer, lr_scheduler, scaler, train_dataset, gpu_audio_augmentation, epoch, global_step, config, tensorboard, USE_CUDA)
        print("\n > EPOCH END -- GLOBAL_STEP:", global_step)

        if config.early_stop_epochs:
            if early_epochs is not None:
                if early_epochs >= config.early_stop_epochs:
                    print("\n --> Train stopped by early stop at  Step:", global_step)
                    break # stop train
