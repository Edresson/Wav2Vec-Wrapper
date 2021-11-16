# -*- coding: utf-8 -*-
import os
import re
import json
import random
import argparse
import pandas as pd

import torchaudio
import librosa
import numpy as np

from shutil import copyfile
from utils.generic_utils import load_config, load_vocab

from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from datasets import load_dataset, load_metric
from datasets import ClassLabel

import transformers
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer
from transformers.trainer_utils import get_last_checkpoint
from transformers import EarlyStoppingCallback

transformers.logging.set_verbosity_info()

wer_metric = load_metric("wer")

def map_data_augmentation(aug_config):
    aug_name = aug_config['name']
    del aug_config['name']
    if aug_name == 'additive': 
        return AddBackgroundNoise(**aug_config)
    elif aug_name == 'gaussian':
        return AddGaussianNoise(**aug_config)
    elif aug_name == 'rir':
        return AddImpulseResponse(**aug_config)
    elif aug_name == 'gain':
        return Gain(**aug_config)
    elif aug_name == 'pitch_shift':
        return PitchShift(**aug_config)
    else:
        raise ValueError("The data augmentation '" + aug_name + "' doesn't exist !!")

def evaluation(pred):
    global processor
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    # remove empty strings
    while "" in label_str or " " in label_str:
        if "" in label_str:
            idx = label_str.index("")
            del label_str[idx], pred_str[idx]

        if " " in label_str:
            idx = label_str.index(" ")
            del label_str[idx], pred_str[idx]

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    # print("PRED:", pred_str, "Label:", label_str)
    return {"wer": wer}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, required=True,
                        help="json file with configurations")
    parser.add_argument('--checkpoint_path', type=str, default='facebook/wav2vec2-large-xlsr-53',
                        help="path of checkpoint pt file, for continue training")
    parser.add_argument('--continue_train',
                        default=False,
                        action='store_true',
                        help='If True Continue the training using the checkpoint_path')
    args = parser.parse_args()

    # config_path = 'example/config_example.json'
    config = load_config(args.config_path)
    
    
    OUTPUT_DIR = config['output_path']
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    vocab = load_vocab(config.vocab['vocab_path'])

    if 'preprocess_dataset' in config.keys() and config['preprocess_dataset']:
        from utils.dataset_preprocessed import Dataset, DataColletor
    else:
        from utils.dataset import Dataset, DataColletor
    dataset = Dataset(config, vocab)

    # preprocess and normalise datasets
    dataset.preprocess_datasets()

    processor = dataset.processor

    # save the feature_extractor and the tokenizer
    processor.save_pretrained(OUTPUT_DIR)

    # save vocab
    with open(os.path.join(OUTPUT_DIR, 'vocab.json'), "w", encoding="utf-8") as vocab_file:
        json.dump(vocab, vocab_file, ensure_ascii=False)

    # save config train
    copyfile(args.config_path, os.path.join(OUTPUT_DIR, 'config_train.json'))

    # Audio Data augmentation
    if 'audio_augmentation' in config.keys(): 
        from audiomentations import Compose, Gain, AddGaussianNoise, PitchShift, AddBackgroundNoise, AddImpulseResponse
        # ToDo: Implement Time mask and Freq mask
        audio_augmentator = Compose([map_data_augmentation(aug_config) for aug_config in config['audio_augmentation']])
    else:
        audio_augmentator = None

    # create data colletor
    data_collator = DataColletor(processor, audio_augmentator=audio_augmentator, sampling_rate=config.sampling_rate, padding=True)

    if os.path.isdir(args.checkpoint_path):
        last_checkpoint = get_last_checkpoint(args.checkpoint_path)
        print("> Resuming Train with checkpoint: ", last_checkpoint)
    else:
        last_checkpoint = None

    # load model
    model = Wav2Vec2ForCTC.from_pretrained(
        last_checkpoint if last_checkpoint else args.checkpoint_path, 
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

    training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    logging_dir=os.path.join(OUTPUT_DIR, "tensorboard"),
    report_to="all",
    group_by_length=True,
    logging_first_step=True,
    per_device_train_batch_size=config['batch_size'],
    dataloader_num_workers=config['num_loader_workers'],
    gradient_accumulation_steps=config['gradient_accumulation_steps'],
    seed=config.seed,
    num_train_epochs=config['epochs'],
    fp16=config.mixed_precision,
    logging_steps=config['logging_steps'],
    learning_rate=config['lr'],
    warmup_steps=config['warmup_steps'],
    warmup_ratio=config['warmup_ratio'],
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=config['load_best_model_at_end'],
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=config['save_total_limit']
    )
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=evaluation,
        train_dataset=dataset.train_dataset,
        eval_dataset=dataset.devel_dataset,
        tokenizer=processor.feature_extractor
    )

    if config['early_stop_epochs']:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=config['early_stop_epochs']))
    
    print("> Starting Training")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint if args.continue_train else None)
    # save best model
    # model.save_pretrained(OUTPUT_DIR)
    trainer.save_model()

    # save train results
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset.train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # save eval results
    print("--- Evaluate ---")
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(dataset.devel_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)