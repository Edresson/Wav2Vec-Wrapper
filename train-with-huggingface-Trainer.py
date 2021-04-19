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


from utils.dataset import Dataset, DataColletor
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
transformers.logging.set_verbosity_info()

wer_metric = load_metric("wer")

def evaluation(pred):
    global processor
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    # print("PRED:", pred_str, "Label:", label_str)
    return {"wer": wer}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, required=True,
                        help="json file with configurations")
    parser.add_argument('--checkpoint_path', type=str, default='facebook/wav2vec2-large-xlsr-53',
                        help="path of checkpoint pt file, for continue training")
    args = parser.parse_args()

    # config_path = 'example/config_example.json'
    config = load_config(args.config_path)
    
    OUTPUT_DIR = config['output_path']
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    vocab = load_vocab(config.vocab['vocab_path'])

    dataset = Dataset(config, vocab)

    # preprocess and normalise datasets
    dataset.preprocess_datasets()

    processor = dataset.processor

    # save the feature_extractor and the tokenizer
    processor.save_pretrained(OUTPUT_DIR)

    # save vocab
    with open(os.path.join(OUTPUT_DIR, 'vocab.json'), "w", encoding="utf-8") as vocab_file:
        json.dump(vocab, vocab_file)

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



    training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    logging_dir=os.path.join(OUTPUT_DIR, "tensorboard"), 
    group_by_length=True,
    logging_first_step=True,
    per_device_train_batch_size=config['batch_size'],
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=config['epochs'],
    fp16=config.mixed_precision,
    save_steps=config['save_step'],
    eval_steps=config['eval_step'],
    logging_steps=config['plot_step'],
    learning_rate=config['lr'],
    warmup_steps=config['warmup_steps'],
    load_best_model_at_end=config['load_best_model_at_end'], 
    save_total_limit=config['save_total_limit']
    )


    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=evaluation,
        train_dataset=dataset.train_dataset,
        eval_dataset=dataset.devel_dataset,
        tokenizer=processor.feature_extractor,
    )


    print("> Starting Training")
    train_result = trainer.train()
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

"""The training loss goes down and we can see that the WER on the test set also improves nicely. Because this notebook is just for demonstration purposes, we can stop here.

The resulting model of this notebook has been saved to [`patrickvonplaten/wav2vec2-large-xlsr-turkish-demo`](https://huggingface.co/patrickvonplaten/wav2vec2-large-xlsr-turkish-demo)

As a final check, let's load the model and verify that it indeed has learned to transcribe Turkish speech.

Let's first load the pretrained checkpoint.
"""

'''model = Wav2Vec2ForCTC.from_pretrained("/test/training-cv-pt-test/").to("cuda")
processor = Wav2Vec2Processor.from_pretrained("/test/training-cv-pt-test/")

"""Now, we will just take the first example of the test set, run it through the model and take the `argmax(...)` of the logits to retrieve the predicted token ids."""

input_dict = processor(common_voice_test[0]["input_values"], return_tensors="pt", padding=True)

logits = model(input_dict.input_values.to("cuda")).logits

pred_ids = torch.argmax(logits, dim=-1)[0]

"""We adapted `common_voice_test` quite a bit so that the dataset instance does not contain the original sentence label anymore. Thus, we re-use the original dataset to get the label of the first example."""
"""Finally, we can decode the example."""

print("Prediction:")
print(processor.decode(pred_ids))

print("\nReference:")
print(common_voice_test_transcription[0]["sentence"].lower())'''

"""Alright! The transcription can definitely be recognized from our prediction, but it is far from being perfect. Training the model a bit longer, spending more time on the data preprocessing, and especially using a language model for decoding would certainly improve the model's overall performance. 

For a demonstration model on a low-resource language, the results are acceptable, however 🤗.
"""