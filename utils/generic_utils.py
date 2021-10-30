import os
import re
import yaml
import json
import torch
import jiwer
import numpy as np
import jiwer.transforms as tr

class SentencesToListOfCharacters(tr.AbstractTransform):
    def process_string(self, s):
        return list(s)

    def process_list(self, inp):
        chars = []
        for sentence in inp:
            chars.extend(self.process_string(sentence))
        
        return chars

cer_transform = tr.Compose(
    [
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(), 
        SentencesToListOfCharacters(), # convert words to chars
        # jiwer.RemoveEmptyStrings()  # remove space strings
    ]
)

# It's the jiwer default transform
wer_transform = jiwer.Compose([
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.SentencesToListOfWords(),
    jiwer.RemoveEmptyStrings()
])

def compute_cer(reference, hypothesis):
    reference = reference.lower()
    hypothesis = hypothesis.lower()
    cer = jiwer.wer(reference, hypothesis, truth_transform=cer_transform, hypothesis_transform=cer_transform)
    return cer

def compute_wer(reference, hypothesis):
    reference = reference.lower()
    hypothesis = hypothesis.lower()
    wer = jiwer.wer(reference, hypothesis, truth_transform=wer_transform, hypothesis_transform=wer_transform) 
    return wer

def replace_special_tokens_and_normalize(text, vocab_string, processor):
    text = text.lower()
    text = text.replace(processor.tokenizer.unk_token, " ")
    text = text.replace(processor.tokenizer.pad_token, " ")
    text = text.replace(processor.tokenizer.word_delimiter_token, " ")
    text = re.sub("[^{}]".format(vocab_string+" "), " ", text)
    text = re.sub("[ ]+", " ", text)
    # remove doble blank spaces
    text = " ".join(text.split())
    return text

def calculate_wer(pred_ids, labels, processor, vocab_string, debug=False):
    labels[labels == -100] = processor.tokenizer.pad_token_id

    pred_string = processor.batch_decode(pred_ids)
    label_string = processor.batch_decode(labels, group_tokens=False)
    # wer = wer_metric.compute(predictions=pred_string, references=label_string)
    wer = 0
    cer = 0
    for i in range(len(pred_string)):
        reference = replace_special_tokens_and_normalize(label_string[i], vocab_string, processor)
        hypothesis = replace_special_tokens_and_normalize(pred_string[i], vocab_string, processor)
        if reference.replace(" ", "") == "":
            print('Setence:"', label_string[i],'"ignored for the metrics calculate')
            continue
        wer += compute_wer(reference, hypothesis)
        cer += compute_cer(reference, hypothesis)
    if debug:
        print(" > DEBUG: \n\n PRED:", pred_string, "\n Label:", label_string)
    return wer, cer

class AttrDict(dict):
    """A custom dict which converts dict keys
    to class attributes"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

def read_json_with_comments(json_path):
    # fallback to json
    with open(json_path, "r", encoding="utf-8") as f:
        input_str = f.read()
    # handle comments
    input_str = re.sub(r"\\\n", "", input_str)
    input_str = re.sub(r"//.*\n", "\n", input_str)
    data = json.loads(input_str)
    return data

def load_config(config_path: str) -> AttrDict:
    """Load config files and discard comments

    Args:
        config_path (str): path to config file.
    """
    config = AttrDict()

    ext = os.path.splitext(config_path)[1]
    if ext in (".yml", ".yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        data = read_json_with_comments(config_path)
    config.update(data)
    return config

def load_vocab(voba_path):
    config = AttrDict()
    config.update(read_json_with_comments(voba_path))
    return config

def save_best_checkpoint(log_dir, model, optimizer, lr_scheduler, scaler, step, epoch, val_loss, best_loss, early_epochs=None):
    if val_loss < best_loss:
        best_loss = val_loss
        if early_epochs is not None:
            early_epochs = 0
        
        model_save_path = os.path.join(log_dir, 'pytorch_model.bin')
        # model.save_pretrained(log_dir) # export model with transformers for save the config too
        torch.save(model.state_dict(), model_save_path)

        optimizer_save_path = os.path.join(log_dir, 'optimizer.pt')        
        checkpoint_dict = {
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
            'step': step,
            'epoch': epoch
        }

        if scaler is not None:
            checkpoint_dict['scaler'] = scaler.state_dict()

        torch.save(checkpoint_dict, optimizer_save_path)
       
        print("\n > BEST MODEL ({0:.5f}) saved at {1:}".format(
            val_loss, model_save_path))
    else:
        if early_epochs is not None:
            early_epochs += 1
    return best_loss, early_epochs
