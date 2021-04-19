import os
import re
import yaml
import json
import torch
import numpy as np

from datasets import load_metric

wer_metric = load_metric("wer")

def calculate_wer(pred_logits, labels,  processor, debug=False):
    pred_ids = np.argmax(pred_logits, axis=-1)
    labels[labels == -100] = processor.tokenizer.pad_token_id

    pred_string = processor.batch_decode(pred_ids)
    label_string = processor.batch_decode(labels, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_string, references=label_string)
    if debug:
        print(" > DEBUG: \n\n PRED:", pred_string, "\n Label:", label_string)
    return wer

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