import re
import math
import torch
import torchaudio
import numpy as np
import argparse
import librosa
import warnings
import itertools as it
from jiwer import wer
from tqdm import tqdm
import pandas as pd
import os
from glob import glob

try:
    from flashlight.lib.sequence.criterion import CpuViterbiPath, get_data_ptr_as_bytes
    from flashlight.lib.text.dictionary import create_word_dict, load_words
    from flashlight.lib.text.decoder import (
        CriterionType,
        LexiconDecoderOptions,
        KenLM,
        LM,
        LMState,
        SmearingMode,
        Trie,
        LexiconDecoder,
    )
    from flashlight.lib.text.decoder import LexiconFreeDecoder, LexiconFreeDecoderOptions

except:
    warnings.warn(
        "flashlight python bindings are required to use this KenLM. Please install from https://github.com/facebookresearch/flashlight/tree/master/bindings/python"
    )
    LM = object
    LMState = object

from datasets import load_dataset, load_metric, concatenate_datasets

from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC

from utils.generic_utils import load_config, load_vocab, calculate_wer

from utils.dataset_preprocessed import remove_extra_columns, parse_dataset_dict, vocab_to_string, DataColletor
from torch.utils.data import DataLoader


def remove_invalid_characters(batch):
    text = batch[text_column].lower()
    text = re.sub("[^{}]".format(vocab_string), " ", text)
    text = re.sub("[ ]+", " ", text)
    batch[text_column] = text + " "
    return batch

def load_audio(batch):
    if dataset_base_path:
        batch[audio_path_column] = os.path.join(dataset_base_path, batch[audio_path_column])
    speech_array, sampling_rate = torchaudio.load(batch[audio_path_column])
    batch["speech"] = speech_array.squeeze().numpy()
    batch["sampling_rate"] = sampling_rate
    if text_column in batch:
        batch["target_text"] = batch[text_column]
    return batch

def resample_audio(batch):
    if batch["sampling_rate"] != config['sampling_rate']:
        batch["speech"] = librosa.resample(np.asarray(batch["speech"]),  batch["sampling_rate"], config['sampling_rate'])
        batch["sampling_rate"] = config['sampling_rate']
    return batch

def prepare_dataset(batch):
    batch['audio_path'] = batch[audio_path_column]
    batch["input_values"] = processor(batch["speech"], sampling_rate=config['sampling_rate']).input_values

    if "target_text" in batch:
        with processor.as_target_processor():
            batch["labels"] = processor(batch["target_text"]).input_ids
    return batch

class KenLMDecoder(object):
    def __init__(self, kenlm_args, vocab_dict, blank="<pad>", silence="|", unk="<unk>"):

        self.vocab_size = len(vocab_dict)
        self.blank_token = (vocab_dict[blank])
        self.silence_token = vocab_dict[silence]
        self.unk_token = vocab_dict[unk]

        self.nbest = kenlm_args['nbest']

        if kenlm_args['lexicon_path']:
            vocab_keys = vocab_dict.keys()
            self.lexicon = load_words(kenlm_args['lexicon_path'])
            self.word_dict = create_word_dict(self.lexicon)
            self.unk_word = self.word_dict.get_index(unk)

            self.lm = KenLM(kenlm_args['kenlm_model_path'], self.word_dict)
            self.trie = Trie(self.vocab_size, self.silence_token)

            start_state = self.lm.start(False)
            for i, (word, spellings) in enumerate(self.lexicon.items()):
                word_idx = self.word_dict.get_index(word)
                _, score = self.lm.score(start_state, word_idx)

                for spelling in spellings:
                    spelling_idxs = []
                    for token in spelling:
                        if token.upper() in vocab_keys:
                            spelling_idxs.append(vocab_dict[token.upper()])
                        elif token.lower() in vocab_keys:
                            spelling_idxs.append(vocab_dict[token.lower()])
                        else:
                            print("WARNING: The token", token, "not exist in your vocabulary, using <unk> token instead")
                            spelling_idxs.append(self.unk_token)
                        
                    self.trie.insert(spelling_idxs, word_idx, score)
            self.trie.smear(SmearingMode.MAX)

            self.decoder_opts = LexiconDecoderOptions(
                beam_size=kenlm_args['beam'],
                beam_size_token=kenlm_args['beam_size_token'] if "beam_size_token" in kenlm_args else len(vocab_dict),
                beam_threshold=kenlm_args['beam_threshold'],
                lm_weight=kenlm_args['lm_weight'],
                word_score=kenlm_args['word_score'],
                unk_score=-math.inf,
                sil_score=kenlm_args['sil_weight'],
                log_add=False,
                criterion_type=CriterionType.CTC,
            )

            self.decoder = LexiconDecoder(
                self.decoder_opts,
                self.trie,
                self.lm,
                self.silence_token,
                self.blank_token,
                self.unk_word,
                [],
                False,
            )
        else:
            d = {w: [[w]] for w in vocab_dict.keys()}
            self.word_dict = create_word_dict(d)
            self.lm = KenLM(kenlm_args['kenlm_model_path'], self.word_dict)
            self.decoder_opts = LexiconFreeDecoderOptions(
                beam_size=kenlm_args['beam'],
                beam_size_token=kenlm_args['beam_size_token'] if "beam_size_token" in kenlm_args else len(vocab_dict),
                beam_threshold=kenlm_args['beam_threshold'],
                lm_weight=kenlm_args['lm_weight'],
                sil_score=kenlm_args['sil_weight'],
                log_add=False,
                criterion_type=CriterionType.CTC,
            )
            self.decoder = LexiconFreeDecoder(
                self.decoder_opts, self.lm, self.silence_token, self.blank_token, []
            )

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank"""
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank_token, idxs)
        return torch.LongTensor(list(idxs))
        
    def decode(self, emissions):
        B, T, N = emissions.size()
        # print(emissions.shape)
        tokens = []
        scores = []
        for b in range(B):
            emissions_ptr = emissions.data_ptr() + 4 * b * emissions.stride(0)
            results = self.decoder.decode(emissions_ptr, T, N)
            nbest_results = results[: self.nbest]
            tokens_nbest = []
            scores_nbest = []
            for result in nbest_results:
                tokens_nbest.append(result.tokens)
                scores_nbest.append(result.score)
            tokens.append(tokens_nbest)
            scores.append(scores_nbest)

        token_array = np.array(tokens, dtype=object).transpose((1, 0, 2))
        scores_arrray = np.array(scores, dtype=object).transpose()
        return token_array, scores_arrray

def test(model, test_dataset, processor, kenlm, calcule_wer=True, return_predictions=False):
    model.eval()
    predictions = []
    tot_samples = 0
    tot_wer = 0
    tot_cer = 0
    with torch.no_grad():
        for batch in tqdm(test_dataset):
            input_values, attention_mask = batch['input_values'], batch['attention_mask']
            if calcule_wer:
                labels = batch['labels']
            
            if USE_CUDA:
                input_values = input_values.cuda(non_blocking=True)
                attention_mask = attention_mask.cuda(non_blocking=True)
                if calcule_wer:
                    labels = labels.cuda(non_blocking=True)
    
            logits = model(input_values, attention_mask=attention_mask).logits

            if kenlm:
                logits = torch.nn.functional.log_softmax(logits.float(), dim=-1)
                # get all candidates
                lm_tokens, lm_scores = kenlm.decode(logits.cpu().detach())
                # choise the best candidate
                pred_ids = lm_tokens[0][:]
            else:
                pred_ids = np.argmax(logits.cpu().detach().numpy(), axis=-1)

            if calcule_wer:
                # compute metrics
                wer, cer = calculate_wer(pred_ids, labels.cpu().detach().numpy(), processor, vocab_string)
                tot_wer += wer
                tot_cer += cer

            if return_predictions:
                audios_path = batch['audio_path']
                # get text
                pred_string = processor.batch_decode(pred_ids)

                for i in range(len(audios_path)):
                    output_wav_path = audios_path[i]
                    if dataset_base_path:
                        output_wav_path = output_wav_path.replace(dataset_base_path, '').replace(dataset_base_path+'/', '')

                    predictions.append([output_wav_path, pred_string[i].lower()])
            tot_samples += input_values.size(0)
    if calcule_wer: 
        # calculate avg of metrics
        avg_wer = tot_wer/tot_samples
        avg_cer = tot_cer/tot_samples
        print("\n\n --> TEST PERFORMANCE\n")
        print("     | > :   WER    ({:.5f})\n".format(avg_wer))
        print("     | > :   CER    ({:.5f})\n".format(avg_cer))

    return predictions

def inference(model, wavs, processor, kenlm):
    model.eval()
    predictions = []
    with torch.no_grad():
        for wav in tqdm(wavs):
            wav, path = wav
            features = processor(wav.numpy(), sampling_rate=16_000, padding=True, return_tensors="pt")
            input_values = features.input_values
            attention_mask = features.attention_mask

            if USE_CUDA:
                input_values = input_values.cuda(non_blocking=True)
                attention_mask = attention_mask.cuda(non_blocking=True)
    
            logits = model(input_values, attention_mask=attention_mask).logits

            if kenlm:
                logits = torch.nn.functional.log_softmax(logits.float(), dim=-1)
                # get all candidates
                lm_tokens, lm_scores = kenlm.decode(logits.cpu().detach())
                # choise the best candidate
                pred_ids = lm_tokens[0][:]
            else:
                pred_ids = np.argmax(logits.cpu().detach().numpy(), axis=-1)
            
            # get text
            pred_string = processor.batch_decode(pred_ids)
            predictions.append([path, pred_string[0]])
    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, required=True,
                        help="json file with configurations")
    parser.add_argument('--checkpoint_path_or_name', type=str, required=True,
                        help="path or name of checkpoints")
    parser.add_argument('--no_kenlm', default=False, action='store_true',
                        help="Not use KenLm during inference ?")
    parser.add_argument('--audio_path', type=str, default=None,
                        help="If it's passed the inference will be done in all audio files in this path and the dataset present in the config json will be ignored")
    parser.add_argument('--output_csv', type=str, default=None,
                        help="CSV for save all predictions")

    args = parser.parse_args()

    config = load_config(args.config_path)

    # Use CUDA
    USE_CUDA = torch.cuda.is_available()

    model = Wav2Vec2ForCTC.from_pretrained(args.checkpoint_path_or_name)

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=config['sampling_rate'], padding_value=0.0, do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor.from_pretrained(args.checkpoint_path_or_name)
    vocab_dict = processor.tokenizer.get_vocab()
     # if the model uses upper words in vocab force tokenizer lower case for compatibility with our data loader
    for c in list(vocab_dict.keys()):
        if c.isupper():
            processor.tokenizer.do_lower_case = True
            print("> Force lowercase Tokenizer !")
            break
    pad_token = processor.tokenizer.pad_token
    silence_token = processor.tokenizer.word_delimiter_token
    unk_token = processor.tokenizer.unk_token

   
    data_collator = DataColletor(processor=processor, padding=True, test=True)

    if USE_CUDA:
        model = model.cuda()

    if not args.no_kenlm:
        print("> Inference using KenLM")
        kenlm = KenLMDecoder(config.KenLM, vocab_dict, blank=pad_token, silence=silence_token, unk=unk_token)
    else:
        print("> Inference without KenLM")
        kenlm = None

    if not args.audio_path:
        # load dataset
        test_dataset_config = config.datasets['test']
        text_column, audio_path_column = parse_dataset_dict(test_dataset_config)

        dataset = load_dataset(**test_dataset_config)
        # made compatibility with csv load
        if isinstance(dataset, dict) and 'train' in dataset.keys():
            concat_list = []
            for k in dataset.keys():
                concat_list.append(dataset[k])
            dataset = concatenate_datasets(concat_list)

        if 'files_path' in config['datasets'].keys() and config.datasets['files_path']:
            if test_dataset_config['name'].lower() == 'csv':
                dataset_base_path = config.datasets['files_path']
            else:
                print("> Warning: datasets['files_path'] igonored because dataset is not CSV !")
                dataset_base_path = None
        else:
            dataset_base_path = None

        # preprocess dataset
        dataset = remove_extra_columns(dataset, text_column, audio_path_column)

        vocab_string = vocab_to_string(vocab_dict, pad_token, silence_token, unk_token).lower()
        print(vocab_string)
        print("\n\n> Remove invalid chars \n\n")
        # remove invalid chars
        dataset = dataset.map(remove_invalid_characters, num_proc=config['num_loader_workers'])

        # Load audio files
        dataset = dataset.map(load_audio)
        print("\n\n> Resample Audio Files \n\n")
        # resample audio files if necessary
        dataset = dataset.map(resample_audio, num_proc=config['num_loader_workers'])

        print("\n\n> Prepare dataset \n\n")
        # batched dataset
        dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names, batch_size=config['batch_size'], num_proc=1, batched=True)

        test_dataset = DataLoader(dataset=dataset,
                    batch_size=config['batch_size'],
                    collate_fn=data_collator,
                    shuffle=True, 
                    num_workers=config['num_loader_workers'])

        print("\n\n> Starting Evaluation \n\n")
        preds = test(model, test_dataset, processor, kenlm,  calcule_wer=True, return_predictions=True)

    else:
        # load dataset
        wavs = glob(os.path.join(args.audio_path,'*.wav'))

        vocab_string = vocab_to_string(vocab_dict, pad_token, silence_token, unk_token).lower()

        print("\n\n> Load and Resample Audio Files\n\n")
        # Load audio files
        wavs = [(torch.tensor(librosa.load(path, config['sampling_rate'])[0]), path) for path in wavs]

        print("\n\n> Starting Evaluation \n\n")
        preds = inference(model, wavs, processor, kenlm)
        
    df = pd.DataFrame(preds, columns=["file_path", "transcription"])
    if args.output_csv:
        root_path = os.path.dirname(args.output_csv)
        os.makedirs(root_path, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print("\n\n> Evaluation outputs saved in: ", args.output_csv)
    else:
        print(df.to_string())
