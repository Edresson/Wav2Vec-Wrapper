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


# bert


# BERT
from transformers import BertTokenizer, BertForMaskedLM

def score_fun_linear(s1, s2):
  return s2 + s1

class BERTScorer:
  def __init__(self, model_name, score_fn=score_fun_linear):
    self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load pre-trained model tokenizer (vocabulary)
    self._tokenizer = BertTokenizer.from_pretrained(model_name)
    # Load pre-trained model (weights)
    self._model = BertForMaskedLM.from_pretrained(model_name).to(self._device)
    self._model.eval()
    self._score_fn = score_fn
    self._CrossEntropyLoss = torch.nn.CrossEntropyLoss(reduction='none') # return loss for batch idxs

    print('---->>> Testing Model.')
    self.test_model(candidates=['Olá meu nome é João',
                                'Olá meu nome é Joao',
                                'Olá meu nome e Joao',
                                'O menino botou fogo no cidade',
                                'O menino botou fogo na cidade'])
    print('---->>> Done testing model')


  @staticmethod
  def chunks(l, n):
    for i in range(0, len(l), n):
      yield l[i:i + n]


  def score_with_candidates2(self, sentences):
    input_ids = self._tokenizer(sentences, padding = True, return_tensors="pt").to(self._device)
    with torch.no_grad():
        mask = input_ids['attention_mask']
        predictions = self._model(input_ids["input_ids"], attention_mask=mask)[0]
        predictions = predictions.transpose(1,2)
        excepted = input_ids["input_ids"]
        
        loss = self._CrossEntropyLoss(predictions.float(), excepted).data 
        loss = loss.masked_fill(mask == 0, 0)
    return loss.sum(dim=1)

  def score_with_candidates(self, sentences):
    ''' Its masked all token in all sentences generate more candidates, before its '''
    masked_sentences = []
    masked_count = []
    no_maked_setences = []
    for i in range(len(sentences)):
      sentence = sentences[i]
      masked_count.append(0)
      for token in sentence.split(" "):
        masked_sentences.append(sentence.replace(' '+token+' ', ' [MASK] '))
        no_maked_setences.append(sentence)
        masked_count[i] += 1

    input_ids = self._tokenizer(masked_sentences, padding = True, return_tensors="pt").to(self._device)
    excepteds_ids = self._tokenizer(no_maked_setences, padding = True, return_tensors="pt").to(self._device)

    with torch.no_grad():
        mask = input_ids['attention_mask']
        excepted = excepteds_ids["input_ids"]
        predictions = self._model(input_ids["input_ids"], attention_mask=mask)[0]
        predictions = predictions.transpose(1, 2)
        if predictions.size(2) > excepted.size(1):
            predictions = predictions[:, :, :excepted.size(1)]
            mask = mask[:, :predictions.size(2)]


        else:
            excepted = excepted[:, :predictions.size(2)]
            mask = mask[:, :predictions.size(2)]

        loss = self._CrossEntropyLoss(predictions.float(), excepted).data 
        loss = loss.masked_fill(mask == 0, 0)
        # unpack sentece parts
        loss_sum = []
        start = 0
        for i in range(len(masked_count)):
          loss_sum.append(loss[start:start+masked_count[i]].sum()/masked_count[i])
          start += masked_count[i]
    return torch.FloatTensor(loss_sum)

  def nlm_compute(self, candidates_full, bert_batch_size=100):
    max_len = 0
    for candidate in candidates_full:
      len_c = len(candidate.split(' '))
      if len_c > max_len:
        max_len = len_c
    # reset batch_size because each sentence generate max_len candidates
    bert_batch_size = int(bert_batch_size/max_len)
    results = torch.zeros(len(candidates_full))
    with torch.no_grad():
      for j, candidates in enumerate(self.chunks(candidates_full, bert_batch_size)):
        result = self.score_with_candidates(candidates)
        results[j*bert_batch_size:j*bert_batch_size + len(result)] = result * -1
    return results

  def test_model(self, candidates):
    for item in zip(list(self.nlm_compute(candidates).cpu().detach().numpy()), candidates):
      print("{0} ---- {1}".format(item[0], item[1]))


  def chose_best_candidate(self, candidates, candidate_scores):
    nln_scores = self.nlm_compute(candidates)
    candidate = candidates[0]
    score = -1000000000000
    for i in range(len(candidates)):
      kenlm_score = candidate_scores[i]
      neural_score = nln_scores[i].item()
      new_score = self._score_fn(kenlm_score, neural_score)
      if new_score >  score:
        # print(score, new_score, s1, s4, i)
        candidate = candidates[i]
        score = new_score
    return (candidate, nln_scores)


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
        BERT_MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'
        if BERT_MODEL_NAME:
            self.neural_lm = BERTScorer(BERT_MODEL_NAME)
        else:
            self.neural_lm = None

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
    steps = 0
    tot_wer = 0
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
                if kenlm.neural_lm:
                    # if external lm ...
                    pred_ids = [] 
                    for b in range(logits.size(0)):
                        candidates = []
                        scores = []
                        for c in range(len(lm_tokens)):
                            candidate = lm_tokens[c][b]
                            score = lm_scores[c][b]
                            candidates.append(candidate)
                            scores.append(score)
                        candidates_text = processor.batch_decode(candidates)

                        out_text, _ = kenlm.neural_lm.chose_best_candidate(candidates_text, scores)

                        with processor.as_target_processor():
                            pred_id = processor([out_text]).input_ids
                        pred_ids.append(pred_id[0])
                    pred_ids = np.array(pred_ids)   
            else:
                pred_ids = np.argmax(logits.cpu().detach().numpy(), axis=-1)

            if calcule_wer:
                # compute metrics 
                tot_wer += calculate_wer(pred_ids, labels.cpu().detach().numpy(), processor)

            if return_predictions:
                audios_path = batch['audio_path']
                # get text
                pred_string = processor.batch_decode(pred_ids)

                for i in range(len(audios_path)):
                    output_wav_path = audios_path[i]
                    if dataset_base_path:
                        output_wav_path = output_wav_path.replace(dataset_base_path, '').replace(dataset_base_path+'/', '')

                    predictions.append([output_wav_path, pred_string[i]])

            steps += 1
    if calcule_wer: 
        # calculate avg of metrics
        avg_wer = tot_wer/steps
        
        print("\n\n --> TEST PERFORMANCE\n")
        print("     | > :   WER    ({:.5f})\n".format(avg_wer))

    return predictions

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, required=True,
                        help="json file with configurations")
    parser.add_argument('--checkpoint_path_or_name', type=str, required=True,
                        help="path or name of checkpoints")
    parser.add_argument('--no_use_kenlm', default=False, action='store_true',
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
    pad_token = processor.tokenizer.pad_token
    silence_token = processor.tokenizer.word_delimiter_token
    unk_token = processor.tokenizer.unk_token

    # if the model uses upper words in vocab force tokenizer lower case for compatibility with our data loader
    if list(vocab_dict.keys())[-1].isupper():
        processor.tokenizer.do_lower_case = True

    data_collator = DataColletor(processor=processor, padding=True, test=True)

    if USE_CUDA:
        model = model.cuda()

    if not args.no_use_kenlm:
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

    if args.output_csv:
        root_path = os.path.dirname(args.output_csv)
        os.makedirs(root_path, exist_ok=True)

        df = pd.DataFrame(preds, columns=["file_path", "transcription"])
        df.to_csv(args.output_csv, index=False)
        print("\n\n> Evaluation outputs saved in: ", args.output_csv)
        
