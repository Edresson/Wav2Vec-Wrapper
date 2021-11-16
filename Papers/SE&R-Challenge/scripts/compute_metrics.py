import os
import re
import argparse
import pandas as pd
from tqdm import tqdm


import sys
# add previous and current path
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')


from utils.generic_utils import compute_wer, compute_cer

def remove_invalid_characters_and_normalise(text, vocab_string):
    text = text.lower()
    text = re.sub("[^{}]".format(vocab_string), " ", text)
    text = re.sub("[ ]+", " ", text)
    # remove doble blank spaces
    text = " ".join(text.split())

    return text

def normalize_path(path):
    return os.path.basename(path)

def compute_asr_metrics(args):
    # load csvs
    df_dataset = pd.read_csv(args.dataset_csv, sep=',')
    df_transcriptions = pd.read_csv(args.transcription_csv, sep=',')

    # remove extra path
    df_transcriptions[args.audio_path_column] = df_transcriptions[args.audio_path_column].apply(normalize_path)
    df_dataset[args.audio_path_column] = df_dataset[args.audio_path_column].apply(normalize_path)    

    # the model batch can geenrate duplicates lines so, dropout all duplicates
    df_dataset.drop_duplicates(args.audio_path_column, inplace = True)
    df_transcriptions.drop_duplicates(args.audio_path_column, inplace = True)

    # sort to guarantee the same order
    df_dataset = df_dataset.sort_values(by=[args.audio_path_column])
    df_transcriptions = df_transcriptions.sort_values(by=[args.audio_path_column])

    # check if have all files in df_transcriptions
    if len(df_transcriptions.values.tolist()) != len(df_dataset.values.tolist()):
        return "ERROR: The following audios are missing in our CSV file: "+ str(set(df_dataset[args.audio_path_column].values.tolist()) - set(df_transcriptions[args.audio_path_column].values.tolist()))

    # dropall except the audio and text key for transcription df 
    df_transcriptions = df_transcriptions.filter([args.audio_path_column, args.text_trans_column])

    # merge dataframes
    df = pd.merge(df_dataset, df_transcriptions, on=args.audio_path_column)

    # create aux dicts
    Spontaneus_sets = ["ALIP", "NURC-Recife", "C-ORAL-BRASIL I", "SP2010"]
    eval_sets = ["default", "Prepared Speech PT_PT", "Prepared Speech PT_BR"] + Spontaneus_sets
    wers ={}
    cers = {}
    instances = {}
    for key in eval_sets:
        wers[key] = 0
        cers[key] = 0
        instances[key] = 0

    for index, line in df.iterrows(): # tqdm()
        # if transcription is "" pandas will convert to null, recover it
        if pd.isna(line[args.text_trans_column]):
            line[args.text_trans_column] = ''
        # if GT is nan ignore this instance
        if pd.isna(line[args.text_column]):
            continue
        # remove invalid chars and normalize  
        pred_text = remove_invalid_characters_and_normalise(line[args.text_trans_column], args.vocab_string)
        text = remove_invalid_characters_and_normalise(line[args.text_column], args.vocab_string)
        # compute the metrics
        wer = compute_wer(text, pred_text)
        cer = compute_cer(text, pred_text)
        
        dataset = line["dataset"]
        accent = line["variety"]
        key = "default"
        if dataset == "TEDx Talks":
            if accent == "pt_br":
                key = "Prepared Speech PT_BR"
            elif accent == "pt_pt":
                key = "Prepared Speech PT_PT"
        else:
            key = dataset

        wers[key] += wer
        cers[key] += cer
        instances[key] += 1

    # prepare the outputs and compute the average for spontaneous Speech
    outputs = {}
    spon_cer = 0
    spon_wer = 0
    for key in instances.keys():
        if instances[key]:
            outputs[key] = {"cer": round((cers[key]/instances[key]) * 100, 4), "wer": round((wers[key]/instances[key]) * 100, 4)}
            if key in Spontaneus_sets:
                spon_cer += cers[key]/instances[key]
                spon_wer += wers[key]/instances[key]
    
    
    # compute spontaneous speech average
    spon_cer = spon_cer/len(Spontaneus_sets)
    spon_wer = spon_wer/len(Spontaneus_sets)
    outputs["Spontaneous Speech"] = {"cer": round((spon_cer) * 100, 4), "wer": round((spon_wer) * 100, 4)}

    # average for prepared speech
    prep_wer = (outputs["Prepared Speech PT_PT"]["wer"] + outputs["Prepared Speech PT_BR"]["wer"]) / 2
    prep_cer = (outputs["Prepared Speech PT_PT"]["cer"] + outputs["Prepared Speech PT_BR"]["cer"]) / 2

    # average spontaneus + prepared speech
    mixed_wer = (outputs["Spontaneous Speech"]["wer"] + prep_wer) / 2
    mixed_cer = (outputs["Spontaneous Speech"]["cer"] + prep_cer) / 2

    outputs["Mixed"] = {"cer": round(mixed_cer, 4), "wer": round(mixed_wer, 4)}
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser("""
        Examples: 
        python3  compute_metrics.py --dataset_csv ../../../../../datasets/CORAA_Dataset/final/dataset/metadata_test_normalized_filtered.csv  --transcription_csv ../CORAA-Final-evaluation/wav2vec/test/transcript.txt --output_file ../CORAA-Final-evaluation/our/test/transcript_ground_truth_wer_normalized.csv
        python3  compute_metrics.py --dataset_csv ../../datasets/CORAA_Dataset/final/dataset/metadata_test_normalized_filtered.csv  --transcription_csv ../CORAA-Final-evaluation/gris2021/test/transcript.txt --output_file ../CORAA-Final-evaluation/gris2021/test/transcript_ground_truth_wer_normalized.csv
        """)
    parser.add_argument('--dataset_csv', required=True, help='CSV of dataset with ground truth texts')
    parser.add_argument('--transcription_csv', required=True, help='CSV with with transcriptions')
    parser.add_argument('--text_column', default='text', help='Column of text in the dataset CSV')
    parser.add_argument('--text_trans_column', default='transcription', help='Column of text in the Ground truth CSV')
    parser.add_argument('--audio_path_column', default='file_path', help='Column of audio path in the dataset CSV')
    parser.add_argument('--vocab_string', default="abcdefghijklmnopqrstuvwxyzçãàáâêéíóôõúû ", help='Vocabulary for the language plus space, default (Portuguese): "abcdefghijklmnopqrstuvwxyzçãàáâêéíóôõúû "')
    parser.add_argument('--output_file', default='log_wer_transcript_text.csv', help='Filename to save CSV with transcripts, original text, and WER')
    args = parser.parse_args()

    out_dict = compute_asr_metrics(args)

    for key in out_dict.keys():
        print(key, "CER:", out_dict[key]["cer"], "WER:", out_dict[key]["wer"])