import re
import sys
import argparse
import pandas as pd
from tqdm import tqdm


from utils.generic_utils import compute_wer, compute_cer


# portuguese char vocab
vocab_string = 'abcdefghijklmnopqrstuvwxyzçãàáâêéíóôõúû '

def remove_invalid_characters_and_normalise(text):
    text = text.lower()
    text = re.sub("[^{}]".format(vocab_string), " ", text)
    text = re.sub("[ ]+", " ", text)
    # remove doble blank spaces
    text = " ".join(text.split())

    return text

def main():
    
    parser = argparse.ArgumentParser("""
        Examples: 
        python3  compute_metrics.py --dataset_csv ../../datasets/CORAA_Dataset/final/dataset/metadata_test_normalized_filtered.csv  --transcription_csv ../CORAA-Final-evaluation/wav2vec/test/transcript.txt --output_file ../CORAA-Final-evaluation/our/test/transcript_ground_truth_wer_normalized.csv
        python3  compute_metrics.py --dataset_csv ../../datasets/CORAA_Dataset/final/dataset/metadata_test_normalized_filtered.csv  --transcription_csv ../CORAA-Final-evaluation/gris2021/test/transcript.txt --output_file ../CORAA-Final-evaluation/gris2021/test/transcript_ground_truth_wer_normalized.csv
        """)
    parser.add_argument('--dataset_csv', required=True, help='CSV of dataset with ground truth texts')
    parser.add_argument('--transcription_csv', required=True, help='CSV with with transcriptions')
    parser.add_argument('--text_column', default='text', help='Column of text in the dataset CSV')
    parser.add_argument('--text_trans_column', default='transcription', help='Column of text in the Ground truth CSV')
    parser.add_argument('--audio_path_column', default='file_path', help='Column of audio path in the dataset CSV')
    parser.add_argument('--output_file', default='log_wer_transcript_text.csv', help='Filename to save CSV with transcripts, original text, and WER')
    args = parser.parse_args()

    df_dataset = pd.read_csv(args.dataset_csv, sep=',')

    df_transcriptions = pd.read_csv(args.transcription_csv, sep=',')
    if args.text_trans_column == args.text_column:
        df_transcriptions = dataset.rename_column(args.text_trans_column, 'transcription')
        args.text_trans_column = 'transcription'

    df = pd.merge(df_dataset, df_transcriptions, on=args.audio_path_column)
    
    wers = {"ALIP":0, "TED":0, "NURC_RE":0, "CORAL":0, "SP2010":0}
    cers = {"ALIP":0, "TED":0, "NURC_RE":0, "CORAL":0, "SP2010":0}
    instances = {"ALIP":0, "TED":0, "NURC_RE":0, "CORAL":0, "SP2010":0}

    tot_wer = 0
    tot_cer = 0
    outputs = []
    tot_instances = 0
    for index, line in tqdm(df.iterrows()):
        if pd.isna(line[args.text_trans_column]):
            line[args.text_trans_column] = ''
        if pd.isna(line[args.text_column]):
            continue 
        pred_text = remove_invalid_characters_and_normalise(line[args.text_trans_column])
        text = remove_invalid_characters_and_normalise(line[args.text_column])
        # print(pred_text, '\n'+text)
        # wer = wer_metric.compute(predictions=[pred_text], references=[text])
        wer = compute_wer(text, pred_text)
        cer = compute_cer(text, pred_text)

        tot_cer += cer
        tot_wer += wer
        tot_instances += 1
        outputs.append([line[args.audio_path_column], text, pred_text, cer, wer])
        if '_alip_' in line['file_path'].lower():
            cers["ALIP"] += cer
            wers["ALIP"] += wer
            instances["ALIP"] += 1
        elif 'ted_' in line['file_path'].lower():
            cers["TED"] += cer
            wers["TED"] += wer
            instances["TED"] += 1
        elif 'nurc_re' in line['file_path'].lower():
            cers["NURC_RE"] += cer
            wers["NURC_RE"] += wer
            instances["NURC_RE"] += 1
        elif '_co_' in line['file_path'].lower():
            cers["CORAL"] += cer
            wers["CORAL"] += wer
            instances["CORAL"] += 1
        elif '_sp_' in line['file_path'].lower():
            cers["SP2010"] += cer
            wers["SP2010"] += wer
            instances["SP2010"] += 1
    # print results
    for key in wers.keys():
        if instances[key]:
            print(key, " CER: ", round((cers[key]/instances[key]) * 100, 2))
            print(key, " WER: ", round((wers[key]/instances[key]) * 100,2))
            

    print("Total CER:", round((tot_cer/tot_instances) * 100, 2))
    print("Total WER:", round((tot_wer/tot_instances) * 100, 2))
    if args.output_file:
        df = pd.DataFrame(outputs, columns=["audio_path", "ground_truth", "transcription", "cer", "wer"])
        df.to_csv(args.output_file, index=False)
        print("\n\n> Outputs saved in: ", args.output_file)
if __name__ == "__main__":
  main()
