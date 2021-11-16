import os
import re
import sys
import argparse
import pandas as pd
from tqdm import tqdm

# add previous and current path
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')


from utils.generic_utils import compute_wer, compute_cer


# portuguese char vocab
vocab_string_pt = 'abcdefghijklmnopqrstuvwxyzçãàáâêéíóôõúû '
vocab_string_russian = "абвгдежзийклмнопрстуфхцчшщъыьэюяё "

def remove_invalid_characters_and_normalise(text, args):
    text = text.lower()
    if args.russian:
        text = re.sub("[^{}]".format(vocab_string_russian), " ", text)
    else:
        text = re.sub("[^{}]".format(vocab_string_pt), " ", text)
    text = re.sub("[ ]+", " ", text)
    # remove doble blank spaces
    text = " ".join(text.split())

    return text

def normalize_path(path):
    return os.path.basename(path).split(".")[0]

def main():
    global args
    parser = argparse.ArgumentParser("""
        Examples: 
        python3  compute_metrics.py --dataset_csv ../../datasets/CORAA_Dataset/final/dataset/metadata_test_normalized_filtered.csv  --transcription_csv ../CORAA-Final-evaluation/wav2vec/test/transcript.txt --output_file ../CORAA-Final-evaluation/our/test/transcript_ground_truth_wer_normalized.csv
        python3  compute_metrics.py --dataset_csv ../../datasets/CORAA_Dataset/final/dataset/metadata_test_normalized_filtered.csv  --transcription_csv ../CORAA-Final-evaluation/gris2021/test/transcript.txt --output_file ../CORAA-Final-evaluation/gris2021/test/transcript_ground_truth_wer_normalized.csv
        """)
    parser.add_argument('--dataset_csv', required=True, help='CSV of dataset with ground truth texts')
    parser.add_argument('--transcription_csv', required=True, help='CSV with with transcriptions')
    parser.add_argument('--text_trans_column', default='transcription', help='Column of text in the Ground truth CSV')
    parser.add_argument('--audio_path_column', default='file_path', help='Column of audio path in the dataset CSV')
    parser.add_argument('--output_file', default='log_wer_transcript_text.csv', help='Filename to save CSV with transcripts, original text, and WER')
    parser.add_argument('--russian',
                        default=False,
                        action='store_true',
                        help='If the run is russian. It is used to control the vocab string')
    args = parser.parse_args()

    common_voice_text_column = "sentence"
    common_voice_audio_path_column = "path"

    df_dataset = pd.read_csv(args.dataset_csv, sep='\t')
    # rename audio column 
    df_dataset.rename(columns={common_voice_audio_path_column:args.audio_path_column}, inplace=True)

    df_transcriptions = pd.read_csv(args.transcription_csv, sep=',')
    if args.text_trans_column == common_voice_text_column:
        df_transcriptions = df_transcriptions.rename_column(args.text_trans_column, 'transcription')
        args.text_trans_column = 'transcription'

    # remove extra path and uses the file name without extension as file path

    df_transcriptions[args.audio_path_column] = df_transcriptions[args.audio_path_column].apply(normalize_path)
    df_dataset[args.audio_path_column] = df_dataset[args.audio_path_column].apply(normalize_path)

    '''
    for index, line in df_transcriptions.iterrows():
        df_transcriptions[args.audio_path_column][index] = os.path.basename(line[args.audio_path_column]).split(".")[0]

    for index, line in df_dataset.iterrows():
        df_dataset[args.audio_path_column][index] = os.path.basename(line[args.audio_path_column]).split(".")[0]
    '''
    # dropout all duplicates
    df_dataset.drop_duplicates(args.audio_path_column, inplace=True)
    df_transcriptions.drop_duplicates(args.audio_path_column, inplace=True)

    # print audios without transcription
    print("Transcript Num. Instances:", len(df_transcriptions.values.tolist()), "Dataset Num. Instances", len(df_dataset.values.tolist()))
    print("Audios with missing transcription:")
    print((set(df_dataset[args.audio_path_column].values.tolist()) - set(df_transcriptions[args.audio_path_column].values.tolist())))

    df = pd.merge(df_dataset, df_transcriptions, on=args.audio_path_column)
    
    wers = {"All":0, "Female":0, "Male":0, "Undefined": 0}
    instances = {"All":0, "Female":0, "Male":0, "Undefined": 0}

    without_gender_instaces = 0
    outputs = []
    for index, line in tqdm(df.iterrows()):
        if pd.isna(line[args.text_trans_column]):
            line[args.text_trans_column] = ''
        if pd.isna(line[common_voice_text_column]):
            continue 

        pred_text = remove_invalid_characters_and_normalise(line[args.text_trans_column], args)
        text = remove_invalid_characters_and_normalise(line[common_voice_text_column], args)
        if not text:
            continue

        gender = str(line["gender"])

        wer = compute_wer(text, pred_text)

        line['file_path'] = line['file_path'].lower()

        if gender == "female":
            wers["Female"] += wer
            instances["Female"] += 1
        elif gender == "male":
            wers["Male"] += wer
            instances["Male"] += 1
        else:
            gender = "undefined"
            wers["Undefined"] += wer
            instances["Undefined"] += 1
            without_gender_instaces += 1
            # print(line[args.audio_path_column], "Instances dont have gender !")

        wers["All"] += wer
        instances["All"] += 1

        outputs.append([line[args.audio_path_column], text, pred_text, gender, wer])
    # print results
    for key in wers.keys():
        if instances[key]:
            print(key, " WER: ", round((wers[key]/instances[key]) * 100,2))
            
    print(without_gender_instaces, "instances without gender !!")
    print("Num Males instances: ", instances["Male"])
    print("Num Females instances: ", instances["Female"])
    if args.output_file:
        df = pd.DataFrame(outputs, columns=["audio_path", "ground_truth", "transcription", "gender", "wer"])
        df.to_csv(args.output_file, index=False)
        print("\n\n> Outputs saved in: ", args.output_file)
if __name__ == "__main__":
  main()
