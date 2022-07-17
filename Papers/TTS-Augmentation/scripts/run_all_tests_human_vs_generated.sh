# Inference 

# Baseline
# PT
mkdir -p ../../../../results_paper/one-speaker/just_TTS_data/PT/test_in_GT/
CUDA_VISIBLE_DEVICES=0 nohup python3 ../../../../test.py -c ../../../../configs/config_train_CV_PT_test.json --checkpoint_path_or_name "Edresson/wav2vec2-large-100k-voxpopuli-ft-TTS-Dataset-portuguese"  --no_kenlm --output_csv ../../../../results_paper/one-speaker/just_TTS_data/PT/test_in_GT/transcription.csv > ../../../../results_paper/one-speaker/just_TTS_data/PT/test_in_GT/results.txt &

# RU
mkdir -p ../../../../results_paper/one-speaker/just_TTS_data/RU/test_in_GT/
CUDA_VISIBLE_DEVICES=0 nohup python3 ../../../test.py -c ../../../../configs/config_train_CV_RU_test.json --checkpoint_path_or_name "Edresson/wav2vec2-large-100k-voxpopuli-ft-TTS-Dataset-russian" --no_kenlm --output_csv ../../../../results_paper/one-speaker/just_TTS_data/RU/test_in_GT/transcription.csv > ../../../../results_paper/one-speaker/just_TTS_data/RU/test_in_GT/results.txt &



# Upper Bound
mkdir -p ../../../../results_paper/one-speaker/GT/PT/test_in_GT/
CUDA_VISIBLE_DEVICES=0 nohup python3 ../../../test.py -c ../../../../configs/CV/GT/config_train_CV_PT.json --checkpoint_path_or_name "Edresson/wav2vec2-large-100k-voxpopuli-ft-Common-Voice_plus_TTS-Dataset-portuguese" --no_kenlm --output_csv ../../../../results_paper/one-speaker/GT/PT/test_in_GT/transcription.csv > ../../../../results_paper/one-speaker/GT/PT/test_in_GT/results.txt &

mkdir -p ../../../../results_paper/one-speaker/GT/RU/test_in_GT/
CUDA_VISIBLE_DEVICES=0 nohup python3 ../../../test.py -c ../../../../configs/CV/GT/config_train_CV_RU.json --checkpoint_path_or_name "Edresson/wav2vec2-large-100k-voxpopuli-ft-Common-Voice_plus_TTS-Dataset-russian" --no_kenlm --output_csv ../../../../results_paper/one-speaker/GT/RU/test_in_GT/transcription.csv > ../../../../results_paper/one-speaker/GT/RU/test_in_GT/results.txt &


# Baseline + TTS/VC Augmentation

mkdir -p ../../../../results_paper/one-speaker/GEN/PT/test_in_GT/
CUDA_VISIBLE_DEVICES=0 nohup python3 ../../../test.py -c ../../../../configs/CV/GT/config_train_CV_PT.json --checkpoint_path_or_name "Edresson/wav2vec2-large-100k-voxpopuli-ft-TTS-Dataset-plus-data-augmentation-portuguese" --no_kenlm --output_csv ../../../../results_paper/one-speaker/GEN/PT/test_in_GT/transcription.csv > ../../../../results_paper/one-speaker/GEN/PT/test_in_GT/results.txt &

mkdir -p ../../../../results_paper/one-speaker/GEN/RU/test_in_GT/
CUDA_VISIBLE_DEVICES=0 nohup python3 ../../../test.py -c ../../../../configs/CV/GT/config_train_CV_RU.json --checkpoint_path_or_name "Edresson/wav2vec2-large-100k-voxpopuli-ft-TTS-Dataset-plus-data-augmentation-russian" --no_kenlm --output_csv ../../../../results_paper/one-speaker/GEN/RU/test_in_GT/transcription.csv > ../../../../results_paper/one-speaker/GEN/RU/test_in_GT/results.txt &


# Upper Bound + TTS/VC Augmentation

mkdir -p ../../../../results_paper/one-speaker/GT+GEN/PT/test_in_GT/
CUDA_VISIBLE_DEVICES=0 nohup python3 ../../../test.py -c ../../../../configs/CV/GT/config_train_CV_PT.json --checkpoint_path_or_name "Edresson/wav2vec2-large-100k-voxpopuli-ft-Common_Voice_plus_TTS-Dataset_plus_Data_Augmentation-portuguese" --no_kenlm --output_csv ../../../../results_paper/one-speaker/GT+GEN/PT/test_in_GT/transcription.csv > ../../../../results_paper/one-speaker/GT+GEN/PT/test_in_GT/results.txt &

mkdir -p ../../../../results_paper/one-speaker/GT+GEN/RU/test_in_GT/
CUDA_VISIBLE_DEVICES=0 nohup python3 ../../../test.py -c ../../../../configs/CV/GT/config_train_CV_RU.json --checkpoint_path_or_name "Edresson/wav2vec2-large-100k-voxpopuli-ft-Common_Voice_plus_TTS-Dataset_plus_Data_Augmentation-russian" --no_kenlm --output_csv ../../../../results_paper/one-speaker/GT+GEN/RU/test_in_GT/transcription.csv > ../../../../results_paper/one-speaker/GT+GEN/RU/test_in_GT/results.txt &


# External WER by gender

# Baseline

# PT 
nohup python3 compute_metrics.py --dataset_csv ../../../../results_paper/test-common_voice-pt.tsv --transcription_csv ../../../../results_paper/one-speaker/just_TTS_data/PT/test_in_GT/transcription.csv  --output_file  ../../../../results_paper/one-speaker/just_TTS_data/PT/test_in_GT/results_log_wer_text_text_pred_gender_wer.csv >  ../../../../results_paper/one-speaker/just_TTS_data/PT/test_in_GT/results_external_by_gender.txt &

# RU
nohup python3 compute_metrics.py --dataset_csv ../../../../results_paper/test-common_voice-ru.tsv --transcription_csv ../../../../results_paper/one-speaker/just_TTS_data/RU/test_in_GT/transcription.csv  --output_file  ../../../../results_paper/one-speaker/just_TTS_data/RU/test_in_GT/results_log_wer_text_text_pred_gender_wer.csv --russian >  ../../../../results_paper/one-speaker/just_TTS_data/RU/test_in_GT/results_external_by_gender.txt &



# Upper Bound
# PT
nohup python3 compute_metrics.py --dataset_csv ../../../../results_paper/test-common_voice-pt.tsv --transcription_csv ../../../../results_paper/one-speaker/GT/PT/test_in_GT/transcription.csv  --output_file  ../../../../results_paper/one-speaker/GT/PT/test_in_GT/results_log_wer_text_text_pred_gender_wer.csv >  ../../../../results_paper/one-speaker/GT/PT/test_in_GT/results_external_by_gender.txt &

# RU
nohup python3 compute_metrics.py --dataset_csv ../../../../results_paper/test-common_voice-ru.tsv --transcription_csv ../../../../results_paper/one-speaker/GT/RU/test_in_GT/transcription.csv  --output_file  ../../../../results_paper/one-speaker/GT/RU/test_in_GT/results_log_wer_text_text_pred_gender_wer.csv --russian >  ../../../../results_paper/one-speaker/GT/RU/test_in_GT/results_external_by_gender.txt &


# Baseline + TTS/VC Augmentation
# PT
nohup python3 compute_metrics.py --dataset_csv ../../../../results_paper/test-common_voice-pt.tsv --transcription_csv ../../../../results_paper/one-speaker/GEN/PT/test_in_GT/transcription.csv  --output_file  ../../../../results_paper/one-speaker/GEN/PT/test_in_GT/results_log_wer_text_text_pred_gender_wer.csv >  ../../../../results_paper/one-speaker/GEN/PT/test_in_GT/results_external_by_gender.txt &

# RU
nohup python3 compute_metrics.py --dataset_csv ../../../../results_paper/test-common_voice-ru.tsv --transcription_csv ../../../../results_paper/one-speaker/GEN/RU/test_in_GT/transcription.csv  --output_file  ../../../../results_paper/one-speaker/GEN/RU/test_in_GT/results_log_wer_text_text_pred_gender_wer.csv --russian >  ../../../../results_paper/one-speaker/GEN/RU/test_in_GT/results_external_by_gender.txt &


# Upper Bound + TTS/VC Augmentation
# PT
nohup python3 compute_metrics.py --dataset_csv ../../../../results_paper/test-common_voice-pt.tsv --transcription_csv ../../../../results_paper/one-speaker/GT+GEN/PT/test_in_GT/transcription.csv  --output_file  ../../../../results_paper/one-speaker/GT+GEN/PT/test_in_GT/results_log_wer_text_text_pred_gender_wer.csv >  ../../../../results_paper/one-speaker/GT+GEN/PT/test_in_GT/results_external_by_gender.txt &

# RU
nohup python3 compute_metrics.py --dataset_csv ../../../../results_paper/test-common_voice-ru.tsv --transcription_csv ../../../../results_paper/one-speaker/GT+GEN/RU/test_in_GT/transcription.csv  --output_file  ../../../../results_paper/one-speaker/GT+GEN/RU/test_in_GT/results_log_wer_text_text_pred_gender_wer.csv --russian >  ../../../../results_paper/one-speaker/GT+GEN/RU/test_in_GT/results_external_by_gender.txt &





# # PT
# nohup python3 compute_metrics.py --dataset_csv ../../../../results_paper/test-common_voice-pt.tsv --transcription_csv ../../../../results_paper/Human_vs_Generated/GEN/PT/test_in_GT/transcription.csv  --output_file  ../../../../results_paper/Human_vs_Generated/GEN/PT/test_in_GT/log_wer_text_text_pred_gender_wer.csv >  ../../../../results_paper/Human_vs_Generated/GEN/PT/test_in_GT/results_external_by_gender.txt &
# nohup python3 compute_metrics.py --dataset_csv ../../../../results_paper/test-common_voice-pt.tsv --transcription_csv ../../../../results_paper/Human_vs_Generated/GEN/PT/test_in_GEN/transcription.csv  --output_file  ../../../../results_paper/Human_vs_Generated/GEN/PT/test_in_GEN/log_wer_text_text_pred_gender_wer.csv >  ../../../../results_paper/Human_vs_Generated/GEN/PT/test_in_GEN/results_external_by_gender.txt &

# nohup python3 compute_metrics.py --dataset_csv ../../../../results_paper/test-common_voice-pt.tsv --transcription_csv ../../../../results_paper/Human_vs_Generated/GEN-with-DA-PS/PT/test_in_GT/transcription.csv  --output_file  ../../../../results_paper/Human_vs_Generated/GEN-with-DA-PS/PT/test_in_GT/log_wer_text_text_pred_gender_wer.csv >  ../../../../results_paper/Human_vs_Generated/GEN-with-DA-PS/PT/test_in_GT/results_external_by_gender.txt &
# nohup python3 compute_metrics.py --dataset_csv ../../../../results_paper/test-common_voice-pt.tsv --transcription_csv ../../../../results_paper/Human_vs_Generated/GEN-with-DA-PS/PT/test_in_GEN/transcription.csv  --output_file  ../../../../results_paper/Human_vs_Generated/GEN-with-DA-PS/PT/test_in_GEN/log_wer_text_text_pred_gender_wer.csv >  ../../../../results_paper/Human_vs_Generated/GEN-with-DA-PS/PT/test_in_GEN/results_external_by_gender.txt &

# nohup python3 compute_metrics.py --dataset_csv ../../../../results_paper/test-common_voice-pt.tsv --transcription_csv ../../../../results_paper/Human_vs_Generated/GT/PT/test_in_GT/transcription.csv  --output_file  ../../../../results_paper/Human_vs_Generated/GT/PT/test_in_GT/log_wer_text_text_pred_gender_wer.csv >  ../../../../results_paper/Human_vs_Generated/GT/PT/test_in_GT/results_external_by_gender.txt &
# nohup python3 compute_metrics.py --dataset_csv ../../../../results_paper/test-common_voice-pt.tsv --transcription_csv ../../../../results_paper/Human_vs_Generated/GT/PT/test_in_GEN/transcription.csv  --output_file  ../../../../results_paper/Human_vs_Generated/GT/PT/test_in_GEN/log_wer_text_text_pred_gender_wer.csv >  ../../../../results_paper/Human_vs_Generated/GT/PT/test_in_GEN/results_external_by_gender.txt &

# nohup python3 compute_metrics.py --dataset_csv ../../../../results_paper/test-common_voice-pt.tsv --transcription_csv ../../../../results_paper/Human_vs_Generated/GT_with_DA_PS/PT/test_in_GT/transcription.csv  --output_file  ../../../../results_paper/Human_vs_Generated/GT_with_DA_PS/PT/test_in_GT/log_wer_text_text_pred_gender_wer.csv >  ../../../../results_paper/Human_vs_Generated/GT_with_DA_PS/PT/test_in_GT/results_external_by_gender.txt &
# nohup python3 compute_metrics.py --dataset_csv ../../../../results_paper/test-common_voice-pt.tsv --transcription_csv ../../../../results_paper/Human_vs_Generated/GT_with_DA_PS/PT/test_in_GEN/transcription.csv  --output_file  ../../../../results_paper/Human_vs_Generated/GT_with_DA_PS/PT/test_in_GEN/log_wer_text_text_pred_gender_wer.csv >  ../../../../results_paper/Human_vs_Generated/GT_with_DA_PS/PT/test_in_GEN/results_external_by_gender.txt &


# # RU

# nohup python3 compute_metrics.py --dataset_csv ../../../../results_paper/test-common_voice-ru.tsv --transcription_csv ../../../../results_paper/Human_vs_Generated/GEN/RU/test_in_GT/transcription.csv  --output_file  ../../../../results_paper/Human_vs_Generated/GEN/RU/test_in_GT/log_wer_text_text_pred_gender_wer.csv --russian >  ../../../../results_paper/Human_vs_Generated/GEN/RU/test_in_GT/results_external_by_gender.txt &
# nohup python3 compute_metrics.py --dataset_csv ../../../../results_paper/test-common_voice-ru.tsv --transcription_csv ../../../../results_paper/Human_vs_Generated/GEN/RU/test_in_GEN/transcription.csv  --output_file  ../../../../results_paper/Human_vs_Generated/GEN/RU/test_in_GEN/log_wer_text_text_pred_gender_wer.csv --russian >  ../../../../results_paper/Human_vs_Generated/GEN/RU/test_in_GEN/results_external_by_gender.txt &

# nohup python3 compute_metrics.py --dataset_csv ../../../../results_paper/test-common_voice-ru.tsv --transcription_csv ../../../../results_paper/Human_vs_Generated/GEN-with-DA-PS/RU/test_in_GT/transcription.csv  --output_file  ../../../../results_paper/Human_vs_Generated/GEN-with-DA-PS/RU/test_in_GT/log_wer_text_text_pred_gender_wer.csv --russian >  ../../../../results_paper/Human_vs_Generated/GEN-with-DA-PS/RU/test_in_GT/results_external_by_gender.txt &
# nohup python3 compute_metrics.py --dataset_csv ../../../../results_paper/test-common_voice-ru.tsv --transcription_csv ../../../../results_paper/Human_vs_Generated/GEN-with-DA-PS/RU/test_in_GEN/transcription.csv  --output_file  ../../../../results_paper/Human_vs_Generated/GEN-with-DA-PS/RU/test_in_GEN/log_wer_text_text_pred_gender_wer.csv --russian >  ../../../../results_paper/Human_vs_Generated/GEN-with-DA-PS/RU/test_in_GEN/results_external_by_gender.txt &

# nohup python3 compute_metrics.py --dataset_csv ../../../../results_paper/test-common_voice-ru.tsv --transcription_csv ../../../../results_paper/Human_vs_Generated/GT/RU/test_in_GT/transcription.csv  --output_file  ../../../../results_paper/Human_vs_Generated/GT/RU/test_in_GT/log_wer_text_text_pred_gender_wer.csv --russian >  ../../../../results_paper/Human_vs_Generated/GT/RU/test_in_GT/results_external_by_gender.txt &
# nohup python3 compute_metrics.py --dataset_csv ../../../../results_paper/test-common_voice-ru.tsv --transcription_csv ../../../../results_paper/Human_vs_Generated/GT/RU/test_in_GEN/transcription.csv  --output_file  ../../../../results_paper/Human_vs_Generated/GT/RU/test_in_GEN/log_wer_text_text_pred_gender_wer.csv --russian >  ../../../../results_paper/Human_vs_Generated/GT/RU/test_in_GEN/results_external_by_gender.txt &

# nohup python3 compute_metrics.py --dataset_csv ../../../../results_paper/test-common_voice-ru.tsv --transcription_csv ../../../../results_paper/Human_vs_Generated/GT_with_DA_PS/RU/test_in_GT/transcription.csv  --output_file  ../../../../results_paper/Human_vs_Generated/GT_with_DA_PS/RU/test_in_GT/log_wer_text_text_pred_gender_wer.csv --russian >  ../../../../results_paper/Human_vs_Generated/GT_with_DA_PS/RU/test_in_GT/results_external_by_gender.txt &
# nohup python3 compute_metrics.py --dataset_csv ../../../../results_paper/test-common_voice-ru.tsv --transcription_csv ../../../../results_paper/Human_vs_Generated/GT_with_DA_PS/RU/test_in_GEN/transcription.csv  --output_file  ../../../../results_paper/Human_vs_Generated/GT_with_DA_PS/RU/test_in_GEN/log_wer_text_text_pred_gender_wer.csv --russian >  ../../../../results_paper/Human_vs_Generated/GT_with_DA_PS/RU/test_in_GEN/results_external_by_gender.txt &

