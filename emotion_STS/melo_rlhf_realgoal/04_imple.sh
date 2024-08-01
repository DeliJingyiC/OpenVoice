bash train.sh /home/ubuntu/OpenVoice/emotion_STS/melo/data/example/config.json 8
['/home/ubuntu/OpenVoice/data/synthesized_data/329_37150_shouting.wav', '329', '/home/ubuntu/OpenVoice/data/synthesized_data/329_41042_angry.wav', 'EN', '/home/ubuntu/OpenVoice/data/synthesized_data/24542_shouting_tts.wav.emo.npy']

python -m torch.distributed.run --nproc_per_node=1 --master_port=10902 --master_addr=localhost train.py --c config.json --model Model-1

bash train.sh /home/ubuntu/OpenVoice/emotion_STS/melo_rlhf_realgoal/data/example/config.json 8
bash /home/ubuntu/OpenVoice/MeloTTS/melo/train.sh /home/ubuntu/OpenVoice/MeloTTS/melo/logs/configs/config.json 1

###NISQA
python run_predict.py --mode predict_dir --pretrained_model weights/nisqa.tar --data_dir /home/ubuntu/OpenVoice/emotion_STS/melo_rlhf_realgoal/logs_emo/example/evaluation/G_56480_real --num_workers 0 --bs 10 --output_dir /home/ubuntu/OpenVoice/emotion_STS/melo_rlhf_realgoal/logs_emo/example/evaluation/G_56480_real
python run_predict.py --mode predict_file --pretrained_model weights/nisqa.tar --deg /home/ubuntu/OpenVoice/emotion_STS/melo_emo/evaluation/458_2922_10000_sad.wav --output_dir /home/ubuntu/OpenVoice/emotion_STS/melo_emo/generate_audio

bash train.sh /home/ubuntu/OpenVoice/emotion_STS/melo_rlhf/data/example/config.json 1
python run_predict.py --mode predict_dir --pretrained_model weights/nisqa.tar --data_dir /home/ubuntu/OpenVoice/emotion_STS/melo_rlhf/logs/example/evaluation/esd_input_ref --num_workers 0 --bs 10 --output_dir /home/ubuntu/OpenVoice/emotion_STS/melo_rlhf/logs/example/evaluation/esd_input_ref

# source /home/ubuntu/miniconda3/bin/activate
# conda activate .openvoiceenv