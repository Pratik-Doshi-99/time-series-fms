cd /home/mltrain/time-series-fms
source train_env/bin/activate
python main.py --num_layers 4 --model_dim 256 --hidden_dim 1024 --num_heads 16 --epochs 2 --max_training_length 1024 --batch_size 16 --save_every 10000 --base_model_name "tsfm" --samples_per_file 10000 --warmup_steps 20000 --data_dir data --device "cuda:1" --run_name "TSFM-4L-256Model-1024Hidden" --base_dir "TSFM-4L-256Model-1024Hidden" --autoreg_expansion_factor 1023
