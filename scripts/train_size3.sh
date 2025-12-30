cd /home/pratikdoshi/projects/time-series-fms
source /home/pratikdoshi/projects/torch_env/bin/activate
python main.py --num_layers 12 --model_dim 768 --hidden_dim 2048 --num_heads 12 --epochs 2 --max_training_length 1024 --batch_size 8 --save_every 10000 --base_model_name "tsfm" --samples_per_file 10000 --warmup_steps 20000 --data_dir data --device "cuda:3" --run_name "TSFM-12L-768Model-2048Hidden" --base_dir "TSFM-12L-768Model-2048Hidden" --autoreg_expansion_factor 1023
