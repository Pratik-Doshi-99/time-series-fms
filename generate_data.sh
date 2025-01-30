cd /home/mltrain/time-series-fms
source train_env/bin/activate
python data.py --samples 100000 --samples_per_file 5000 --min_len 10 --max_len 10000 --base_dir data
