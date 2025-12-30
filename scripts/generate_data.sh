cd /home/pratikdoshi/projects/time-series-fms
source /home/pratikdoshi/projects/torch_env/bin/activate
python data/dataset.py --samples 100000 --samples_per_file 10000 --min_len 10 --max_len 10000 --base_dir synth-data