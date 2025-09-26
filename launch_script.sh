TRAINING_ARGS="$@"
python3 src/prepare_yaml.py
accelerate launch --config_file acc_cfg.yml src/pretrain.py $TRAINING_ARGS
