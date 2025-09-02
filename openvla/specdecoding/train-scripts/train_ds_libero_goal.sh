#export PYTHONPATH='/OpenVLA'
WANDB_MODE='offline' deepspeed --master_port 23333 --include=localhost:2,3 "train_deepspeed_libero_goal.py" --deepspeed_config "ds_config.json"