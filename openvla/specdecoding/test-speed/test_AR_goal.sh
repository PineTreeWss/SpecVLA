CUDA_VISIBLE_DEVICES=3 MUJOCO_EGL_DEVICE_ID=3 python /SpecVLA/openvla/experiments/robot/libero/rucheng/run_libero_eval_AR.py\
  --model_family openvla \
  --pretrained_checkpoint /openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --center_crop True