CUDA_VISIBLE_DEVICES=3 MUJOCO_EGL_DEVICE_ID=3 python PATH_TO_SPECVLA/openvla/experiments/robot/libero/run_libero_goal_AR.py\
  --model_family openvla \
  --pretrained_checkpoint PATH_TO_SPECVLA/backbone_models/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --center_crop True