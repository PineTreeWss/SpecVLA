CUDA_VISIBLE_DEVICES=7 MUJOCO_EGL_DEVICE_ID=7 python PATH_TO_SPECVLA/openvla/experiments/robot/libero/run_libero_goal_Spec_Relaxed.py \
    --model_family openvla \
    --pretrained_checkpoint PATH_TO_SPECVLA/backbone_models/openvla-7b-finetuned-libero-goal \
    --task_suite_name libero_goal \
    --center_crop True