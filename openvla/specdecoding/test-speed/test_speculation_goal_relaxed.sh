CUDA_VISIBLE_DEVICES=7 MUJOCO_EGL_DEVICE_ID=7 python /SpecVLA/openvla/experiments/robot/libero/new_run_libero_eval_Speculation_set_params.py \
    --model_family openvla \
    --pretrained_checkpoint /openvla-7b-finetuned-libero-goal \
    --task_suite_name libero_goal \
    --center_crop True