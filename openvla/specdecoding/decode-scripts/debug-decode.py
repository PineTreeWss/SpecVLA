import os
import sys
import torch
#torch.cuda.set_device(2)
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
import time
os.chdir("/mnt/public/wangsongsheng/home/Projects/20250223-OpenVLA")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

from draccus.argparsing import ArgumentParser


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "/mnt/public/wangsongsheng/home/Projects/20250223-OpenVLA/exp-scripts/decode-libro/openvla-7b-finetuned-libero-goal"     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    #################################################################################################################
    # Speculative-execution parameters
    #################################################################################################################
    use_spec: bool = True
    spec_checkpoint: Union[str, Path] = "/mnt/public/wangsongsheng/home/Projects/20250223-OpenVLA/openvla/specdecoding/scripts/ckpt_libero_goal/state_199"
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_goal"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

# cfg=parser.parse_args()
cfg=GenerateConfig()

from openvla.specdecoding.model.cnets import MMModel
from openvla.specdecoding.model.cnets import EConfig

config = EConfig.from_pretrained(cfg.spec_checkpoint)

load_model_path=os.path.join(cfg.spec_checkpoint, "pytorch_model.bin")
ea_layer_state_dict = torch.load(load_model_path)


#from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
'''vla = OpenVLAForActionPrediction.from_pretrained(
            cfg.pretrained_checkpoint,
            torch_dtype=torch.bfloat16,
            load_in_8bit=cfg.load_in_8bit,
            load_in_4bit=cfg.load_in_4bit,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )'''
model = get_model(cfg)
#print('execute complete')
task_id=0
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict[cfg.task_suite_name]()
task = task_suite.get_task(task_id)
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict[cfg.task_suite_name]()
initial_states = task_suite.get_task_init_states(task_id)

env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

env.reset()

obs = env.set_init_state(initial_states[0])

t = 0
replay_images = []
if cfg.task_suite_name == "libero_spatial":
    max_steps = 220  # longest training demo has 193 steps
elif cfg.task_suite_name == "libero_object":
    max_steps = 280  # longest training demo has 254 steps
elif cfg.task_suite_name == "libero_goal":
    max_steps = 300  # longest training demo has 270 steps
elif cfg.task_suite_name == "libero_10":
    max_steps = 520  # longest training demo has 505 steps
elif cfg.task_suite_name == "libero_90":
    max_steps = 400  # longest training demo has 373 steps
resize_size = get_image_resize_size(cfg)
img = get_libero_image(obs, resize_size)

replay_images.append(img)

observation = {
    "full_image": img,
    "state": np.concatenate(
        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    ),
}
cfg.unnorm_key = cfg.task_suite_name
processor = get_processor(cfg)
action = get_action(
    cfg,
    model,
    observation,
    task_description,
    processor=processor,
    return_hidden_states=False
)

print('action: ',action)
print('execute complete')