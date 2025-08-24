import argparse
import copy
#参数：开始/结束的数据idx，gpu-idx，用于多线程并行生成
parser = argparse.ArgumentParser(description='sp')

parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
parser.add_argument('--index', type=int, default=1)
parser.add_argument('--gpu_index', type=int, nargs='+', default=[0])
parser.add_argument('--outdir', type=str, default='outdir0')
#Config
from pathlib import Path
from typing import Optional, Union

class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "../../backbone_models/openvla-7b-finetuned-libero-goal"     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

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
gen_model_cfg=GenerateConfig()

class DataGenerationConfig:
    # fmt: off
    vla_path: str = "../../backbone_models/openvla-7b-finetuned-libero-goal"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("../../dataset/modified_libero_rlds")        # Path to Open-X dataset directory
    dataset_name: str = "libero_goal_no_noops"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    batch_size: int = 1                                          # Generation bsz
#暂时粘贴过来
import os
os.system("export PYTHONPATH=PATH_TO_SPECVLA")
os.chdir("PATH_TO_SPECVLA")
os.environ['RANK']='1'
os.environ['WORLD_SIZE']='1'
os.environ['MASTER_ADDR']='localhost'
os.environ['MASTER_PORT']='23456'
#倒入需要的modules
#import openvla
import torch
from torch.utils.data import DataLoader
from openvla.prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from openvla.prismatic.vla.action_tokenizer import ActionTokenizer
from transformers import AutoConfig, AutoImageProcessor
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from openvla.prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from openvla.prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from openvla.prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from openvla.prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from openvla.prismatic.util.data_utils import PaddedCollatorForActionPrediction
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from experiments.robot.openvla_utils import get_processor
from torch.nn.parallel import DistributedDataParallel as DDP
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from accelerate import PartialState

AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
distributed_state = PartialState()
torch.cuda.set_device(device_id := distributed_state.local_process_index)
torch.cuda.empty_cache()

cfg=DataGenerationConfig()
processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
action_tokenizer = ActionTokenizer(processor.tokenizer)
#Load大模型
quantization_config = None
print('loading vla')
model = get_model(gen_model_cfg)
processor = get_processor(gen_model_cfg)
print('loading data')
batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple([224, 224]),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )
print('start enumerating')

def writedata(name,data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length=len(os.listdir(name))
    idx=current_length
    torch.save(data_point, f'{name}/data_{idx}.ckpt')

#from transformers.modeling_outputs import CausalLMOutputWithPast
gen_model_cfg.unnorm_key = gen_model_cfg.task_suite_name
outdir = '/specdecoding/libero_goal_dataset_debug'
sample_num = 0
write_sample_num = 0
for batch_idx, batch in enumerate(dataloader):
        action,token,hidden = model.predict_action(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    unnorm_key=gen_model_cfg.unnorm_key,
                    return_hidden_states=True,
                    do_sample=False
                )
        td={"input_ids":batch["input_ids"].cpu()[0],"pixel_values":batch["pixel_values"],"hidden_state":hidden,"loss_mask":batch["attention_mask"].cpu()[0],'predicted_tokens':token}
        hidden_state = torch.cat([item for item in td['hidden_state'][1]],dim=0).shape[0]
        origin_hidden_state = td['hidden_state'][0][0].shape[0]
        if hidden_state == origin_hidden_state + 6:
            writedata(outdir,td)
            write_sample_num += 1
        sample_num += 1
        if sample_num % 1000 == 0:
            print(sample_num)
        #elif (sample_num==dataloader.__len__()):
        elif (sample_num==10):
            break 
print('generation ended')
print('sample num',sample_num)
print('valid sample num',write_sample_num)