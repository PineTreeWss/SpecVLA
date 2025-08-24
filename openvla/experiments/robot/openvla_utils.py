"""Utils for evaluating the OpenVLA policy."""

import json
import os
import time

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from openvla.prismatic.extern.hf.configuration_prismatic import OpenVLAConfig,SpecVLAConfig
from openvla.prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from openvla.prismatic.extern.hf.modeling_speculation import SpecVLAforActionPrediction
from openvla.prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

#import the speculative decoding dependency
from openvla.specdecoding.model.cnets import MMModel

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading in BF16 with Flash-Attention Enabled")

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoConfig.register("specvla", SpecVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    
    # 使用transformers的from_pretrained加载模型
    print("[*] 使用本地OpenVLAForActionPrediction类并从预训练检查点加载")
    if cfg.use_spec:
        print('load the vla model')
        #vla = OpenVLAforActionPrediction.from_pretrained(
        #    cfg.pretrained_checkpoint,
        #    load_in_8bit=cfg.load_in_8bit,
        #    load_in_4bit=cfg.load_in_4bit,
        #    low_cpu_mem_usage=True,
        #    trust_remote_code=True
            #use_spec = cfg.use_spec,
            #spec_checkpoint = cfg.spec_checkpoint
        #)
        vla = OpenVLAForActionPrediction.from_pretrained(
            cfg.pretrained_checkpoint,
            torch_dtype=torch.bfloat16,
            load_in_8bit=cfg.load_in_8bit,
            load_in_4bit=cfg.load_in_4bit,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_spec = True
        )
        #load_head_path=os.path.join(cfg.spec_checkpoint, "pytorch_model.bin")
        #print(load_model_path)
        #ea_layer_head = torch.load(load_head_path)
        if cfg.parallel_draft:
            vla = SpecVLAforActionPrediction(base_model=vla,base_model_name_or_path=cfg.pretrained_checkpoint,ea_model_path=cfg.spec_checkpoint,parallel_draft=cfg.parallel_draft)
            #if cfg.parallel_draft:
            #    print('parallel drafter loaded')
        else:
            vla = SpecVLAforActionPrediction(base_model=vla,base_model_name_or_path=cfg.pretrained_checkpoint,ea_model_path=cfg.spec_checkpoint,accept_threshold=cfg.accept_threshold)

            #breakpoint()
        #head = 
        #print('load the draft model')
        #load_model_path=os.path.join(cfg.spec_checkpoint, "pytorch_model.bin")
        #ea_layer_state_dict = torch.load(load_model_path)
        #print('reunify both models')
        #spec_vla = xx. xx  (vla,spec_head)
    else:
    # 使用from_pretrained直接加载模型
        vla = OpenVLAForActionPrediction.from_pretrained(
            cfg.pretrained_checkpoint,
            torch_dtype=torch.bfloat16,
            load_in_8bit=cfg.load_in_8bit,
            load_in_4bit=cfg.load_in_4bit,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    
    # 添加调试钩子
    # print("[*] 添加调试钩子")
    
    # 对predict_action方法添加调试钩子
    #original_predict_action = vla.predict_action
    
    #def debug_predict_action(*args, **kwargs):
        # print("\n=== 调用predict_action ===")
        # print(f"参数: {kwargs.keys()}")
        #if 'unnorm_key' in kwargs:
        #    pass
            # print(f"unnorm_key: {kwargs['unnorm_key']}")
        #result = original_predict_action(*args, **kwargs)
        # print(f"predict_action返回类型: {type(result)}")
        # if hasattr(result, 'shape'):
        #     print(f"predict_action返回shape: {result.shape}")
        # print("=== predict_action执行完毕 ===\n")
        #return result
    
    # 对generate方法添加调试钩子
    #original_generate = vla.generate
    
    #def debug_generate(*args, **kwargs):
        # print("\n=== 调用generate ===")
        # print(f"参数: {kwargs.keys()}")
        # print("max_new_tokens:", kwargs.get('max_new_tokens', 'not specified'))
    #    result = original_generate(*args, **kwargs)
        # print("调用self.generate方法")
        # print('1111111111111111')
        # print(f"generate返回类型: {type(result)}")
        # if hasattr(result, 'shape'):
        #     print(f"generate返回shape: {result.shape}")
        # print("=== generate执行完毕 ===\n")
    #    return result
    
    # 替换方法
    #vla.predict_action = debug_predict_action
    #vla.generate = debug_generate
    # print("已添加调试钩子到predict_action和generate方法")
    
    # Move model to device if not already
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    # Load dataset stats used during finetuning (for action un-normalization).
    dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, return_hidden_states=False,return_time=False,center_crop=False,generate_mode=None,accept_threshold=None,return_topk_index=False,token=None):
    """Generates an action with the VLA policy."""
    # print("\n====== 开始调用 get_vla_action ======")
    # print(f"模型类型: {type(vla)}")
    # print(f"模型类名: {vla.__class__.__name__}")
    # print(f"任务描述: {task_label}")
    
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # print(f"使用的提示语: {prompt}")
    
    # Process inputs.
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
    
    # 调试信息
    # print(f"处理后的输入类型: {type(inputs)}")
    # print(f"输入包含的键: {list(inputs.keys())}")
    # for key in inputs:
    #     if isinstance(inputs[key], torch.Tensor):
    #         print(f"  - {key} shape: {inputs[key].shape}")
    
    # print(f"predict_action方法的调用参数:")
    # print(f"  - unnorm_key: {unnorm_key}")
    # print(f"  - do_sample: False")
    
    # 打印模型上实际可用的方法
    # available_methods = [method_name for method_name in dir(vla) if callable(getattr(vla, method_name)) and not method_name.startswith('_')]
    # print(f"模型实例上可用的方法: {available_methods}")
    
    # 检查predict_action方法是否被我们的钩子替换
    # print(f"predict_action是否被钩子替换: {'debug_predict_action' in str(vla.predict_action)}")
    
    # Get action.
    # print("正在调用predict_action...")
    if return_hidden_states:
        action,token,hidden = vla.predict_action(**inputs, unnorm_key=unnorm_key, return_hidden_states=return_hidden_states,do_sample=False)
        return action,token,hidden
    if return_topk_index:
        action,token,hidden = vla.eval_topk(**inputs, unnorm_key=unnorm_key, return_hidden_states=return_hidden_states,do_sample=False)
        return action
    start_time = time.time()
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, return_hidden_states=return_hidden_states,do_sample=False,generate_mode=generate_mode)
    end_time = time.time()
    # print(f"predict_action返回的动作: shape={action.shape}")
    # print("====== 结束调用 get_vla_action ======\n")
    if return_time:
        return action,(end_time,start_time)
    return action
