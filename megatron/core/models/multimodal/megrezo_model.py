import math
import re
from typing import Dict
import json
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union
import os
import importlib
from functools import partial

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler

import transformers
from transformers import AutoProcessor
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

### vlm related import ###

# # run without megatron
# from modeling_navit_siglip import SiglipVisionTransformer, SiglipVisionConfig
# from audio import AudioEncoder
# from resampler import Resampler
# from utils import (
#     insert_audio_embeddings, insert_image_embeddings, load_inputs, prepare_audio_embeddings, prepare_image_embeddings, prepare_bounds_audio, prepare_raw_msgs_minicpm_v26)
# from dataset import SupervisedDataset, data_collator

# run megatron
from .modeling_navit_siglip import SiglipVisionTransformer, SiglipVisionConfig
from .audio import AudioEncoder
from .resampler import Resampler
from .utils import (
    insert_audio_embeddings, insert_image_embeddings, load_inputs, prepare_audio_embeddings, prepare_image_embeddings, prepare_bounds_audio, prepare_raw_msgs_minicpm_v26)
from .dataset import SupervisedDataset, data_collator
from megatron.training import get_args

### vlm related import ###

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="output/init_model/init_0828")


@dataclass
class DataArguments:
    data_path: str = field(
        default="output/demo_data_fix", metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    tune_vision_encoder: Optional[bool] = field(default=True)
    tune_vision_proj: Optional[bool] = field(default=True)
    tune_llm: Optional[bool] = field(default=True)
    tune_audio_encoder: Optional[bool] = field(default=True)
    tune_audio_proj: Optional[bool] = field(default=True)
    use_lora: Optional[bool] = field(default=False)
    max_slice_nums: Optional[int] = field(default=9)
    scale_resolution: Optional[int] = field(default=448)
    output_dir: str = field(
        default="output", metadata={"help": "Path to the output directory."}
    )


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = r"llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)"
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    lora_modules_to_save: str = ""
    lora_layer_replication: Optional[List[Tuple[int, int]]] = None
    lora_layers_to_transform: Optional[List[int]] = None
    lora_layers_pattern: Optional[str] = None


# from finetune.py
def load_model_from_pretrained(model_path, dtype=torch.bfloat16):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        _attn_implementation='flash_attention_2',
        trust_remote_code=True,
        torch_dtype=dtype)
    return model


# from finetune.py
def load_tokenizer_from_pretrained(model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True)
    return tokenizer


# from finetune.py
def make_supervised_data_module(
    data_args,
    processor,
    process_func,
    data_collator=None,
    max_length=2048,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = SupervisedDataset

    print("Loading data...")
    data_path = data_args.data_path
    if data_path.endswith(".cache"):
        data_paths = list(glob(data_path))
        train_dataset_list = []
        for data_path in data_paths:
            print("load dataset from disks: {}".format(data_path))
            dataset = load_from_disk(data_path)
            train_dataset_list.append(dataset)
        train_dataset = concatenate_datasets(train_dataset_list)
        train_dataset.set_format(type="torch")
    else:
        if os.path.isdir(data_path) and os.path.exists(data_path):
            # data_paths = list(glob(os.path.join(data_path, "*.jsonl")))
            data_paths = []
            for root, dirs, files in os.walk(data_path, followlinks=True):
                for file in files:
                    if file.endswith("jsonl"):
                        data_paths.append(os.path.join(root, file))
        else:
            if '#' in data_path:
                data_paths = data_path.split('#')
                data_paths_r = []
                for data_path_i in data_paths:
                    print("root: ")
                    print(data_path_i)
                    data_paths_p = list(glob(os.path.join(data_path_i, "*.jsonl")))
                    data_paths_r.extend(data_paths_p)
                data_paths = data_paths_r
            else:
                data_paths = [data_path]

        train_json = []
        for data_path in data_paths:
            data_json = open(data_path).readlines()
            print("data path: {}, nr {}".format(data_path, len(data_json)))
            train_json.extend(data_json)
        print("total nr: {}".format(len(train_json)))

        train_dataset = dataset_cls(
            train_json,
            processor,
            process_func
        )
    eval_dataset = None

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=partial(data_collator, max_length=max_length),
    )


class AudioModel(torch.nn.Module):

    def __init__(self, **kwargs):
        super(AudioModel, self).__init__()
        self.audio = AudioEncoder(**kwargs)
        
    def forward(self, audio_info):
        if isinstance(audio_info, Dict):
            audios = audio_info["input_audios"]
            audio_span_tokens = audio_info["audio_span_tokens"]
            input_audio_lengths = audio_info["input_audio_lengths"]
            audios = self.audio.encode(
                audios, input_audio_lengths, audio_span_tokens)
        else:
            audios = torch.concat([_["input_audios"] for _ in audio_info])
            input_audio_lengths = torch.concat([_["input_audio_lengths"] for _ in audio_info])
            audio_span_tokens = []
            for _ in audio_info:
                audio_span_tokens.extend(_['audio_span_tokens'])
            audios = self.audio.encode(
                audios, input_audio_lengths, audio_span_tokens)
        return audios


class VisionModel(torch.nn.Module):

    def __init__(self, config):
        super(VisionModel, self).__init__()
        self.config = config
        # print("\n")
        # print("Inside class VisionModel.__init__, after self.config = config...")
        # print(f"self.config: {self.config}")
        # print("\n")
        self.vpm = self.init_vision_module()

        # # self.vpm = self.vpm.type(torch.bfloat16)
        # print("\n")
        # print("After self.vpm = self.init_vision_module()...")
        # print(f"self.vpm.dtype: {self.vpm.dtype}")      # torch.float32, shoule be torch.bfloat16
        # print(f"self.vpm: {self.vpm}")
        # print("\n")
        self.resampler = self.init_resampler(self.config.hidden_size, self.vpm.embed_dim)

    def init_vision_module(self):
        if self.config._attn_implementation == 'flash_attention_2':
            self.config.vision_config._attn_implementation = 'flash_attention_2'
        else:
            # not suport sdpa
            self.config.vision_config._attn_implementation = 'eager'
        # print("\n")
        # print("Inside class VisionModel.init_vision_module(), before model = SiglipVisionTransformer(self.config.vision_config)")
        # print(f"self.config.vision_config: {self.config.vision_config}")
        # # self.config.vision_config: SiglipVisionConfig {
        # #     "attention_dropout": 0.0,
        # #     "hidden_act": "gelu_pytorch_tanh",
        # #     "hidden_size": 1152,
        # #     "image_size": 980,
        # #     "intermediate_size": 4304,
        # #     "layer_norm_eps": 1e-06,
        # #     "model_type": "siglip_vision_model",
        # #     "num_attention_heads": 16,
        # #     "num_channels": 3,
        # #     "num_hidden_layers": 27,
        # #     "patch_size": 14,
        # #     "transformers_version": "4.41.2"
        # # }
        # print("\n")

        # self.config.vision_config 没有问题
        model = SiglipVisionTransformer(self.config.vision_config)

        # workaround
        model = model.type(torch.bfloat16)

        # print("\n")
        # print("After model = SiglipVisionTransformer(self.config.vision_config)...")
        # print(f"model: {model}")
        # print(f"model.dtype: {model.dtype}")    # torch.float32
        # print("\n")
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]

        setattr(model, 'embed_dim', model.embeddings.embed_dim)
        setattr(model, 'patch_size', model.embeddings.patch_size)

        # # self.vpm got dtype issue(fp32), shoule be bf16
        # setattr(model, 'dtype', torch.bfloat16)     # AttributeError: can't set attribute 'dtype'

        # print("\n")
        # print("Inside class VisionModel.init_vision_module(), before return")
        # print(f"model: {model}")
        # print(f"model.dtype: {model.dtype}")    # torch.float32
        # print("\n")
        return model

    def init_resampler(self, embed_dim, vision_dim):
        return Resampler(
            num_queries=self.config.query_num,
            embed_dim=embed_dim,
            num_heads=embed_dim // 128,
            kv_dim=vision_dim,
            adaptive=True
        )

    def prepare_pixel_values(self, images):
        pixel_values = []
        device = self.vpm.device
        for image in images:
            image = (image.to(device).float() - 127.5) / 127.5
            pixel_values.append(image)
        return pixel_values

    def get_vision_embedding(self, pixel_values_list, tgt_sizes):
        device = self.vpm.device
        dtype = self.vpm.dtype
        # print("\n")
        # print("Inside def get_vision_embedding, after dtype = self.vpm.dtype...")
        # print(f"dtype: {dtype}")        # torch.float32 (from config.json "vision_config"."torch_dtype")
        # print("\n")

        all_pixel_values = []
        for pixel_value in pixel_values_list:
            all_pixel_values.append(pixel_value.flatten(end_dim=1).permute(1, 0))
            # all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

        max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])
        all_pixel_values = torch.nn.utils.rnn.pad_sequence(
            all_pixel_values, batch_first=True, padding_value=0.0)
        B, L, _ = all_pixel_values.shape
        all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

        patch_attn_mask = torch.zeros(
            (B, 1, max_patches), dtype=torch.bool, device=device)
        for i in range(B):
            patch_attn_mask[i, 0, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True

        vision_batch_size = self.config.vision_batch_size
        all_pixel_values = all_pixel_values.type(dtype)
        if B > vision_batch_size:
            hs = []
            for i in range(0, B, vision_batch_size):
                start_idx = i
                end_idx = i + vision_batch_size
                tmp_hs = self.vpm(
                    all_pixel_values[start_idx:end_idx],
                    patch_attention_mask=patch_attn_mask[start_idx:end_idx],
                    tgt_sizes=tgt_sizes[start_idx:end_idx]).last_hidden_state
                hs.append(tmp_hs)
            vision_embedding = torch.cat(hs, dim=0)
        else:
            # print("\n")
            # print("Before vision_embedding = self.vpm()...")
            # # print(f"type(all_pixel_values): {type(all_pixel_values)}")  # <class 'torch.Tensor'>
            # # print(f"type(patch_attn_mask): {type(patch_attn_mask)}")    # <class 'torch.Tensor'>
            # # print(f"type(tgt_sizes): {type(tgt_sizes)}")                # <class 'torch.Tensor'>
            # print(f"all_pixel_values.dtype: {all_pixel_values.dtype}")      # torch.float32, shoule be torch.bfloat16
            # print(f"patch_attn_mask.dtype: {patch_attn_mask.dtype}")        # torch.bool
            # print(f"tgt_sizes.dtype: {tgt_sizes.dtype}")                    # torch.int32
            # print("\n")
            vision_embedding = self.vpm(
                all_pixel_values,
                patch_attention_mask=patch_attn_mask,
                tgt_sizes=tgt_sizes).last_hidden_state

        return vision_embedding

    def forward(self, images, tgt_sizes):
        tgt_sizes = [
            tgt_size for tgt_size in tgt_sizes if isinstance(
                tgt_size, torch.Tensor)]
        tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)
        pixel_values = self.prepare_pixel_values(images)
        # print("\n")
        # print("Inside class VisionModel.forward, before embedding = self.get_vision_embedding(pixel_values, tgt_sizes)")
        # print(f"pixel_values: {pixel_values}")
        # for i in range(len(pixel_values)):
        #     print(f"i: {i}, pixel_values[i].dtype: {pixel_values[i].dtype}")    # torch.float32
        # print(f"tgt_sizes.dtype: {tgt_sizes.dtype}")        # torch.int32
        # print("\n")
        embedding = self.get_vision_embedding(pixel_values, tgt_sizes)
        # print("\n")
        # print("Before embedding = self.resampler(embedding, tgt_sizes)...")
        # print(f"embedding.dtype: {embedding.dtype}")            # torch.bfloat16(没问题)
        # print("\n")
        embedding = self.resampler(embedding, tgt_sizes)
        # print("\n")
        # print("After embedding = self.resampler(embedding, tgt_sizes)...")
        # print(f"embedding.dtype: {embedding.dtype}")
        # print("\n")
        return embedding


import sys
sys.path.append("/home/infiniai/linrongjian/Megatron-LM")
from megatron.core.transformer import MegatronModule, TransformerConfig

class MegRezOModel(MegatronModule):

    def __init__(
        self,
        transformer_config: TransformerConfig,
        megrezo_config: LlamaConfig,
    ) -> None:
        # transformer_config = TransformerConfig()

        args = get_args()

        config_dict = {field: getattr(transformer_config, field) for field in TransformerConfig.__dataclass_fields__}
        config_dict.update({
            "tensor_model_parallel_size": args.tensor_model_parallel_size,
            "pipeline_model_parallel_size": args.pipeline_model_parallel_size,
            "params_dtype": torch.bfloat16,
            "pipeline_dtype": torch.bfloat16,
            "autocast_dtype": torch.bfloat16,
            "perform_initialization": True,
            "async_tensor_model_parallel_allreduce": True,
            "bf16": True,
            "num_layers": 1,
            "hidden_size": 1024,
            "ffn_hidden_size": 4096,
            "num_attention_heads": 8,
            "num_query_groups": 8,
            "kv_channels": 8,
            "gated_linear_unit": True,
            "persist_layer_norm": True,
            "tp_comm_overlap": False,
            "pipeline_dtype": torch.bfloat16,
            "deallocate_pipeline_outputs": False,
            "apply_rope_fusion": True,
            "sequence_parallel": False,
            "hidden_dropout": 0.0,
            "attention_dropout": 0.1,
            "init_method_std": 0.02,
        })
        self.transformer_config = TransformerConfig(**config_dict)
        # print("\n")
        # print("After self.transformer_config = TransformerConfig(**config_dict)...")
        # print(f"self.transformer_config: {self.transformer_config}")
        # print("\n")

        super().__init__(config=self.transformer_config)
        
        # read config.json and update megrezo_config (use for build llm/vision/audio model later)
        with open('/data/megrez-o-3b-lrj/finetune/output/init_model/init_0828/config.json', 'r') as f:
            config_data = json.load(f)
        
        # TODO: update megrezo config based on args from pretrain_megrezo.sh
        self.megrezo_config = LlamaConfig(**config_data)

        # init llm/vision/audio model, all good
        self.megrezo_config.vision_config = SiglipVisionConfig(**self.megrezo_config.vision_config)
        self.llm = AutoModelForCausalLM.from_config(self.megrezo_config)
        self.vision = VisionModel(self.megrezo_config)
        self.audio = AudioModel(**self.megrezo_config.audio)

        self.tune_llm = True
        self.tune_vision = True
        self.tune_audio = True

        self.device = 'cuda:0'

        # from finetune.py - def train()
        self.audio.requires_grad_(False)
        self.audio.audio.proj.requires_grad_(True)

    def convert_to_device(self, mini_batch):
        for key in mini_batch:
            if isinstance(mini_batch[key], torch.Tensor):
                # mini_batch[key] = mini_batch[key].to(
                #     self.device)
                mini_batch[key] = mini_batch[key].to(
                    torch.cuda.current_device())
                # mini_batch[key] = mini_batch[key].to(
                #     'cpu')
            if isinstance(mini_batch[key], list):
                return_value = []
                for value in mini_batch[key]:
                    if isinstance(value, torch.Tensor):
                        # value = value.to(self.device)
                        value = value.to(torch.cuda.current_device())
                        # value = value.to('cpu')
                    return_value.append(value)
                mini_batch[key] = return_value

        return mini_batch

    def compose_embeddings(self, data):
        mini_batch = self.convert_to_device(data)
        position_ids = mini_batch["position_ids"]
        input_ids = mini_batch["input_ids"]
        msgs_image = mini_batch["msgs_image"]
        msgs_audio = mini_batch["msgs_audio"]
        bounds_image = mini_batch["bounds_image"]
        bounds_audio = mini_batch["bounds_audio"]
        tgt_sizes = mini_batch["tgt_sizes"]
        if position_ids.dtype != torch.int64:
            position_ids = position_ids.long()

        # print("\n")
        # print("Before embeddings_text = self.llm.model.embed_tokens(input_ids)...")
        # print(f"input_ids: {input_ids}")        # device='cuda:0', dtype=torch.int32 (这个没问题)
        # print("\n")
        embeddings_text = self.llm.model.embed_tokens(input_ids)
        input_embeds = embeddings_text
        device = input_embeds.device
        if len(msgs_image) > 0:
            embeddings_image = prepare_image_embeddings(
                self.vision, msgs_image, tgt_sizes)
            input_embeds = insert_image_embeddings(
                embeddings_text,
                embeddings_image,
                bounds_image)
        else:
            if self.training and self.tune_vision:
                dummy_image = torch.zeros((3, 14, 3584), dtype=torch.float32, device=device)
                tgt_sizes = torch.tensor([[16, 16]], dtype=torch.int64, device=device)
                embeddings_image = prepare_image_embeddings(
                    self.vision, [dummy_image], tgt_sizes)
                input_embeds += embeddings_image[0].sum() * 0.0

        if len(msgs_audio) > 0:
            embeddings_audio = prepare_audio_embeddings(
                self.audio, msgs_audio)
            input_embeds = insert_audio_embeddings(
                embeddings_text,
                embeddings_audio,
                bounds_audio)
        else:
            if self.training and self.tune_audio:
                dummy_audio = torch.zeros((1, 128, 3000), dtype=torch.float32, device=device)
                dummp_audio_lengths = torch.tensor([[125,  62]], dtype=torch.int32, device=device)
                dummp_span_tokens = [64]
                msgs_audio = [
                    {
                        "input_audios": dummy_audio,
                        "input_audio_lengths": dummp_audio_lengths,
                        "audio_span_tokens": dummp_span_tokens,
                    }
                ]
                embeddings_audio = prepare_audio_embeddings(
                    self.audio, msgs_audio)
                input_embeds += embeddings_audio[0].sum() * 0.0

        return input_ids, input_embeds, position_ids

    def set_input_tensor(self, input_tensor) -> None:
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for megrezo'

        self.input_tensor = input_tensor
        # print("\n")
        # print("Inside class MegRezOModel.set_input_tensor...")
        # print(f"self.input_tensor: {self.input_tensor}")            # [None]
        # print("\n")

    def forward(self, data, **kwargs):
        # 'bounds_image' and 'bounds_audio' got dtype issue(fp32), shoule be bf16
        if data['bounds_image'].dtype == torch.float32:
            data['bounds_image'] = data['bounds_image'].to(torch.bfloat16)
        if data['bounds_audio'].dtype == torch.float32:
            data['bounds_audio'] = data['bounds_audio'].to(torch.bfloat16)

        # print("\n")
        # print("Before input_ids, input_embeds, position_ids = self.compose_embeddings(data)...")
        # for key, value in data.items():
        #     # key: input_ids, value.shape: torch.Size([1, 461]), value.dtype: torch.int32
        #     # key: position_ids, value.shape: torch.Size([1, 461]), value.dtype: torch.int64
        #     # key: attention_mask, value.shape: torch.Size([1, 461]), value.dtype: torch.bool
        #     # key: labels, value.shape: torch.Size([1, 461]), value.dtype: torch.int64
        #     # key: msgs_image, value.shape: None, value.dtype: None
        #     # key: bounds_image, value.shape: torch.Size([4, 3]), value.dtype: torch.int64
        #     # key: msgs_audio, value.shape: None, value.dtype: None
        #     # key: bounds_audio, value.shape: torch.Size([0]), value.dtype: torch.float32
        #     # key: tgt_sizes, value.shape: None, value.dtype: None
        #     print(f"key: {key}, value.shape: {value.shape if isinstance(value, torch.Tensor) else None}, value.dtype: {value.dtype if isinstance(value, torch.Tensor) else None}")
        #     if isinstance(value, torch.Tensor):
        #         print(f"value.requires_grad: {value.requires_grad}")            # all False
        #         value.requires_grad_()                                          # RuntimeError: only Tensors of floating point dtype can require gradients
        #     # if key == 'msgs_image':                   # 4
        #     #     print(f"len(msgs_image): {len(value)}")
        # print("\n")

        input_ids, input_embeds, position_ids = \
                self.compose_embeddings(data)
        return self.llm.forward(
            input_ids=None,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
            **kwargs
        )


        # deal with self.input_tensor(what's the relationship between self.input_tensor and data?)
        # ...


def compute_loss(model, inputs):
    if "labels" in inputs:
        labels = inputs.pop("labels")
    else:
        labels = None

    outputs = model(data=inputs, use_cache=False)
    # print("\n")
    # print("Inside def compute_loss, after outputs = model(data=inputs, use_cache=False)...")
    # print(f"outputs: {outputs}")
    # print(f"outputs.loss: {outputs.loss}")      # None
    # print("\n")

    loss = outputs.loss

    return loss


# if __name__ == "__main__":

#     # init MegRezOModel
#     megrezo_model = MegRezOModel(
#         transformer_config=TransformerConfig,
#         megrezo_config=LlamaConfig,
#     )

#     # if not set, weight would be on 'cpu'
#     megrezo_model = megrezo_model.to('cuda:0')

#     # get default arguments
#     parser = transformers.HfArgumentParser(
#         (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
#     )
#     (
#         model_args,
#         data_args,
#         training_args,
#         lora_args,
#     ) = parser.parse_args_into_dataclasses()

#     # load model, tokenizer, processor using transformers
#     tokenizer = load_tokenizer_from_pretrained(model_args.model_name_or_path)
#     processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

#     utils_path = os.path.join(model_args.model_name_or_path, "utils.py")
#     spec = importlib.util.spec_from_file_location("utils", utils_path)
#     utils = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(utils)

#     data_module = make_supervised_data_module(
#         data_args=data_args,
#         processor=processor,
#         process_func=utils.process_data,
#         data_collator=data_collator,
#         max_length=training_args.model_max_length,
#     )

#     # current_seed = torch.initial_seed()
#     # print(f"Current seed: {current_seed}")      # 16021959027474135583 vs 42
#     torch.manual_seed(42)

#     sampler=RandomSampler(data_module['train_dataset'])

#     train_dataloader = DataLoader(
#         dataset=data_module['train_dataset'],
#         sampler=sampler,
#         pin_memory=True,
#         num_workers=4,
#         collate_fn=data_collator,
#         persistent_workers=True,
#     )
    
#     for step, inputs in enumerate(train_dataloader):
#         # print("\n")
#         # print(f"step: {step}, inputs: {inputs}")
#         # for key, value in inputs.items():
#         #     print(f"key: {key}, value.shape: {value.shape if isinstance(value, torch.Tensor) else None}, value.dtype: {value.dtype if isinstance(value, torch.Tensor) else None}")
#         # print("\n")

#         if "labels" in inputs:
#             labels = inputs.pop("labels")
#         else:
#             labels = None

#         res = megrezo_model(data=inputs)
#         loss = compute_loss(model=megrezo_model, inputs=inputs)
#         print("\n")
#         print(f"step: {step}, class MegRezOModel.forward result: {res}, loss: {loss}")
#         print("\n")
