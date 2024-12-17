import os
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler

from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.core import mpu
from megatron.core.models.multimodal.megrezo_model import MegRezOModel
from megatron.core.enums import ModelType
from megatron.training import get_args, pretrain

from megatron.core.models.multimodal.utils import process_data
from megatron.core.models.multimodal.dataset import SupervisedDataset, data_collator

from transformers.models.llama.configuration_llama import LlamaConfig
import transformers
from transformers import AutoProcessor
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM


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


def model_provider(
    pre_process=True, post_process=True, add_encoder=True, add_decoder=True, parallel_output=True
) -> MegRezOModel:
    # args = get_args()

    model = MegRezOModel(
        transformer_config=TransformerConfig,
        megrezo_config=LlamaConfig,
    )

    return model


# from finetune.py
def data_collator(examples, padding_value=0, max_length=4096):

    def trim_and_pad(seq, batch_first, padding_value):
        return pad_sequence(
            [s[:max_length] for s in seq],
            batch_first=True,
            padding_value=padding_value,
        )

    input_ids = trim_and_pad(
        [example["input_ids"] for example in examples],
        batch_first=True,
        padding_value=padding_value,
    )
    position_ids = trim_and_pad(
        [example["position_ids"] for example in examples],
        batch_first=True,
        padding_value=padding_value,
    )

    targets = trim_and_pad(
        [example["labels"] for example in examples],
        batch_first=True,
        padding_value=-100,
    )
    attention_mask = trim_and_pad(
        [example["attention_mask"] for example in examples],
        batch_first=True,
        padding_value=padding_value,
    )

    msgs_image, bounds_image_list = [], []
    for bid, example in enumerate(examples):
        nr_image = len(example["msgs_image"])
        for idx in range(nr_image):
            image = example["msgs_image"][idx]
            bound_image = np.array(example["bounds_image"][idx])
            bound_image_s = bound_image[0]
            bound_image_e = bound_image[1]
            if bound_image_s <= max_length and bound_image_e <= max_length:
                msgs_image.append(image)
                bounds_image_list.append([bid, bound_image_s, bound_image_e])
    bounds_image = torch.tensor(bounds_image_list)

    msgs_audio, bounds_audio_list = [], []
    for bid, example in enumerate(examples):
        nr_audio = len(example["msgs_audio"])
        for idx in range(nr_audio):
            audio = example["msgs_audio"][idx]
            bound_audio = np.array(example["bounds_audio"][idx])
            bound_audio_s = bound_audio[0]
            bound_audio_e = bound_audio[1]
            if bound_audio_s <= max_length and bound_audio_e <= max_length:
                msgs_audio.append(audio)
                bounds_audio_list.append([bid, bound_audio_s, bound_audio_e])
    bounds_audio = torch.tensor(bounds_audio_list)
    tgt_sizes = [example["tgt_sizes"] for example in examples]

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "labels": targets,
        "msgs_image": msgs_image,
        "bounds_image": bounds_image,
        "msgs_audio": msgs_audio,
        "bounds_audio": bounds_audio,
        "tgt_sizes": tgt_sizes
    }


# from finetune.py
def make_supervised_data_module(
    data_path,
    processor,
    process_func,
    data_collator=None,
    max_length=2048,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = SupervisedDataset

    print("Loading data...")
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


def train_valid_test_datasets_provider(train_val_test_num_samples):
    args = get_args()
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    data_dict = make_supervised_data_module(
        data_path=args.vlm_data_path,
        processor=processor,
        process_func=process_data,
        data_collator=data_collator,
        max_length=args.model_max_length,
    )

    train_ds = data_dict['train_dataset']
    valid_ds = data_dict['eval_dataset']
    test_ds = None

    print("\n")
    print("Inside pretrain_megrezo.py def train_valid_test_datasets_provider, before return...")
    print(f"train_ds: {train_ds}")
    print(f"valid_ds: {valid_ds}")
    print(f"test_ds: {test_ds}")
    print("\n")
    return train_ds, valid_ds, test_ds


def get_batch(data_iterator):
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    
    return data


def loss_func(data, output_tensor: torch.Tensor):
    # print("\n")
    # print("Inside pretrain_megrezo.py - def loss_func(), in the beginning...")
    # print(f"type(output_tensor): {type(output_tensor)}")            # <class 'transformers.modeling_outputs.CausalLMOutputWithPast'>
    # print(f"output_tensor.logits.shape: {output_tensor.logits.shape if output_tensor.logits is not None else None}")            # torch.Size([1, 199, 122880])
    # print(f"output_tensor.past_key_values.shape: {output_tensor.past_key_values.shape if output_tensor.past_key_values is not None else None}") # None
    # print(f"output_tensor.hidden_states.shape: {output_tensor.hidden_states.shape if output_tensor.hidden_states is not None else None}")       # None
    # print(f"output_tensor.attentions.shape: {output_tensor.attentions.shape if output_tensor.attentions is not None else None}")                # None
    # print("\n")
    labels = data.pop("labels")
    loss_fct = nn.CrossEntropyLoss()
    logits = output_tensor.logits.view(-1, 122880).contiguous()       # config.vocab_size=122880
    labels = labels.view(-1).long().contiguous()
    # print("\n")
    # print("After labels = labels.view(-1).long().contiguous()...")
    # print(f"labels.shape: {labels.shape}")                              # torch.Size([199])
    # print("\n")
    labels = labels.to(logits.device)
    loss = loss_fct(logits, labels)
    # print("\n")
    # print(f"loss: {loss}")
    # print("\n")

    # print("\n")
    # print("Inside pretrain_megrezo.py def loss_func(), before return...")
    # print(f"output_tensor: {output_tensor}")      # loss=None, logits={}, past_key_values=None, hidden_states=None
    # print("\n")
    # output_tensor=torch.tensor(0.0)
    # output_tensor.requires_grad_(True)
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

    return (
        loss,
        {'megrezo loss': (reporting_loss)},
    )


def forward_step(data_iterator, model: MegRezOModel):
    # data = model_provider().data_module['data_collator']

    data = get_batch(data_iterator)

    res = model(data)

    # TODO: deal with loss

    return res, partial(loss_func, data)


# # see arguments.py
# def add_megrezo_extra_args(parser):
#     group = parser.add_argument_group(title='megrezo specific arguments')
#     group.add_argument("--lr-scheduler-type", action="store_true", default="cosine")
#     group.add_argument("--model-name-or-path", action="store_true",
#         default="/home/infiniai/linrongjian/Megatron-LM/megatron/core/models/multimodal/output/init_model/init_0828"
#     )
#     group.add_argument("--model-max-length", action="store_true", default=4096)
#     return parser


def megrezo_embedding_ranks(pp_ranks):
    args = get_args()

    epp = args.encoder_pipeline_model_parallel_size

    last_rank = pp_ranks[-1]
    if len(pp_ranks) == 1 or pp_ranks[epp] == last_rank:
        return [last_rank]
    else:
        return [pp_ranks[epp], last_rank]


def megrezo_position_embedding_ranks(pp_ranks):
    args = get_args()

    epp = args.encoder_pipeline_model_parallel_size

    last_rank = pp_ranks[-1]
    if len(pp_ranks) == 1:
        return [last_rank]
    else:
        return [pp_ranks[epp]]


if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_and_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'VlmTokenizer'},
        # extra_args_provider=add_megrezo_extra_args,
        extra_args_provider=None,
        get_embedding_ranks=megrezo_embedding_ranks,
        get_position_embedding_ranks=megrezo_position_embedding_ranks,
    )