import json
import os

import numpy as np
import redis
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

Image.MAX_IMAGE_PIXELS = None
ip_addr = os.environ.get("REDIS_IP", "10.208.17.244")
port = os.environ.get("REDIS_PORT", "16397")
r = redis.StrictRedis(host=ip_addr, port=port, db=0)


def data_collator(examples, padding_value=0, max_length=4096):
    # print("\n")
    # print("Inside def data_collator, in the beginning...")
    # print(f"padding_value: {padding_value}")
    # print(f"max_length: {max_length}")
    # for i in range(len(examples)):
    #     for key, value in examples[i].items():
    #         print(f"key: {key}, value.shape: {value.shape if isinstance(value, torch.Tensor) else None}, value.dtype: {value.dtype if isinstance(value, torch.Tensor) else None}")
    # # print(f"examples: {examples}")
    # print("\n")


    # print("max_length: {}".format(max_length))
    # print("\n")
    # print("Inside /data/megrez-o-3b-lrj/finetune/dataset.py def data_collator")
    # print(f"type(examples): {type(examples)}")
    # print(f"examples: {examples}")
    # print("\n")

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
        # print("\n")
        # print(f"len(example[\"msgs_image\"]): {len(example['msgs_image'])}")
        # print(f"example[\"msgs_image\"]: {example['msgs_image']}")
        # print("\n")
        nr_image = len(example["msgs_image"])
        for idx in range(nr_image):
            image = example["msgs_image"][idx]
            bound_image = np.array(example["bounds_image"][idx])
            # if nr_image == 5:
            #     print("\n")
            #     print("After bound_image = np.array(example[\"bounds_image\"][idx])...")
            #     # image.shape: torch.Size([3, 14, 14504]), bound_image.shape: (2,)
            #     print(f"image.shape: {image.shape}, bound_image.shape: {bound_image.shape}")
            #     print("\n")
            bound_image_s = bound_image[0]
            bound_image_e = bound_image[1]
            # if nr_image == 5:
            #     print("\n")
            #     print("Before if bound_image_s <= max_length and bound_image_e <= max_length:")
            #     # bound_image_s: 29, bound_image_e: 93, max_length: 4096
            #     print(f"bound_image_s: {bound_image_s}, bound_image_e: {bound_image_e}, max_length: {max_length}")
            #     print("\n")
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

    # print("input ids shape: {}".format(input_ids.shape))
    # print("bounds_audio shape: {}".format(bounds_audio.shape))
    # print("bounds_imagei shape: {}".format(bounds_image.shape))

    # return {
    #     "input_ids": input_ids,
    #     "position_ids": position_ids,
    #     "attention_mask": attention_mask,
    #     "labels": targets,
    #     "msgs_image": msgs_image,
    #     "bounds_image": bounds_image,
    #     "msgs_audio": msgs_audio,
    #     "bounds_audio": bounds_audio,
    #     "tgt_sizes": tgt_sizes
    # }
    res = {
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

    # print("Inside def data_collator, before return...")
    # for key, value in res.items():
    #     print(f"key: {key}, value.shape: {value.shape if isinstance(value, torch.Tensor) else None}, value.dtype: {value.dtype if isinstance(value, torch.Tensor) else None}")
    # # print(f"res: {res}")
    # print("\n")
    return res


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        raw_data,
        processor,
        process_func
    ):
        super(SupervisedDataset, self).__init__()
        self.raw_data_list = raw_data
        self.processor = processor
        self.process_func = process_func

    def __len__(self):
        return len(self.raw_data_list)

    def raw_data(self, i):
        raw_data_item = self.raw_data_list[i]
        raw_data_item = json.loads(raw_data_item)
        if "json_md5" in raw_data_item:
            md5 = raw_data_item["json_md5"]
            data_str = r.get(md5)
            raw_data_item = json.loads(data_str)
        return raw_data_item

    def __getitem__(self, i):
        data_dict = self.raw_data(i)
        input_msgs = data_dict["conversations"]
        ret = self.process_func(self.processor, input_msgs, training=True)
        # while True:
        #     try:
        #         data_dict = self.raw_data(i)
        #         input_msgs = data_dict["conversations"]
        #         ret = self.process_func(self.processor, input_msgs, training=True)
        #         break
        #     except Exception as e:
        #         print("index {} failed".format(i))
        #         print(self.raw_data(i))
        #         i = np.random.randint(0, len(self.raw_data_list))
        return ret