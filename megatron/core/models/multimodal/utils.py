import base64
import io
import math
import os
from copy import deepcopy
from functools import lru_cache
from subprocess import PIPE, CalledProcessError, Popen, run
from typing import Optional, Union

import numpy as np
import redis
import requests
import torch
import torch.nn.functional as F
from PIL import Image

# from .audio import (get_T_after_cnn, load_audio, load_bytesio_audio,
#                     log_mel_spectrogram, pad_or_trim)

Image.MAX_IMAGE_PIXELS = None
ip_addr = os.environ.get("REDIS_IP", "10.208.15.192")
port = os.environ.get("REDIS_PORT", "6379")
r = redis.StrictRedis(host=ip_addr, port=port, db=0)

filter_string = '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'
filter_array = np.load(io.BytesIO(base64.b64decode(filter_string)))


def exact_div(x, y):
    assert x % y == 0
    return x // y


# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 128
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES,
                     HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE,
                              N_SAMPLES_PER_TOKEN)  # 20ms per audio token


def get_T_after_cnn(L_in, dilation=1):
    for (padding, kernel_size, stride) in eval("[(1,3,1)] + [(1,3,2)] "):
        L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
        L_out = 1 + L_out // stride
        L_in = L_out
    return L_out


def load_bytesio_audio(content, sr: int = SAMPLE_RATE):
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0", "-i", "pipe:", "-f", "s16le",
        "-ac", "1", "-acodec", "pcm_s16le", "-ar",
        str(sr), "pipe:"
    ]
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, bufsize=-1)
    out, _ = p.communicate(input=content)
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(dim=axis,
                                       index=torch.arange(length,
                                                          device=array.device))

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array,
                          [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


def trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(dim=axis,
                                       index=torch.arange(length,
                                                          device=array.device))
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)
    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = N_MELS) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels in [80, 128], f"Unsupported n_mels: {n_mels}"
    return torch.from_numpy(filter_array[f"mel_{n_mels}"]).to(device)
    # with np.load(
    #     os.path.join(os.path.dirname(__file__), "mel_filters.npz") # todo
    #     # os.path.join("assets", "mel_filters.npz")
    # ) as f:
    #     return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = N_MELS,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio,
                      N_FFT,
                      HOP_LENGTH,
                      window=window,
                      return_complex=True)
    magnitudes = stft[..., :-1].abs()**2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def process_audio(text, use_md5=False):
    if isinstance(text, list):
        audio_urls = text
    else:
        audio_urls = [text]
    if len(audio_urls) > 0:
        audios, audio_lens, audio_span_tokens = [], [], []
        for audio_path in audio_urls:
            if use_md5:
                audio_bytes = r.get(audio_path)
                audio = load_bytesio_audio(audio_bytes)
            elif audio_path.startswith("http://") or audio_path.startswith(
                    "https://"):  # http
                data = bytes(requests.get(audio_path, stream=True).content)
                audio = load_bytesio_audio(data)
            else:
                audio = load_audio(audio_path)
            L = (audio.shape[0] if audio.shape[0] <= 480000 else 480000
                 )  # max_length < 30s
            mel_len = L // 160
            audio = pad_or_trim(audio.flatten())
            mel = log_mel_spectrogram(audio)
            audio_len_after_cnn = get_T_after_cnn(mel_len)
            audio_token_num = (audio_len_after_cnn - 2) // 2 + 1
            audio_len = [audio_len_after_cnn, audio_token_num]
            audios.append(mel)
            audio_lens.append(audio_len)
            audio_span_tokens.append(audio_token_num + 2)  # add audio bos eos
        input_audio_lengths = torch.IntTensor(audio_lens)
        input_audios = torch.stack(audios, dim=0)
        return {
            "input_audios": input_audios,
            "input_audio_lengths": input_audio_lengths,
            "audio_span_tokens": audio_span_tokens,
            "audio_urls": audio_urls
        }
    else:
        return None


def process_image(text):
    if isinstance(text, str):
        image_list = [text]
    if isinstance(text, list):
        image_list = text
    return_images = [
        np.array(Image.open(image_path).convert("RGB")).astype(np.uint8)
        for image_path in image_list
    ]
    return return_images


def process_image_md5(md5):
    image_bytes = r.get(md5)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = np.array(image)
    return image


def load_inputs(input_msgs):
    r_input_msgs = []
    for msg in input_msgs:
        content = msg['content']
        if isinstance(content, dict):
            r_msg = deepcopy(msg)
            r_content = deepcopy(content)
            if "image" in content:
                r_content["image"] = process_image(content['image'])
            if "audio" in content:
                if content["audio"] is None:
                    content.pop("audio")
                    r_content.pop("audio")
                else:
                    r_content['audio'] = process_audio(content["audio"])
            r_msg['content'] = r_content
        elif isinstance(content, str):
            r_msg = msg
        else:
            raise NotImplementedError(
                "Content Parse Error: {}".format(content))
        r_input_msgs.append(r_msg)

    return r_input_msgs


def prepare_raw_msgs_minicpm_v26(input_msgs, processor):

    tokenizer = processor.tokenizer
    msgs_text, msgs_image, msgs_audio = [], [], []

    image_id, audio_id = 0, 0
    for msg in input_msgs:
        role = msg["role"]
        content = msg["content"]

        # assert role in ["user", "assistant"]

        if isinstance(content, str):
            r_msg = {"role": role, "content": content}
        elif isinstance(content, dict):
            r_content = ""
            if "image" in content:
                image_list = content["image"]
                for c in image_list:
                    msgs_image.append(c)
                    r_content += "(<image>./</image>)\n"
                    # r_content_list.append(r_content)
                    image_id += 1
            if "audio" in content:
                audio_id += 1
                r_content += f"Audio {audio_id}: "
                audio = content["audio"]
                msgs_audio.append(content["audio"])
                nr_audios = len(audio["audio_span_tokens"])
                for idx in range(nr_audios):
                    audio_query_num = audio["audio_span_tokens"][idx]
                    r_content += tokenizer.audio_start + tokenizer.unk_token * audio_query_num + tokenizer.audio_end + '\n'
                    # r_content_list.append(r_content)
            elif "text" in content:
                r_content += content["text"]
                # r_content_list.append(r_content)
            if "instruction" in content:
                r_content += content["instruction"]
                # r_content_list.append(r_content)
            # r_content = "\n".join(r_content_list)
            r_msg = {"role": role, "content": r_content}
        else:
            raise NotImplementedError(
                "Content Parse Error: {}".format(content))
        msgs_text.append(r_msg)

    return msgs_text, msgs_image, msgs_audio


def prepare_bounds_audio(tokenizer, input_ids):
    audio_bos_ids = torch.where(input_ids == tokenizer.audio_start_id)
    audio_eos_ids = torch.where(input_ids == tokenizer.audio_end_id)
    bounds_audio = torch.stack([audio_bos_ids[0], audio_eos_ids[0]], 1)
    return bounds_audio


def prepare_image_embeddings(vision_model, msgs_image, tgt_sizes):
    embeddings = vision_model(msgs_image, tgt_sizes)
    return embeddings


def prepare_audio_embeddings(audio_model, msgs_audio):
    embeddings = audio_model(msgs_audio)
    return embeddings


def compose_embeddings(text_embeddings, image_embeddings, image_bounds,
                       audio_embeddings, audio_bounds):

    for idx in range(len(image_embeddings)):
        bid = image_bounds[idx][0]
        start_id = image_bounds[idx][1]
        end_id = image_bounds[idx][2]
        embedding = image_embeddings[idx]
        text_embeddings[bid, start_id + 1:end_id] = embedding

    for idx in range(len(audio_embeddings)):
        bid = audio_bounds[idx][0]
        start_id = audio_bounds[idx][1]
        end_id = audio_bounds[idx][2]
        embedding = audio_embeddings[idx]
        text_embeddings[bid, start_id + 1:end_id] = embedding

    return text_embeddings


def insert_audio_embeddings(text_embeddings, inserted_embeddings,
                            inserted_bounds):

    inserted_bounds = inserted_bounds.long()

    for idx in range(len(inserted_embeddings)):
        bid = inserted_bounds[idx][0]
        start_id = inserted_bounds[idx][1]
        end_id = inserted_bounds[idx][2]
        embedding = inserted_embeddings[idx]
        text_embeddings[bid, start_id + 1:end_id] = embedding

    return text_embeddings


def insert_image_embeddings(text_embeddings, inserted_embeddings,
                            inserted_bounds):

    inserted_bounds = inserted_bounds.long()
    for idx in range(len(inserted_embeddings)):
        bid = inserted_bounds[idx][0]
        start_id = inserted_bounds[idx][1]
        end_id = inserted_bounds[idx][2]
        embedding = inserted_embeddings[idx]
        text_embeddings[bid, start_id:end_id] = embedding

    return text_embeddings


def slice_image(image,
                max_slice_nums=9,
                scale_resolution=448,
                patch_size=14,
                never_split=False):
    original_size = image.size
    original_width, original_height = original_size
    log_ratio = math.log(original_width / original_height)
    ratio = original_width * original_height / \
        (scale_resolution * scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)

    source_image = None
    best_grid = None
    patches = []

    if multiple <= 1 or never_split:
        # dont need to slice, upsample
        best_size = find_best_resize(original_size,
                                     scale_resolution,
                                     patch_size,
                                     allow_upscale=True)
        source_image = image.resize(best_size, Image.Resampling.BICUBIC)
    else:
        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i == 1 or i > max_slice_nums:
                continue
            candidate_split_grids_nums.append(i)

        # source image, down-sampling and ensure divided by patch_size
        best_resize = find_best_resize(original_size, scale_resolution,
                                       patch_size)
        source_image = image.copy().resize(best_resize,
                                           Image.Resampling.BICUBIC)
        candidate_grids = []

        # find best grid
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])
                m += 1

        best_grid = [1, 1]
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error

        refine_size = get_refine_size(original_size,
                                      best_grid,
                                      scale_resolution,
                                      patch_size,
                                      allow_upscale=True)

        refine_image = image.resize(refine_size, Image.Resampling.BICUBIC)
        patches = split_to_patches(refine_image, best_grid)

    return source_image, patches, best_grid


def ensure_divide(length, patch_size):
    return max(round(length / patch_size) * patch_size, patch_size)


def find_best_resize(original_size,
                     scale_resolution,
                     patch_size,
                     allow_upscale=False):
    width, height = original_size
    if (width * height > scale_resolution * scale_resolution) or allow_upscale:
        r = width / height
        height = int(scale_resolution / math.sqrt(r))
        width = int(height * r)
    best_width = ensure_divide(width, patch_size)
    best_height = ensure_divide(height, patch_size)
    return (best_width, best_height)


def get_refine_size(original_size,
                    grid,
                    scale_resolution,
                    patch_size,
                    allow_upscale=False):
    width, height = original_size
    grid_x, grid_y = grid

    refine_width = ensure_divide(width, grid_x)
    refine_height = ensure_divide(height, grid_y)

    grid_width = refine_width / grid_x
    grid_height = refine_height / grid_y

    best_grid_size = find_best_resize(
        (grid_width, grid_height),
        scale_resolution,
        patch_size,
        allow_upscale=allow_upscale,
    )

    refine_size = (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)

    return refine_size


def split_to_patches(image, grid):
    patches = []
    width, height = image.size
    grid_x = int(width / grid[0])
    grid_y = int(height / grid[1])

    for i in range(0, height, grid_y):
        images = []
        for j in range(0, width, grid_x):
            box = (j, i, j + grid_x, i + grid_y)
            patch = image.crop(box)
            images.append(patch)
        patches.append(images)

    return patches


def get_grid_placeholder(tokenizer, grid, query_num):
    image_placeholder = (tokenizer.image_start +
                         tokenizer.unk_token * query_num + tokenizer.image_end)

    cols = grid[0]
    rows = grid[1]
    slices = []
    for i in range(rows):
        lines = []
        for j in range(cols):
            lines.append(image_placeholder)
        slices.append("".join(lines))
    slice_placeholder = tokenizer.slice_start + \
        "\n".join(slices) + tokenizer.slice_end
    return slice_placeholder


def reshape_by_patch(image_tensor, patch_size):
    """
    :param image_tensor: shape [3, H, W]
    :param patch_size:
    :return: [3, patch_size, HW/patch_size]
    """
    patches = torch.nn.functional.unfold(image_tensor,
                                         (patch_size, patch_size),
                                         stride=(patch_size, patch_size))

    patches = patches.reshape(image_tensor.size(0), patch_size, patch_size, -1)
    patches = patches.permute(0, 1, 3, 2).reshape(image_tensor.size(0),
                                                  patch_size, -1)
    return patches


def prepare_labels(tokenizer, input_ids, padding_value=-100):
    # <|role_start|>assistant<|role_end|> 后面的内容才是需要算loss的部分
    def find_start_header_idxs():
        start_header_tokens = tokenizer.encode(
            "<|role_start|>assistant<|role_end|>", add_special_tokens=False)
        start_header_idxs = np.where(input_ids == start_header_tokens[-1])[0]
        kept_start_header_idxs = []
        for start_header_idx in start_header_idxs:
            keep = True
            for i in range(1, len(start_header_tokens)):
                if start_header_tokens[-(i + 1)] != input_ids[start_header_idx
                                                              - i]:
                    keep = False
                    break
            if keep:
                kept_start_header_idxs.append(start_header_idx)
        return kept_start_header_idxs

    start_header_idxs = find_start_header_idxs()
    end_header_idxs = np.where(input_ids == tokenizer.eos_token_id)[0]
    label_mask = np.zeros_like(input_ids, dtype=np.bool_)

    def find_next_greater_number(lst, num):
        next_greater = None
        for n in lst:
            if n > num:
                if next_greater is None or n < next_greater:
                    next_greater = n
        return next_greater

    nr_tokens = len(input_ids)
    for start_head_idx in start_header_idxs:
        start_idx = start_head_idx + 1
        end_idx = find_next_greater_number(end_header_idxs, start_head_idx)
        end_idx = min(end_idx + 1, nr_tokens)
        label_mask[start_idx:end_idx] = True

    labels = torch.ones(input_ids.shape[0] + 1) * padding_value
    labels[:input_ids.shape[0]] = input_ids
    labels[:input_ids.shape[0]][~label_mask] = padding_value
    labels = labels[1:]
    # print("labels max: {}, label min: {}".format(
    #     labels[labels != padding_value].max(),
    #     labels[labels != padding_value].min()))
    return labels.long()


def process_data(processor, input_msgs, training=True):

    input_msgs = load_inputs(input_msgs)
    msgs_text, msgs_image, msgs_audio = prepare_raw_msgs_minicpm_v26(
        input_msgs, processor)
    prompt = processor.tokenizer.apply_chat_template(
        msgs_text, tokenize=False, add_generation_prompt=False)

    res = processor([prompt], [msgs_image], return_tensors="pt")

    input_ids = res["input_ids"].reshape(-1)
    attention_mask = res["attention_mask"].reshape(-1)
    pixel_values = res["pixel_values"][0]
    image_sizes = res["image_sizes"][0]
    image_bound = res["image_bound"][0]
    tgt_sizes = res["tgt_sizes"][0]

    bounds_audio = prepare_bounds_audio(processor.tokenizer, input_ids)

    if training:
        labels = prepare_labels(processor.tokenizer, input_ids)

    data_dict = dict()
    data_dict["input_ids"] = input_ids
    data_dict["position_ids"] = torch.arange(input_ids.size(0)).long()
    data_dict["attention_mask"] = attention_mask
    if training:
        data_dict["labels"] = labels

    data_dict["bounds_image"] = image_bound
    data_dict["msgs_image"] = pixel_values
    data_dict["image_sizes"] = image_sizes

    data_dict["bounds_audio"] = bounds_audio
    data_dict["msgs_audio"] = msgs_audio
    data_dict["tgt_sizes"] = tgt_sizes

    return data_dict
