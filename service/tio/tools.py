from transformers import AutoTokenizer, AutoImageProcessor
from functools import partial
from PIL import Image as PILImage
import cv2
import numpy as np
import json

from transformers import PreTrainedTokenizer
from transformers.image_processing_utils import BaseImageProcessor
import os
import re
import torch

def check_chatml(sequence):
    assert isinstance(sequence, (list, tuple)) \
        and isinstance(sequence[0], dict) \
        and "role" in sequence[0] and "type" in sequence[0] and "content" in sequence[0], \
        f"input must be ChatML format, like [{{'role': 'user', 'type': 'text', 'content': 'hello!'}}]. Got {sequence}"


def encode_bbox_to_string(bbox: list) -> str:
    """把bbox转换为字符串. e.g. [0, 0, 0.5, 0.5] -> "<bin_0><bin_0><bin_500><bin_500>" """
    bbox = np.asarray(bbox).reshape(-1) * 1000
    assert len(bbox) == 4, f"Got: {bbox}"
    bbox = np.clip(bbox.astype('int'), 0, 999)
    sbbox = "".join([f"<bin_{i}>" for i in bbox])
    return sbbox

def auto_eos(labels, tokenizer, min_eos_length=0):
    for i, tokens in enumerate(labels.input_ids):
        eos_mask = (tokens == tokenizer.eos_token_id)
        eos_index = torch.where(eos_mask)[0][0].item()
        if eos_index < min_eos_length and not tokenizer.decode(tokens[1]).startswith("<bin_"):
            labels.input_ids[i][eos_index] = tokenizer.pad_token_id
            labels.attention_mask[i][eos_index] = 0
    return labels

def show_mask(image: PILImage.Image, bboxes=None, masks=None, show_id=False, text_size=1) -> PILImage.Image:
    import colorsys
    colors = [tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i * 1.0 / 36, 1, 1)) for i in range(18)] 
    size = image.size
    image = np.array(image)
    if bboxes is not None:
        bboxes = np.array(bboxes).reshape(-1, 4)
        for k, bbox in enumerate(bboxes):
            bbox = (np.asarray(bbox) * np.asarray([*size, *size])).astype(int)
            # print(image.shape, tuple(bbox[:2]), tuple(bbox[2:]), tuple(colors[k]))
            image = cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:]), tuple(colors[k]), thickness=2)
        if show_id:
            for k, bbox in enumerate(bboxes):
                bbox = (np.asarray(bbox) * np.asarray([*size, *size])).astype(int)
                image = cv2.putText(image, str(k), tuple(bbox[:2] + np.array([2, 28 * text_size])), cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 255, 255), 2, cv2.LINE_AA)
                image = cv2.putText(image, str(k), tuple(bbox[:2] + np.array([2, 28 * text_size])), cv2.FONT_HERSHEY_SIMPLEX, text_size, tuple(colors[k%len(colors)]), 1, cv2.LINE_AA)

    if masks is not None:
        for k, mask in enumerate(masks):
            mask_color = (mask[..., None] * colors[k%len(colors)][:3]).astype(np.uint8)
            image_mask = cv2.addWeighted(mask_color, 0.5, image * mask[..., None], 0.5, 0)
            image = cv2.add(image * ~mask[..., None], image_mask)
    return PILImage.fromarray(image)

def convert_chatbot_to_chatml(history, system_prompt: str = None, roles=['user', 'assistant']):
    """把chatbot格式转换为chatml格式. e.g. [["hello", "world"]] -> [{"role": ..., "content": ...}]"""
    def encode_item(item: dict):
        if isinstance(item['content'], PILImage.Image):
            item['type'] = "image"
        elif isinstance(item['content'], tuple):
            item['content'] = PILImage.open(item['content'][0])
            item['type'] = "image"
        elif isinstance(item['content'], str):
            item['type'] = "text"
            try:
                c = json.loads(item['content'])
                if isinstance(c, list) and len(c) == 4:
                    item['content'] = c
                    item['type'] = "bbox"
            except json.JSONDecodeError:
                pass
        else:
            raise NotImplementedError(f"Got {item}")
        return item

    sequence: list[dict] = []
    for i, h in enumerate(history):
        for j, s in enumerate(h):
            if s is not None:
                sequence += [encode_item({"role": roles[j], "content": s})]
    if system_prompt is not None:
        sequence = [{
            "role": "system",
            "type": "text",
            "content": system_prompt
        }] + sequence
    return sequence

def collate_chatml_by_tio(
    sequences: list[list[dict]], 
    tokenizer: PreTrainedTokenizer, 
    image_processor: BaseImageProcessor,
    max_src_length: int = None, 
    max_tgt_length: int = None, 
    truncation: bool = True,
    padding: bool = "max_length", 
    return_tensors: str = "pt",
    output_labels: bool = True,
    min_eos_length: int = None,
    **args
):
    """collate_fn
    e.g. [[{"role": "user", "type": "image", "content": image}, ...], ...]
     -> {"input_ids": ..., "attention_mask": ..., "pixel_values": ...}
    """
    max_src_length = args.pop("max_length", max_src_length)
    # 添加recall的candidates适配
    addition_dict = {}
    has_candidates = sequences[0][-1].get("candidates") is not None
    if has_candidates:
        candidates = [s[-1]['candidates'] for s in sequences]
        for ss in candidates:
            assert isinstance(ss, list), sequences
        # candidate_labels = []
        # candidate_attention_masks = []
        # for c in candidates:
        #     l = tokenizer(
        #         [cc.lower() for cc in c], 
        #         max_length=64, 
        #         truncation=truncation, 
        #         padding="max_length", 
        #         return_tensors=return_tensors,
        #         **args
        #     )
        #     candidate_labels += [l.input_ids]
        #     candidate_attention_masks += [l.attention_mask]
        # sequences = [s[:-1] for s in sequences]
        # addition_dict = dict(
        #     candidate_labels=torch.stack(candidate_labels),
        #     candidate_attention_masks=torch.stack(candidate_attention_masks)
        # )

        batch_size = len(candidates)
        _candidates = [cc.lower() for c in candidates for cc in c]
        l = tokenizer(
            _candidates, 
            max_length=max_tgt_length, 
            truncation=truncation, 
            padding=padding, 
            return_tensors=return_tensors,
            **args
        )
        _, dim = l.input_ids.shape
        addition_dict = dict(
            candidate_labels=l.input_ids.reshape([batch_size, -1, dim]),
            candidate_attention_masks=l.attention_mask.reshape([batch_size, -1, dim])
        )
        sequences = [s[:-1] for s in sequences]

    def encode_sequence_stage_1(sequence):
        sequence = [{
            **item, 
            "type": "text", 
            "content": encode_bbox_to_string(item['content'])
        } if item['type'] == 'bbox' else item for item in sequence]
        return sequence

    # split images and texts
    def encode_sequence_stage_2(sequence: list[dict]) -> tuple[PILImage.Image, str]:
        check_chatml(sequence)
        images: list[PILImage.Image] = [item['content'] for item in sequence if item['type'] == 'image']
        image = images[0]
        texts = [item for item in sequence if item['type'] == 'text']
        text = tokenizer.apply_chat_template(texts, tokenize=False)
        assert len(images) == 1 and len(texts) >= 0, f"Got {images}, {texts}"
        return image, text

    check_chatml(sequences[0])
    sequences = [encode_sequence_stage_1(s) for s in sequences]
    if output_labels:
        for s in sequences:
            assert s[-1]['role'] == "assistant", f"the role of last sequence item must be `assistant` when `output_labels=True`. Got {s}"
        src_sequences = [s[:-1] for s in sequences]
        tgt_sequences = [s[-1]['content'] for s in sequences]
    else:
        src_sequences = sequences
        tgt_sequences = None
    # encode src_sequences
    input_images, src_sequences = zip(*[encode_sequence_stage_2(s) for s in src_sequences])
    input_images = [i.convert("RGB") for i in input_images]
    input_images = image_processor(input_images, return_tensors=return_tensors)
    inputs = tokenizer(
        [t.lower() for t in src_sequences], 
        max_length=max_src_length, 
        truncation=truncation, 
        padding=padding, 
        return_tensors=return_tensors,
        **args
    )
    # encode tgt_sequences
    from transformers import BatchEncoding
    if tgt_sequences:
        labels = tokenizer(
            [t.lower() for t in tgt_sequences],
            max_length=max_tgt_length, 
            truncation=truncation, 
            padding=padding, 
            return_tensors=return_tensors,
            **args
        )
        labels = auto_eos(labels, tokenizer, min_eos_length=min_eos_length or 0)
        inputs = BatchEncoding({
            "input_ids": inputs.input_ids,
            "patch_images": input_images.pixel_values,
            "attention_mask": labels.attention_mask[..., :-1],
            "decoder_input_ids": labels.input_ids[..., :-1],
            "labels": labels.input_ids[..., 1:],
            **addition_dict
        })
    else:
        inputs = BatchEncoding({
            "input_ids": inputs.input_ids,
            "patch_images": input_images.pixel_values,
        })
    return inputs

def decode_string_to_bbox(sbbox: str, using_default=True):
    """把字符串解码成bbox.
    例如: <bin_12><bin_23><bin_34><bin_45> -> [0.12, 0.23, 0.34, 0.45].
    using_default (默认为True) : 当无法解析输入字符串时, 若为True则输出[0, 0, 1, 1], 否则输出None.
    """
    assert isinstance(sbbox, str), f"sbbox must be str, got {type(sbbox)}. {sbbox}"
    sbbox = re.findall(r"<bin_(\d+)>", sbbox)
    bbox = [int(s) / 1000 for s in sbbox][:4]
    if len(bbox) < 4:
        if using_default:
            bbox = [0, 0, 1, 1]
        else:
            return None
    return {"type": "bbox", "content": np.clip(bbox, 1e-5, 1 - 1e-5).tolist()}


def decode_string(string: str, dtype: str = None):
    """尝试自动判断字符串类型并进行解码转换.
    """
    if dtype == "bbox":
        string = decode_string_to_bbox(string)
    elif dtype == "text":
        pass
    else:
        string = decode_string_to_bbox(string, using_default=False) or string
    return string