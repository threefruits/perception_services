from transformers import Pipeline
from tqdm import tqdm
import torch
import numpy as np


from tools import convert_chatbot_to_chatml, collate_chatml_by_tio, decode_string


"""
# Usage Examples

model = AutoModel.from_pretrained("./models/TiO-v207", trust_remote_code=True).half().cuda()
tokenizer = robollm.TiOTokenizer.from_pretrained("./models/TiO")
image = Image.open("3rd_part/output.png")
bbox = [0.161, 0.661, 0.314, 0.775]

dialog = [image, "Can you help me with?"]
pp_questioner = TiOQuestionerPipeline(model=model, tokenizer=tokenizer, device="cuda", torch_dtype=torch.float16)
pp_questioner([dialog], temperature=0.7, top_p=0.95)[0]

dialog = [image, bbox, ""]  # 空字符串用于角色转换
pp_oracle = TiOOraclePipeline(model=model, tokenizer=tokenizer, device="cuda", torch_dtype=torch.float16)
pp_oracle([dialog], temperature=0.7, top_p=0.95)[0]

dialog = [image, "Can you help me with?", "what is it?", "the balloon on the far left."]
pp_groudning = TiOGuesserPipeline(model=model, tokenizer=tokenizer, device="cuda", torch_dtype=torch.float16)
bbox_pred = pp_groudning([dialog], temperature=0.7, top_p=0.95)[0]

from robollm.data.visualizer import show_mask
show_mask(image, bbox_pred)
"""


class TiOPipeline(Pipeline):
    system_prompt: str = None
    def _sanitize_parameters(self, **kwargs):
        def sp(input_args, used_args):
            return {k: v for k, v in input_args.items() if k in used_args}
        kw1, kw2, kw3 = {}, {}, {}
        for k in list(kwargs.keys()):
            if k in ["padding", "truncation", "max_src_length", "system_prompt"]:
                kw1[k] = kwargs.pop(k)
            if k in ["stream_output", "max_length", "do_sample", "num_beams", "num_beam_groups", \
                "temperature", "top_k", "top_p", "diversity_penalty", "skip_special_tokens", "decode_dtype"]:
                kw2[k] = kwargs.pop(k)
            if k in ["decode_dtype"]:
                kw3[k] = kwargs.pop(k)
        if len(kwargs):
            print(f"Unuse args: {list(kwargs.keys())}")
        return (kw1, kw2, kw3)

    def preprocess(self, inputs: dict, padding=False, truncation=False, max_src_length=512, system_prompt=False):
        # inputs: {"sequence": [image, ..., ...]}
        if system_prompt is False:
            system_prompt = self.system_prompt
        print(inputs)
        sequence = [convert_chatbot_to_chatml(inputs['chatbot'], system_prompt)]
        # print("*" * 20)
        # print(sequence)
        # print("*" * 20)
        model_inputs = collate_chatml_by_tio(
            sequence, self.tokenizer, 
            self.image_processor, 
            padding=padding, 
            truncation=truncation, 
            max_length=max_src_length, 
            return_tensors="pt", 
            output_labels=False
        )
        model_inputs["patch_images"] = model_inputs["patch_images"].to(self.torch_dtype)
        return model_inputs

    def _forward(
        self, 
        model_inputs, 
        stream_output=False, 
        max_length=256,
        do_sample=True, 
        num_beams=1,
        num_beam_groups=1,
        temperature=1.0, 
        top_k=50, 
        top_p=0.9,
        diversity_penalty=0.0, 
        skip_special_tokens=True,
        decode_dtype=None,
        **args
    ):
        generate_kwargs = dict(
            model_inputs,
            max_length=max_length,
            do_sample=do_sample,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            diversity_penalty=diversity_penalty,
        )
        if not stream_output:
            model_outputs = self.model.generate(**generate_kwargs)[0]
            model_outputs = self.tokenizer.decode(model_outputs, skip_special_tokens=skip_special_tokens)
            return decode_string(model_outputs, dtype=decode_dtype)
        else:
            from threading import Thread
            from transformers import TextIteratorStreamer
            streamer = TextIteratorStreamer(self.tokenizer, timeout=10.0, skip_prompt=False, skip_special_tokens=skip_special_tokens)
            t = Thread(target=self.model.generate, kwargs=dict(generate_kwargs, streamer=streamer))
            t.start()
            def model_outputs():
                outputs = []
                for text in streamer:
                    outputs.append(text)
                    yield decode_string("".join(outputs), dtype=decode_dtype)
            return model_outputs

    def postprocess(self, model_outputs):
        return model_outputs


class TiOQuestionerPipeline(TiOPipeline):
    system_prompt: str = "Be helpful, and ask for clarification if unsure."


class TiOOraclePipeline(TiOPipeline):
    system_prompt: str = "Be helpful, and answering questions."


class TiOGuesserPipeline(TiOPipeline):
    system_prompt: str = "Be helpful, and output bounding box only."
    grounding_prompt: str = "**TASK** Output Bounding Box."
    def preprocess(self, inputs, **args):
        inputs['chatbot'] = inputs['chatbot'] + [[self.grounding_prompt, None]]
        return super().preprocess(inputs, **args)
    def _forward(self, model_inputs, **args):
        decode_dtype = args.pop("decode_dtype", "bbox")
        return super()._forward(model_inputs, decode_dtype=decode_dtype, **args)


class TiOEnd2endPipeline(TiOPipeline):
    pp_questioner: TiOQuestionerPipeline = None
    pp_oracle: TiOOraclePipeline = None
    pp_guesser: TiOGuesserPipeline = None

    def __init__(
        self, 
        *,
        questioner_model=None,
        oracle_model=None,
        guesser_model=None,
        **args
    ):
        q_args = {**args, "model": questioner_model or args.get('model')}
        o_args = {**args, "model": oracle_model or args.get('model')}
        g_args = {**args, "model": guesser_model or args.get('model')}
        self.pp_questioner = TiOQuestionerPipeline(**q_args)
        self.pp_oracle = TiOOraclePipeline(**o_args)
        self.pp_guesser = TiOGuesserPipeline(**g_args)
        super().__init__(**args)

    def preprocess(self, inputs, **args):
        # inputs = {"image": ..., "bbox": ...}
        return inputs

    def _forward(self, model_inputs, **args):
        # {'index': 0, 'image': <PIL.Image.Image image mode=RGB size=480x640 at 0x7FA89F708790>, 'dialog': [['i want a bottle of wine.', 'which bottle do you want?'], ['i want the bottle with the brown label.', 'is it the bottle on the left?'], ['yes.', 'ok, i see.']], 'bbox': [0.0, 0.3203, 0.3042, 0.8812], 'bbox_pred': [0.011, 0.172, 0.323, 0.88]}
        if isinstance(model_inputs, dict):
            model_inputs = [model_inputs]
        history = end2end_dialog(self.pp_questioner, self.pp_oracle, self.pp_guesser, dataset=model_inputs, disable_tqdm=True)[0]
        return history

    def postprocess(self, model_outputs: list[dict], **args):
        if isinstance(model_outputs, list):
            model_outputs = sum(model_outputs, [])
        return model_outputs


def end2end_dialog(pp_questioner, pp_oracle, pp_guesser, dataset, disable_tqdm=False, max_turn=8, **args):
    def prepare_for_oracle(dialog, bbox):
        dialog_flatten = sum(dialog, [])
        image, dialog_flatten = dialog[0][0], dialog_flatten[1:] + [None]
        dialog = [[image, None], [bbox, None]] + \
            list(zip(dialog_flatten[::2], dialog_flatten[1::2]))
        return dialog

    # 准备变量
    references = []
    predictions = []
    history: list[dict] = []
    # 迭代数据集
    t = tqdm(dataset, disable=disable_tqdm)
    for index, data in enumerate(t):
        image = data.get("image")
        bbox = data.get("bbox")
        # 构造对话并交替执行
        dialog = data.get("dialog") or [[image, None]]
        # dialog += [[None, "which one do you want?"]]
        # dialog += [[None, "which object do you want?"]]
        dialog += [[None, "can you describe the object you want?"]]
        dialog += [[None, "what can i help you today?"]]
        for _ in range(max_turn):
            # 判断第一次是否只需要计算ai的回应.
            dialog_oracle = prepare_for_oracle(dialog, f"{bbox}")
            dialog += [[None, None]]
            dialog[-1][0] = pp_oracle({"chatbot": dialog_oracle}, temperature=0.5, top_p=0.9, max_length=64)
            # dialog[-1][0] = pp_oracle({"chatbot": dialog_oracle}, top_p=None, top_k=None, num_beams=5, do_sample=False, max_length=64)
            dialog[-1][1] = pp_questioner({"chatbot": dialog}, temperature=0.5, top_p=0.9, max_length=64)
            # dialog[-1][1] = pp_questioner({"chatbot": dialog}, top_p=None, top_k=None, num_beams=5, do_sample=False, max_length=64)
            # 结束条件: 最后一轮ai的回应以句号结尾; 连续两轮ai的回应以句号结尾; 输出的话发生重复
            assistant_prev, assistant_now = dialog[-2][1], dialog[-1][1]
            if len(assistant_now) < 2:
                assistant_now = assistant_now + " maybe?"
            elif (assistant_now[-1:] in [".", "!"] and assistant_now[:2] in ["ok", "ye", "hm", 'su', "go"]):
                break
            if isinstance(assistant_prev, str):
                if len(assistant_prev) < 2:
                    assistant_now = assistant_now + " maybe?"
                elif assistant_now[-1] in [".", "!"] and assistant_prev[-1] in [".", "!"]:
                    break
                if assistant_now == assistant_prev or dialog[-1][0] == dialog[-2][0]:  # 重复对话了
                    # dialog = dialog[:-1]
                    break
    
        bbox_pred = pp_guesser({"chatbot": dialog}, top_p=None, top_k=None, num_beams=5, do_sample=False, max_length=16)["content"]
        references += [bbox]
        predictions += [bbox_pred]
        # 保存历史样本
        history += [{
            "index": index, 
            "image": dialog[0][0],
            "dialog": dialog[1:], 
            "bbox": np.round(bbox, 4).tolist(), 
            "bbox_pred": np.round(bbox_pred, 4).tolist(),
        }]
    return history
