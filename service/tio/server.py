from functools import partial
from pprint import pp
from PIL import Image
import os
import time
import requests
import json
import gradio as gr
import transformers
import torch
import tempfile
import numpy as np


from tools import show_mask

from pipeline import TiOPipeline
# import sinvig


models = {}
SERVICE_ROOT = os.path.dirname(os.path.abspath(__file__))
tokenizer = transformers.AutoTokenizer.from_pretrained(os.path.join(SERVICE_ROOT, "models/TiO"))
image_processor = transformers.AutoImageProcessor.from_pretrained(os.path.join(SERVICE_ROOT,"models/TiO"))


INSTALL_DOC = """
Installing SAM ...

Plan:
1. "cd /tmp && git clone https://github.com/facebookresearch/segment-anything.git"
2. "pip install setuptools==58.0"
3. "sudo pip install /tmp/segment-anything"
4. "pip install opencv-python pycocotools matplotlib onnxruntime onnx"
5. "cd /tmp && wget -c https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
"""


predictor = None
def get_sam_predictor():
    global predictor
    if predictor is None:
        # 自动配置sam环境
        try:
            # 初始化sam
            from segment_anything import sam_model_registry, SamPredictor
            sam_checkpoint = "/tmp/sam_vit_h_4b8939.pth"
            model_type = "vit_h"
            device = "cuda"
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)
            predictor = SamPredictor(sam)
        except:
            print(INSTALL_DOC)
            commands = [
                "cd /tmp && git clone https://github.com/facebookresearch/segment-anything.git",
                "pip install setuptools==58.0",
                "sudo pip install /tmp/segment-anything",
                "pip install opencv-python pycocotools matplotlib onnxruntime onnx",
                "cd /tmp && wget -c https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            ]
            for c in commands:
                os.system(c)
            predictor = get_sam_predictor()
    return predictor


def gradio_sam_api(image, bbox):
    predictor = get_sam_predictor()

    predictor.set_image(np.asarray(image))
    input_box = np.asarray(json.loads(bbox), dtype=float).reshape(4) * np.asarray([*image.size, *image.size])
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    
    return json.dumps(masks[0].astype(np.uint8).tolist())


def my_tio_pipeline(chatbot, model_id, **args):
    # lazy load
    if model_id not in models:
        models[model_id] = transformers.AutoModel.from_pretrained(model_id, trust_remote_code=True).half().cuda()
    tio_pipeline = TiOPipeline(
        model=models[model_id], tokenizer=tokenizer, image_processor=image_processor, device="cuda", torch_dtype=torch.float16)
    print("Pipeline Input: ", chatbot, type(chatbot))
    gens = tio_pipeline({"chatbot": chatbot}, stream_output=True, **args)()
    for output in gens:
        if isinstance(output, dict) and output['type'] == "bbox":
            assert isinstance(chatbot[0][0], tuple), f"input an image first please, got {type(image)}."
            image = Image.open(chatbot[0][0][0]).convert("RGB")
            bbox = output['content']
            bbox_string = "_".join([f"{b:.03f}" for b in bbox]) + "_"
            with tempfile.NamedTemporaryFile(prefix=bbox_string, suffix=".jpg", delete=False) as f:
                image_w_bbox = show_mask(image, [bbox])
                image_w_bbox.save(f)
            output = (f.name,)
        yield output

# ========================================
# □ ChatGPT API
# ========================================
def chatgpt_api(dialog, url, model, system_prompt="", **options):
    # 构造数据
    messages = []
    if system_prompt:
        messages += [{"role": "system", "content": system_prompt}]
    for turn in dialog:
        for i, message in enumerate(turn):
            if isinstance(message, str):
                messages += [{"role": ["user", "assistant"][i%2], "content": message}]
    if messages[-1]["role"] == "assistant":
        messages = messages[:-1]  # 最后一句的assistant应该为空
    json_data = {
        'model': model,
        "stream": True,
        'messages': messages,
        **options
    }
    # streaming 通讯
    s = requests.Session()
    responses = ""
    with s.post(url, headers={'Content-Type': 'application/json'}, json=json_data, stream=True) as resp:
        for c in resp.iter_lines():
            c = c.decode('utf-8')
            if c.startswith('data: '):
                c = c[6:]
            if not c or c == '[DONE]':
                continue
            c = json.loads(c)
            if c.get('object') == 'error':
                raise RuntimeError(c.get("message"))
            try:
                if c['choices'][0]["finish_reason"]:
                    continue
                responses += c['choices'][0]['delta']['content']
                yield responses
            except KeyError:
                pass


MY_OPTIONS = [
    {
        "name": "TiO | v20-v2",
        "pipeline": partial(my_tio_pipeline, model_id=os.path.join(SERVICE_ROOT, "models/TiO-pretrain-240117_pretrain-v20-invig2-from-1"))
    },
    {
        "name": "TiO | v20-v1",
        "pipeline": partial(my_tio_pipeline, model_id=os.path.join(SERVICE_ROOT, "models/TiO-pretrain-240117_pretrain-v20-invig1-from-0"))
        
    },
    {
        "name": "TiO | v20-v0",
        "pipeline": partial(my_tio_pipeline, model_id=os.path.join(SERVICE_ROOT, "models/TiO-pretrain-240117_pretrain-v22-invig0-sft"))
    },
    # {
    #     "name": "TiO | v2_sft",
    #     "pipeline": partial(my_tio_pipeline, model_id=os.path.join(SERVICE_ROOT, "models/TiO-v8-sft-invig2"))
    # },
    # {
    #     "name": "TiO | v2_pt",
    #     "pipeline": partial(my_tio_pipeline, model_id=os.path.join(SERVICE_ROOT, "models/TiO-v5-pt-invig2"))
    # },
    # {
    #     "name": "TiO | v1_sft",
    #     "pipeline": partial(my_tio_pipeline, model_id=os.path.join(SERVICE_ROOT, "models/TiO-v11-sft-invig1"))
    # },
    # {
    #     "name": "TiO | v1_pt",
    #     "pipeline": partial(my_tio_pipeline, model_id=os.path.join(SERVICE_ROOT, "models/TiO-v212-pt-invig1"))
    # },
    # {
    #     "name": "TiO | v0_sft",
    #     "pipeline": partial(my_tio_pipeline, model_id=os.path.join(SERVICE_ROOT, "models/TiO-v209-sft-invig0"))
    # },
    # {
    #     "name": "TiO | v0_pt",
    #     "pipeline": partial(my_tio_pipeline, model_id=os.path.join(SERVICE_ROOT, "models/TiO-v191-pt-invig0"))
    # },
    # {
    #     "name": "TiO | v10_0",
    #     "pipeline": partial(my_tio_pipeline, model_id=os.path.join(SERVICE_ROOT, "logs/lightning_logs/pretrain-v10-invig0/version_1/TiO-pretrain-v10-invig0"))
    # },
    # {
    #     "name": "TiO | v10_1",
    #     "pipeline": partial(my_tio_pipeline, model_id=os.path.join(SERVICE_ROOT, "logs/lightning_logs/pretrain-v10-invig1-from-0/version_0/TiO-pretrain-v10-invig1-from-0"))
    # },
]


class MyChatInterface(gr.ChatInterface):
    def _delete_prev_fn(self, history):
        history, message, _ = super()._delete_prev_fn(history)
        if not isinstance(message, str):
            message = ""
        return history, message or "", history


# ========================================
# □ Gradio Server
# ========================================
def main(host: str = "0.0.0.0", port: int = 55392):
    model_names = [i['name'] for i in MY_OPTIONS]
    lookup_options = lambda name, key: [i for i in MY_OPTIONS if i['name'] == name][0][key]
    def echo_streaming(message, history, system_prompt, model_name, temperature=1.0, max_tokens=512, top_p=0.9, **args):
        assert len(message), "Empty message."
        if isinstance(history, str):
            history = json.loads(history)
        assert len(history) and isinstance(history[0][0], tuple), "Please upload an image first."
        history = history + [[message, None]]
        options = {
            "temperature": temperature,
            "max_length": max_tokens,
            "top_p": top_p,
            **args
        }
        pp = lookup_options(model_name, 'pipeline')
        yield from pp(history, system_prompt=system_prompt, **options)

    def echo_streaming_api(image, history, model_name):
        assert model_name in ["TiO | v20-v0", "TiO | v20-v1", "TiO | v20-v2"]
        if isinstance(history, str):
            history = json.loads(history)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            image.save(f)
        message = history[-1][0]
        history = [[(f.name,), "how can i help you today?"]] + history[:-1]
        system_prompt = ""
        temperature = 0.5
        max_tokens = 64
        print("log\n", message, history, system_prompt, model_name, temperature, max_tokens)
        for o in echo_streaming(message, history, system_prompt, model_name, temperature):
            if isinstance(o, tuple):
                bbox_pred = os.path.split(o[0])[1].split("_")
                bbox_pred = [float(bbox_pred[i]) for i in range(4)]
                o = json.dumps(bbox_pred)
            yield o

    with gr.Blocks(theme=gr.themes.Soft(), analytics_enabled=False) as demo:
        system_prompt = gr.Textbox('Be helpful, and ask for clarification if unsure.', label="Optional System Prompt", render=False)
        model_choices = gr.Dropdown(model_names, value=model_names[0], label="Model Backend", render=False)
        temperature = gr.Slider(0.0, 4.0, 1.0, label="Temperature", render=False)
        max_tokens = gr.Slider(8, 4096, 512, label="Max new tokens", render=False)
        top_p = gr.Slider(0.01, 1.0, 0.9, label="Top-p (nucleus sampling)", render=False)

        gr.Markdown(f"<h1 style='text-align: center; margin-bottom: 1rem'>{'Chatbot Demo'}</h1>")
        chatbot = gr.Chatbot(height=500)
        chatbot.render = lambda: chatbot
        with gr.Group():
            with gr.Row():
                textbox = gr.Textbox(placeholder="Typing something ...", scale=8, lines=4)
                textbox.render = lambda: textbox
                with gr.Column(scale=1, min_width=120):
                    grounding_btn = gr.Button("Grounding", variant="secondary", scale=1)
                with gr.Column(scale=1, min_width=120):
                    upload_btn = gr.UploadButton("Upload", variant="secondary", file_types=["image", "video"], scale=1)
                    submit_btn = gr.Button("Submit", variant="primary", scale=1)
                    submit_btn.render = lambda: submit_btn

        chat_interface = MyChatInterface(
            echo_streaming, 
            chatbot=chatbot,
            textbox=textbox,
            submit_btn=submit_btn,
            stop_btn=False,
            analytics_enabled=False,
            additional_inputs_accordion_name="Options",
            additional_inputs=[system_prompt, model_choices, temperature, max_tokens, top_p],
        )
        upload_btn.upload(lambda f, h: [*h, [(f.name,), None]], [upload_btn, chatbot], [chatbot])\
            .then(lambda x: x, [chatbot], [chat_interface.chatbot_state])
        grounding_btn.click(lambda: "**TASK** Output Bounding Box.", [], [textbox])

        # for tio_api
        api_image = gr.Image(visible=False, label="image", type="pil")
        api_text = gr.Textbox(visible=False, label="history")
        api_text.change(
            lambda image, history, model_name: list(echo_streaming_api(image, history, model_name))[-1], 
            [api_image, api_text, gr.Textbox(visible=False, label="model_name")], 
            [api_text],
            api_name="tio_api"
        )

        # for sam_api
        sam_bbox = gr.Textbox(label='bbox', visible=False)
        sam_mask = gr.Textbox(label='mask', visible=False)
        sam_mask.change(gradio_sam_api, [api_image, sam_bbox], [sam_mask], api_name="sam_api")
    demo.queue(10).launch(server_name=host, server_port=port, show_error=True, share=True)


if __name__ == "__main__":
    import fire
    fire.Fire(main)