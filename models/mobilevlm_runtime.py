import sys
import os
import time
import torch
from PIL import Image
import config

from mobilevlm_official.mobilevlm.model.mobilevlm import load_pretrained_model
from mobilevlm_official.mobilevlm.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from mobilevlm_official.mobilevlm.conversation import conv_templates, SeparatorStyle
from mobilevlm_official.mobilevlm.utils import tokenizer_image_token, process_images

class MobileVLMRuntime:
    def __init__(self, model_path="mtgv/MobileVLM_V2-1.7B", device=None):
        print(f"🔄 Loading model from {model_path} ...")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path,
            device_map="auto",
            device=self.device
        )
        self.model.eval()
        print("✅ Model loaded.")

    def run(self, image_path: str, prompt: str, conv_mode="v1", temperature=0, top_p=None, num_beams=1, max_new_tokens=48):
        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model.config).to(self.model.device, dtype=torch.float16)

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        input_ids = tokenizer_image_token(full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        
        # Stopping Criteria 준비
        from mobilevlm_official.mobilevlm.utils import KeywordsStoppingCriteria
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        start_time = time.time()
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        duration = time.time() - start_time

        input_token_len = input_ids.shape[1]
        output_text = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()

        if output_text.endswith(stop_str):
            output_text = output_text[: -len(stop_str)]

        return output_text, duration