import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bs4 import BeautifulSoup
import re

class GemmaTranslator:
    def __init__(self, model_name="kaitchup/translategemma-4b-it-NVFP4", device="cuda"):
        print(f"Loading {model_name} on {device}...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # NVFP4 usually requires special loading, but we can fall back to standard BitsAndBytes if needed
        # kaitchup's NVFP4 models might require bitsandbytes / accelerate.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("Model loaded successfully!")

    def _convert_lang_name(self, lang_code):
        """Map language code to language name for the prompt."""
        lang_map = {
            "en": "English",
            "vi": "Vietnamese",
            "ko": "Korean",
            "zh": "Chinese",
            "fr": "French",
            "de": "German",
            "ja": "Japanese",
            "es": "Spanish",
            "ru": "Russian",
            "th": "Thai",
            # Add more mappings as needed based on your dataset
        }
        # Fallback to uppercase code if not found in map
        return lang_map.get(lang_code, lang_code.upper())

    def _format_prompt(self, text, src_lang, tgt_lang):
        src_name = self._convert_lang_name(src_lang)
        tgt_name = self._convert_lang_name(tgt_lang)
        
        prompt = (
            f"user\nYou are a professional {src_name} ({src_lang}) to {tgt_name} ({tgt_lang}) translator. "
            f"Your goal is to accurately convey the meaning and nuances of the original {src_name} text "
            f"while adhering to {tgt_name} grammar, vocabulary, and cultural sensitivities.\n"
            f"Produce only the {tgt_name} translation, without any additional explanations or commentary. "
            f"Please translate the following {src_name} text into {tgt_name}:\n\n\n"
            f"{text}\nmodel\n"
        )
        return prompt

    def translate_batch(self, texts, src_lang, tgt_lang):
        """Basic translation for a list of texts."""
        if not texts:
            return []
            
        translated_texts = []
        for text in texts:
            prompt = self._format_prompt(text, src_lang, tgt_lang)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False, # Use greedy decoding for translation
                    temperature=0.0,
                    top_p=1.0,
                )
                
            # Extract the generated portion (after the prompt)
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            
            translated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            translated_texts.append(translated_text)
            
        return translated_texts

    def translate_html_safe(self, html_text, src_lang, tgt_lang):
        """
        Duyệt cây DOM, nhặt đúng Text đi dịch rồi nhét lại vào HTML.
        Đảm bảo tag HTML, XML không bao giờ bị phá hỏng.
        """
        soup = BeautifulSoup(html_text, 'html.parser')
        nodes_to_translate = []
        
        # Chỉ lấy text có nghĩa, bỏ qua khoảng trắng hoặc tags chức năng
        for node in soup.find_all(string=True):
            if node.parent.name not in ['style', 'script', 'head', 'title', 'meta', '[document]']:
                text_strip = node.strip()
                # Skip nếu chỉ là số hoặc ký tự đặc biệt (tiết kiệm thời gian dịch)
                if text_strip and any(c.isalpha() for c in text_strip):
                    nodes_to_translate.append((node, text_strip))
        
        if not nodes_to_translate:
             return html_text
             
        texts = [t for n, t in nodes_to_translate]
        translated_texts = self.translate_batch(texts, src_lang, tgt_lang)
        
        for (node, orig_text), trans_text in zip(nodes_to_translate, translated_texts):
             # Chỉ thay thế phần text có nghĩa, bảo toàn dấu cách thừa xung quanh Node
             # Do escape HTML có thể gặp vấn đề, string replace có giới hạn
             new_string = node.replace(orig_text, trans_text)
             node.replace_with(new_string)
             
        return str(soup)

    def translate(self, source, src_lang, tgt_lang, kind="plain"):
        """Hàm API gọi dịch chính."""
        # 1. Early-return: Không paraphrase nếu source == target
        if src_lang == tgt_lang:
            return source

        # 2. Xử lý tùy theo plain text hay HTML
        if kind == "html" or ("<" in source and ">" in source):
            return self.translate_html_safe(source, src_lang, tgt_lang)
        else:
            return self.translate_batch([source], src_lang, tgt_lang)[0]
