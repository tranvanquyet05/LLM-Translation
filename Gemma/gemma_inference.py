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
        
        # Prompt được tinh chỉnh tỉ mỉ dựa trên yêu cầu từ file log benchmark:
        # Bắt buộc dịch chính xác, giữ nguyên số (digit_preservation), 
        # HTML tags, format và dịch đúng ngôn ngữ đích mà không paraphrase bừa bãi.
        prompt = (
            f"<bos><start_of_turn>user\n"
            f"You are a highly capable AI translation assistant focusing on enterprise-grade accuracy.\n"
            f"Your task is to translate the following text from {src_name} ({src_lang}) to {tgt_name} ({tgt_lang}).\n"
            f"CRITICAL REQUIREMENTS:\n"
            f"1. Accurately convey the meaning and nuances of the original text.\n"
            f"2. Preserve all numbers, dates, and currencies exactly as they appear.\n"
            f"3. Preserve all original formatting, special characters, and HTML/XML structures completely intact.\n"
            f"4. For mixed-language inputs (e.g., source contains some English words), translate the main language and keep the technical terms as appropriate.\n"
            f"5. Produce ONLY the {tgt_name} translation. Do not add any conversational text, explanations, notes, or markdown formatting.\n\n"
            f"Source text:\n{text}<end_of_turn>\n"
            f"<start_of_turn>model\n"
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
                    do_sample=False,             # Beam cực ngặt để độ chính xác doanh nghiệp (deterministic)
                    num_beams=3,                 # Dùng Beam Search = 3 để văn phong dịch mượt và đúng ngữ cảnh hơn
                    repetition_penalty=1.1,      # Phạt lặp từ nhẹ để không bao giờ bị dính lỗi lặp "Tr Tr Tr..."
                    early_stopping=True
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
