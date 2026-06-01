import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bs4 import BeautifulSoup
import re

class GemmaTranslator:
    def __init__(self, model_name="google/translategemma-4b-it", device="cuda", quantization="fp16"):
        """
        Khởi tạo translator cho TranslateGemma.
        quantization có các giá trị:
        - "fp16": Chạy ở dạng half-precision FP16 (Khuyên dùng cho T4x2 vì model 4B chiếm ~8.6GB VRAM, chạy nhanh nhất và chất lượng tốt nhất).
        - "bf16": Chạy ở dạng BF16 (nếu GPU hỗ trợ).
        - "4bit": Chạy ở dạng 4-bit NF4 qua bitsandbytes để tiết kiệm VRAM tối đa (~3-4GB).
        - "none": Load model gốc không quantize (FP32).
        """
        print(f"Loading {model_name} on {device} (quantization: {quantization})...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if quantization == "4bit":
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, # Sử dụng bfloat16 để tránh lỗi tràn số (overflow/NaN)
                bnb_4bit_use_double_quant=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                attn_implementation="sdpa"
            )
        elif quantization == "8bit":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto",
                attn_implementation="sdpa"
            )
        else:
            # Native FP16 / BF16
            # CHÚ Ý QUAN TRỌNG: Các dòng Gemma rất không ổn định ở dạng float16 và dễ bị lỗi tràn số (NaN),
            # dẫn đến lỗi 'CUDA error: device-side assert triggered'. 
            # Bắt buộc phải sử dụng bfloat16 để đảm bảo tính ổn định số học.
            dtype = torch.bfloat16
                
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                attn_implementation="sdpa"
            )
            
        print("Model loaded successfully!")
        self._warmup()

    def _format_prompt(self, text, src_lang, tgt_lang):
        """
        Sử dụng định dạng structured chat template chính thức của TranslateGemma.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": src_lang,
                        "target_lang_code": tgt_lang,
                        "text": text
                    }
                ]
            }
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def translate_batch(self, texts, src_lang, tgt_lang):
        """
        Dịch đồng thời một danh sách văn bản sử dụng GPU batching để tối ưu hiệu năng.
        """
        if not texts:
            return []
            
        # Xử lý làm sạch mã ngôn ngữ dựa vào C:\Users\tvquyet\Code\LLM_Translation\data\test.json
        src_lang = src_lang.lower().strip()
        tgt_lang = tgt_lang.lower().strip()

        # Nếu chỉ có 1 câu, dịch tuần tự nhanh
        if len(texts) == 1:
            prompt = self._format_prompt(texts[0], src_lang, tgt_lang)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.05
                )
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            return [self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()]

        # Dịch song song (Batching)
        prompts = [self._format_prompt(t, src_lang, tgt_lang) for t in texts]
        
        # Cấu hình padding bên trái cho generator của CausalLM
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.05
            )
            
        translated_texts = []
        for i, out in enumerate(outputs):
            input_len = inputs.input_ids[i].shape[0]
            gen_tokens = out[input_len:]
            decoded = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            translated_texts.append(decoded)
            
        return translated_texts

    def translate_html_safe(self, html_text, src_lang, tgt_lang):
        """
        Duyệt cây DOM, gom các Node text có nghĩa đi dịch theo lô (batching) và chèn lại vào HTML.
        Bảo toàn tags HTML hoàn hảo và tối ưu hóa thời gian thực thi.
        """
        soup = BeautifulSoup(html_text, 'html.parser')
        nodes_to_translate = []
        
        # Chỉ lấy text có nghĩa, bỏ qua tags chức năng
        for node in soup.find_all(string=True):
            if node.parent.name not in ['style', 'script', 'head', 'title', 'meta', '[document]']:
                text_strip = node.strip()
                # Bỏ qua các text chỉ chứa số hoặc ký tự đặc biệt
                if text_strip and any(c.isalpha() for c in text_strip):
                    nodes_to_translate.append((node, text_strip))
        
        if not nodes_to_translate:
             return html_text
             
        texts = [t for n, t in nodes_to_translate]
        translated_texts = self.translate_batch(texts, src_lang, tgt_lang)
        
        for (node, orig_text), trans_text in zip(nodes_to_translate, translated_texts):
             new_string = node.replace(orig_text, trans_text)
             node.replace_with(new_string)
             
        return str(soup)

    def translate(self, source, src_lang, tgt_lang, kind="plain"):
        """
        API dịch chính cho TranslateGemma.
        """
        if src_lang == tgt_lang:
            return source

        if kind == "html" or ("<" in source and ">" in source):
            return self.translate_html_safe(source, src_lang, tgt_lang)
        else:
            return self.translate_batch([source], src_lang, tgt_lang)[0]

    def _warmup(self):
        """
        Khởi tạo CUDA kernels và chạy thử trước 1 lượt để tránh bị trễ ở lượt chạy thật đầu tiên.
        """
        print("Warming up CUDA kernels...")
        dummy_prompt = self._format_prompt("Hello", "en", "vi")
        dummy_input = self.tokenizer(dummy_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            _ = self.model.generate(**dummy_input, max_new_tokens=5)
        print("Warm-up done!")
