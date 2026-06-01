import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bs4 import BeautifulSoup

class HunyuanTranslator:
    def __init__(self, model_name="tencent/Hy-MT2-7B", device="cuda", quantization="fp16"):
        """
        Khởi tạo translator cho Tencent Hunyuan-MT / Hy-MT2.
        quantization có các giá trị:
        - "fp16": Chạy ở dạng half-precision FP16/BF16 (Khuyên dùng cho T4 GPU vì dòng model này cực ổn định với bfloat16).
        - "4bit": Chạy ở dạng 4-bit NF4 qua bitsandbytes để tiết kiệm VRAM tối đa (~5-6GB).
        - "8bit": Chạy ở dạng 8-bit qua bitsandbytes.
        - "none": Load model gốc không quantize (FP32).
        """
        print(f"Loading {model_name} on {device} (quantization: {quantization})...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Bản đồ ánh xạ ngôn ngữ từ mã ISO 639-1 sang tên Tiếng Anh đầy đủ
        self.lang_to_english = {
            "vi": "Vietnamese",
            "en": "English",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "km": "Khmer",
            "lo": "Lao",
            "my": "Burmese",
            "ne": "Nepali",
            "si": "Sinhala",
            "ta": "Tamil",
            "fr": "French",
            "de": "German",
            "es": "Spanish",
            "pt": "Portuguese",
            "it": "Italian",
            "ru": "Russian",
            "ar": "Arabic",
            "th": "Thai",
            "ms": "Malay",
            "id": "Indonesian",
            "tl": "Filipino",
            "pl": "Polish",
            "tr": "Turkish",
            "hi": "Hindi"
        }

        # Bản đồ ánh xạ ngôn ngữ từ mã ISO 639-1 sang tên Tiếng Trung đầy đủ (cho prompt ZH <=> XX)
        self.lang_to_chinese = {
            "vi": "越南语",
            "en": "英语",
            "zh": "中文",
            "ja": "日语",
            "ko": "韩语",
            "km": "高棉语",
            "lo": "老挝语",
            "my": "缅甸语",
            "ne": "尼泊尔语",
            "si": "僧伽罗语",
            "ta": "泰米尔语",
            "fr": "法语",
            "de": "德语",
            "es": "西班牙语",
            "pt": "葡萄语",
            "it": "意大利语",
            "ru": "俄语",
            "ar": "阿拉伯语",
            "th": "泰语",
            "ms": "马来语",
            "id": "印尼语",
            "tl": "菲律宾语",
            "pl": "波兰语",
            "tr": "土耳其语",
            "hi": "印地语"
        }

        if quantization == "4bit":
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
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
            # Mặc định sử dụng bfloat16 cho các card GPU hiện đại hoặc float16 cho GPU T4 nguyên bản
            # bfloat16 khuyên dùng để tránh lỗi tràn số và tăng độ ổn định.
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            if quantization == "none":
                dtype = torch.float32

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
        Xây dựng prompt tối ưu theo mẫu chính thức của Tencent Hunyuan-MT / Hy-MT2.
        - Với các cặp ngôn ngữ có chứa Tiếng Trung (zh): Dùng mẫu prompt tiếng Trung.
        - Với các cặp ngôn ngữ khác: Dùng mẫu prompt tiếng Anh.
        """
        src_lang = src_lang.lower().strip()
        tgt_lang = tgt_lang.lower().strip()

        # 1. Xác định prompt template và tên ngôn ngữ phù hợp
        if src_lang == "zh" or tgt_lang == "zh":
            # Sử dụng mẫu prompt tiếng Trung
            target_name = self.lang_to_chinese.get(tgt_lang, tgt_lang.capitalize())
            prompt_content = f"把下面的文本翻译成{target_name}，不要额外解释。 {text}"
        else:
            # Sử dụng mẫu prompt tiếng Anh
            target_name = self.lang_to_english.get(tgt_lang, tgt_lang.capitalize())
            prompt_content = f"Translate the following segment into {target_name}, without additional explanation. {text}"

        # 2. Định dạng prompt theo Chat Template (user role)
        messages = [
            {"role": "user", "content": prompt_content}
        ]
        
        try:
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Dự phòng nếu tokenizer không hỗ trợ hoặc lỗi template
            formatted_prompt = f"<|im_start|>user\n{prompt_content}<|im_end|>\n<|im_start|>assistant\n"
            
        return formatted_prompt

    def translate_batch(self, texts, src_lang, tgt_lang):
        """
        Dịch đồng thời một danh sách văn bản sử dụng GPU batching để tối ưu hiệu năng.
        """
        if not texts:
            return []
            
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
        API dịch chính cho Hunyuan-MT.
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
