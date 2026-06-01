import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from bs4 import BeautifulSoup
from src.lang_map import get_nllb_code

class NLLBTranslator:
    def __init__(self, model_name="facebook/nllb-200-3.3B", device="cuda"):
        print(f"Loading {model_name} on {device}...")
        self.device = device
        # Ép dùng Tokenizer chậm (Slow Tokenizer) của NLLB để tránh lỗi của FastTokenizer trên Kaggle
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            load_in_8bit=True,        # 8-bit quantization giảm một nửa VRAM (cần cài bitsandbytes)
            device_map="auto"         # Tự động map vào GPU cho lượng VRAM còn lại
        )
        print("Model loaded successfully!")

    def translate_batch(self, texts, src_lang, tgt_lang):
        """Hàm dịch cơ bản cho list of texts."""
        if not texts:
            return []
            
        src_code = get_nllb_code(src_lang)
        tgt_code = get_nllb_code(tgt_lang)
        
        self.tokenizer.src_lang = src_code
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(self.device)
        
        # Phương pháp an toàn 100% để lấy token ID của ngôn ngữ đích
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_code)
        
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=1024,
                num_beams=1,           # Deterministic (không dùng beam search để chạy nhanh nhất)
                do_sample=False        # Không mix sáng tạo
            )
            
        translated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
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
