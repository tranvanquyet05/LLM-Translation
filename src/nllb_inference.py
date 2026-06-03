import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from bs4 import BeautifulSoup
from src.lang_map import get_nllb_code


class NLLBTranslator:
    def __init__(self, model_name="facebook/nllb-200-3.3B", device="cuda"):
        print(f"Loading {model_name} on {device}...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        print("Model loaded successfully!")

    def translate_batch(
        self,
        texts,
        src_lang,
        tgt_lang,
        num_beams=3,
        max_length=512,
    ):
        """Dịch một batch texts với cấu hình tối ưu cho NLLB."""
        if not texts:
            return []

        src_code = get_nllb_code(src_lang)
        tgt_code = get_nllb_code(tgt_lang)

        self.tokenizer.src_lang = src_code
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_code)

        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                # BẮT BUỘC với NLLB: repetition_penalty ngăn loop thoái hóa
                repetition_penalty=1.2,
                # no_repeat_ngram_size=3: ngăn lặp 3-gram mà không phá
                # proper nouns / mã số 2 ký tự như HD-2026
                no_repeat_ngram_size=3,
            )

        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    def translate_html_safe(self, html_text, src_lang, tgt_lang):
        """
        Duyệt DOM, chỉ dịch text nodes có nghĩa và chèn lại vào HTML.
        Bảo toàn hoàn hảo HTML tags.
        """
        soup = BeautifulSoup(html_text, "html.parser")
        nodes = []

        for node in soup.find_all(string=True):
            if node.parent.name not in [
                "style", "script", "head", "title", "meta", "[document]",
            ]:
                stripped = node.strip()
                if stripped and any(c.isalpha() for c in stripped):
                    nodes.append((node, stripped))

        if not nodes:
            return html_text

        texts = [t for _, t in nodes]
        translated_texts = self.translate_batch(texts, src_lang, tgt_lang)

        for (node, orig_text), trans_text in zip(nodes, translated_texts):
            node.replace_with(node.replace(orig_text, trans_text))

        return str(soup)

    def translate(self, source, src_lang, tgt_lang, kind="plain"):
        """API dịch chính."""
        if src_lang == tgt_lang:
            return source

        if kind == "html" or ("<" in source and ">" in source):
            return self.translate_html_safe(source, src_lang, tgt_lang)

        return self.translate_batch([source], src_lang, tgt_lang)[0]
