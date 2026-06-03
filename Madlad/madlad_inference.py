import torch
from bs4 import BeautifulSoup
from transformers import T5ForConditionalGeneration, T5Tokenizer


class MadladTranslator:
    def __init__(
        self,
        model_name: str = "google/madlad400-3b-mt",
        device: str = "cuda",
        quantization: str = "fp16",
    ):
        """
        Khởi tạo MADLAD-400 translator (T5-based Seq2Seq).

        quantization:
          - "fp16" : float16, khuyên dùng cho T4 (~6-7 GB VRAM cho 3B)
          - "4bit" : NF4 qua bitsandbytes (~2-3 GB VRAM)
          - "8bit" : INT8 qua bitsandbytes (~4-5 GB VRAM)
          - "none" : float32 gốc (cần ~12 GB VRAM)
        """
        print(f"Loading {model_name} (quantization={quantization}) ...")
        self.device = device

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        if quantization == "4bit":
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=bnb,
                device_map="auto",
            )
        elif quantization == "8bit":
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto",
            )
        else:
            dtype = torch.float16 if quantization == "fp16" else torch.float32
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
            )

        self.model.eval()
        print("Model loaded successfully!")

    def translate_batch(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        num_beams: int = 4,
        max_length: int = 512,
    ) -> list[str]:
        """
        Dịch một batch texts.
        MADLAD chỉ cần target lang token `<2xx>` — source tự detect.
        """
        if not texts:
            return []

        lang_token = f"<2{tgt_lang}>"
        prefixed = [f"{lang_token} {t}" for t in texts]

        self.tokenizer.padding_side = "right"
        inputs = self.tokenizer(
            prefixed,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                num_beams=num_beams,
                max_length=max_length,
                early_stopping=True,
                # Giống NLLB: repetition_penalty ngăn loop thoái hóa
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def translate_html_safe(
        self,
        html_text: str,
        src_lang: str,
        tgt_lang: str,
    ) -> str:
        """
        Duyệt DOM, chỉ dịch text nodes có chữ cái.
        KHÔNG dùng placeholder — T5 tokenizer không hiểu token nhân tạo.
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
        translated = self.translate_batch(texts, src_lang, tgt_lang)

        for (node, orig), trans in zip(nodes, translated):
            node.replace_with(node.replace(orig, trans))

        return str(soup)

    def translate(
        self,
        source: str,
        src_lang: str,
        tgt_lang: str,
        kind: str = "plain",
    ) -> str:
        """API dịch chính."""
        if src_lang == tgt_lang:
            return source

        if kind == "html" or ("<" in source and ">" in source):
            return self.translate_html_safe(source, src_lang, tgt_lang)

        results = self.translate_batch([source], src_lang, tgt_lang)
        return results[0] if results else source
