import json
import time
import os
from tqdm import tqdm
from llama_cpp import Llama
from validators import check_digits, check_html_structure
from prompts import get_translate_prompt

class TranslatorBenchmark:
    def __init__(self, model_path, n_ctx=2048, n_gpu_layers=-1, model_type="gemma"):
        """ Khởi tạo model quantize GGUF thông qua llama.cpp """
        print(f"Loading model from {model_path}...")
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )
        self.model_type = model_type

    def run_inference(self, source_text, source_lang, target_lang):
        prompt = get_translate_prompt(self.model_type, source_text, source_lang, target_lang)
        
        start_time = time.time()
        
        # Stop words theo từng model format để ngắt sinh chữ
        stop_words = ["<|im_end|>", "<end_of_turn>", "<|eot_id|>"]
        
        try:
            output = self.model(
                prompt,
                max_tokens=1024,
                temperature=0.1, # Temperature thấp cho deterministic output
                top_p=0.9,
                stop=stop_words,
                echo=False
            )
            translation = output["choices"][0]["text"].strip()
            elapsed_ms = int((time.time() - start_time) * 1000)
            return translation, elapsed_ms, None
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return None, elapsed_ms, str(e)

    def benchmark_dataset(self, dataset, output_file):
        """ Chạy pipeline evaluation """
        results = []
        for case in tqdm(dataset, desc="Benchmarking"):
            source = case["source"]
            s_lang = case["source_lang"]
            t_lang = case["target_lang"]
            
            translated, elapsed_ms, err = self.run_inference(source, s_lang, t_lang)
            
            if err:
                case["error"] = err
                case["elapsed_ms"] = elapsed_ms
                results.append(case)
                continue
                
            # Chạy validators
            digit_status = check_digits(source, translated)
            html_status = check_html_structure(source, translated)
            
            case["translated"] = translated
            case["elapsed_ms"] = elapsed_ms
            case["validators"] = {
                "digit_check": digit_status,
                "html_check": html_status
            }
            results.append(case)
            
            # Save liên tục để tránh crash mất dữ liệu
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # Example Usage:
    # 1. Tải GGUF: khuyên dùng Qwen2.5-7B-Instruct-Q4_K_M.gguf cho hệ máy 16GB
    # MODEL_PATH = "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf" 
    
    # 2. Khai báo tests:
    test_cases = [
        {
            "id": "test-1", "source_lang": "vi", "target_lang": "en",
            "source": "<p>Doanh thu quý 1: <b>1.500.000đ</b></p>"
        }
    ]
    
    # runner = TranslatorBenchmark(MODEL_PATH, model_type="qwen")
    # runner.benchmark_dataset(test_cases, "results/benchmark_output.json")
    print("Benchmark framework ready.")
