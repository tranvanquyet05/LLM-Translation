# Quản lý Prompt templates cho các architecture khác nhau

SYSTEM_PROMPTS = {
    "default": "You are a highly capable AI translation assistant focusing on enterprise-grade accuracy. Your task is to translate directly and accurately without adding any conversational text, notes, or markdown formatting outside of the existing HTML tags."
}

def get_translate_prompt(model_type, source_text, source_lang, target_lang):
    """
    Tạo prompt tương ứng cho mỗi loại model.
    Các model Qwen, Gemma, Llama instruction following sẽ có format khác nhau một chút nếu cần.
    """
    
    instruction = f"Translate the following text from {source_lang} to {target_lang}. Preserve all numbers, original formatting, and HTML structures exactly as they appear."
    
    if model_type.lower() == "qwen":
        return f"<|im_start|>system\n{SYSTEM_PROMPTS['default']}<|im_end|>\n<|im_start|>user\n{instruction}\n\nSource text:\n{source_text}<|im_end|>\n<|im_start|>assistant\n"
    elif model_type.lower() == "gemma":
        return f"<start_of_turn>user\n{SYSTEM_PROMPTS['default']}\n{instruction}\n\nSource text:\n{source_text}<end_of_turn>\n<start_of_turn>model\n"
    elif model_type.lower() == "aya" or "llama" in model_type.lower():
        # Llama-3 format
        return f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPTS['default']}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}\n\nSource text:\n{source_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    # Generic format
    return f"{SYSTEM_PROMPTS['default']}\n\n{instruction}\n\nSource text:\n{source_text}\n\nTranslation:"
