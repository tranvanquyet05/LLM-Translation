# Cấu hình mapping giữa mã ngôn ngữ rút gọn và mã BCP-47 của NLLB-200.
# NLLB-200 FLORES-200 Language Codes

LANG_MAP = {
    "vi": "vie_Latn",      # Tiếng Việt
    "en": "eng_Latn",      # Tiếng Anh
    "zh": "zho_Hans",      # Tiếng Trung (Giản thể) - có thể dùng zho_Hant cho Phồn thể
    "ja": "jpn_Jpan",      # Tiếng Nhật
    "ko": "kor_Hang",      # Tiếng Hàn
    "km": "khm_Khmr",      # Tiếng Khmer
    "lo": "lao_Laoo",      # Tiếng Lào
    "my": "mya_Mymr",      # Tiếng Myanmar (Miến Điện)
    "ta": "tam_Taml",      # Tiếng Tamil
    "ne": "npi_Deva",      # Tiếng Nepali
    "si": "sin_Sinh",      # Tiếng Sinhala
}

def get_nllb_code(lang_code):
    """Trả về mã NLLB cho ngôn ngữ, nếu không tìm thấy thì fallback dùng đúng code đó."""
    return LANG_MAP.get(lang_code.lower(), lang_code)
