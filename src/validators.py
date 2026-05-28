import re
from bs4 import BeautifulSoup
import unicodedata

def extract_numbers(text):
    """Trích xuất tất cả các cụm số từ chuỗi."""
    if not isinstance(text, str):
        return []
    # Normalize unicode to handle full-width digits (e.g., in Japanese/Chinese)
    normalized = unicodedata.normalize('NFKC', text)
    return re.findall(r'\d+', normalized)

def check_digits(source, target):
    """Kiểm tra xem bản dịch có giữ nguyên các số liệu hay không."""
    source_nums = extract_numbers(source)
    target_nums = extract_numbers(target)
    
    missing = [n for n in source_nums if n not in target_nums]
    return {
        "ok": len(missing) == 0,
        "source_runs": source_nums,
        "translated_runs": target_nums,
        "missing": missing
    }

def check_html_structure(source, target):
    """Kiểm tra cấu trúc HTML cơ bản có bị thay đổi hoặc mất tag không."""
    if '<' not in source and '>' not in source:
        return {"ok": True, "note": "No HTML in source"}
        
    soup_src = BeautifulSoup(source, "html.parser")
    soup_tgt = BeautifulSoup(target, "html.parser")
    
    src_tags = sorted([tag.name for tag in soup_src.find_all()])
    tgt_tags = sorted([tag.name for tag in soup_tgt.find_all()])
    
    missing = [t for t in src_tags if t not in tgt_tags]
    hallucinated = [t for t in tgt_tags if t not in src_tags]
    
    return {
        "ok": src_tags == tgt_tags,
        "source_tags": src_tags,
        "target_tags": tgt_tags,
        "missing_tags": missing,
        "hallucinated_tags": hallucinated
    }
