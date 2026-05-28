import json
import pandas as pd

def analyze_results(json_report_path):
    with open(json_report_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    records = []
    # Xử lý data từ format cũ hoặc format chuẩn xuất ra từ validators
    if isinstance(data, list) and "fixture" in data[0]:
        # Xử lý input từ file JSON kết quả cũ bạn đã cung cấp 
        for fixture in data:
            for item in fixture.get('results', []):
                digit_ok = item.get('digit_check', {}).get('ok', False)
                records.append({
                    "id": item["id"],
                    "source_lang": item["source_lang"],
                    "target_lang": item["target_lang"],
                    "elapsed_ms": item.get("elapsed_ms", 0),
                    "digit_preservation": digit_ok,
                    "target_lang_match": item.get("judge", {}).get("target_lang_match", 0),
                    "fidelity": item.get("judge", {}).get("fidelity", 0)
                })
    else:
        # Xử lý data từ framework src/benchmark.py mới
        for item in data:
            # Flatten dictionary for dataframe
            records.append({
                "id": item.get("id"),
                "source_lang": item.get("source_lang"),
                "target_lang": item.get("target_lang"),
                "elapsed_ms": item.get("elapsed_ms", 0),
                "digit_preservation": item.get("validators", {}).get("digit_check", {}).get("ok"),
                "html_preservation": item.get("validators", {}).get("html_check", {}).get("ok"),
                "error": item.get("error", None)
            })

    df = pd.DataFrame(records)
    print("=== SUMMARY METRICS ===")
    
    # Tính Timeout/Error rate (ví dụ những case > 15s hoặc HTTP Error)
    if 'error' in df.columns:
        error_rate = df['error'].notna().mean() * 100
        print(f"Error/Timeout Rate: {error_rate:.2f}%")
    else:
        error_rate = (df['elapsed_ms'] > 30000).mean() * 100 # coi > 30s là timeout cục bộ
        print(f"Latency > 30s (Simulated Timeout): {error_rate:.2f}%")
        
    print(f"Average Latency: {df['elapsed_ms'].mean():.0f} ms")
    print(f"Digit Preservation Pass Rate: {df['digit_preservation'].mean() * 100:.2f}%")
    if 'fidelity' in df.columns:
         print(f"Fidelity Score (Avg): {df['fidelity'].mean():.2f}/5.0")
    if 'html_preservation' in df.columns:
         print(f"HTML Structure Pass Rate: {df['html_preservation'].mean() * 100:.2f}%")
    
    print("\n=== BREAKDOWN BY TARGET LANG ===")
    lang_stats = df.groupby('target_lang').agg(
        Count=('id', 'count'),
        Avg_Latency_ms=('elapsed_ms', 'mean'),
        Digit_Pass_Rate=('digit_preservation', 'mean')
    ).round(2)
    print(lang_stats)

if __name__ == "__main__":
    # Ví dụ đọc dữ liệu benchmark mẫu cũ bạn đang có ở ngoài root để phân tích:
    analyze_results("../20260527-102649-misa-translategemma-4b-it.json.txt")
