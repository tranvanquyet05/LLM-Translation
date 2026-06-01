import json
import os

input_file = r"C:\Users\tvquyet\Code\LLM_Translation\data\20260527-102649-misa-translategemma-4b-it.json.txt"
output_file = r"C:\Users\tvquyet\Code\LLM_Translation\output\benchmark.json"

def process_benchmark():
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    extracted = []
    # Kiểm tra xem list ở ngoài cùng hay trong object
    if isinstance(data, list):
        for item in data:
            if "results" in item:
                for result in item["results"]:
                    extracted.append({
                        "id": result.get("id"),
                        "source": result.get("source"),
                        "translated": result.get("translated")
                    })
    else:
        print("Định dạng file không giống như mong đợi.")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã trích xuất thành công {len(extracted)} items sang {output_file}")

if __name__ == "__main__":
    process_benchmark()
