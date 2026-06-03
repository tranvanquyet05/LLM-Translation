import json
import os

INPUT_FILE = "/home/misa/Code/LLM-Translation/data/20260527-102649-misa-translategemma-4b-it.json.txt"
OUTPUT_FILE = "/home/misa/Code/LLM-Translation/output/benchmark.json"


def process_benchmark():
    output_dir = os.path.dirname(OUTPUT_FILE)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    extracted = []

    # Trường hợp file là list
    if isinstance(data, list):
        for item in data:
            results = item.get("results", [])
            for result in results:
                extracted.append({
                    "id": result.get("id"),
                    "kind": result.get("kind"),
                    "source_lang": result.get("source_lang"),
                    "target_lang": result.get("target_lang"),
                    "source": result.get("source"),
                    "translated": result.get("translated"),
                    "elapsed_ms": result.get("elapsed_ms"),
                    "error": result.get("error")
                })

    # Trường hợp file là object chứa results
    elif isinstance(data, dict):
        results = data.get("results", [])

        for result in results:
            extracted.append({
                "id": result.get("id"),
                "kind": result.get("kind"),
                "source_lang": result.get("source_lang"),
                "target_lang": result.get("target_lang"),
                "source": result.get("source"),
                "translated": result.get("translated"),
                "elapsed_ms": result.get("elapsed_ms"),
                "error": result.get("error")
            })

    else:
        print("❌ Định dạng JSON không được hỗ trợ.")
        return

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã trích xuất {len(extracted)} items")
    print(f"📄 Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    process_benchmark()