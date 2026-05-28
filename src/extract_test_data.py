import json
import os

def extract_test_data(input_file, output_file):
    """
    Đọc file kết quả cũ, trích xuất source text và metadata cần thiết 
    để tạo thành file test.json phục vụ cho benchmark pipeline mới.
    """
    print(f"Reading from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_cases = []
    
    # Lặp qua từng nhóm test (fixture)
    for fixture in data:
        results = fixture.get('results', [])
        for item in results:
            # Lấy ra các trường dữ liệu cần thiết làm input cho quá trình dịch
            test_case = {
                "id": item.get("id"),
                "kind": item.get("kind"),
                "source_lang": item.get("source_lang"),
                "target_lang": item.get("target_lang"),
                "source": item.get("source")
            }
            test_cases.append(test_case)
    
    # Ghi ra file test.json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_cases, f, ensure_ascii=False, indent=2)
        
    print(f"Extracted {len(test_cases)} test cases and saved to {output_file}")

if __name__ == "__main__":
    # Sử dụng file đầu vào có sẵn trong thư mục gốc
    input_path = "c:\\Users\\tvquyet\\Code\\LLM_Translation\\20260527-102649-misa-translategemma-4b-it.json.txt"
    output_path = "c:\\Users\\tvquyet\\Code\\LLM_Translation\\test.json"

    extract_test_data(input_path, output_path)
import json
import os

def extract_test_data(input_file, output_file):
    """
    Đọc file kết quả cũ, trích xuất source text và metadata cần thiết 
    để tạo thành file test.json phục vụ cho benchmark pipeline mới.
    """
    print(f"Reading from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_cases = []
    
    # Lặp qua từng nhóm test (fixture)
    for fixture in data:
        results = fixture.get('results', [])
        for item in results:
            # Lấy ra các trường dữ liệu cần thiết làm input cho quá trình dịch
            test_case = {
                "id": item.get("id"),
                "kind": item.get("kind"),
                "source_lang": item.get("source_lang"),
                "target_lang": item.get("target_lang"),
                "source": item.get("source")
            }
            test_cases.append(test_case)
    
    # Ghi ra file test.json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_cases, f, ensure_ascii=False, indent=2)
        
    print(f"Extracted {len(test_cases)} test cases and saved to {output_file}")

if __name__ == "__main__":
    # Cấu hình đường dẫn (có thể chạy từ root hoặc thư mục src)
    input_filename = "20260527-102649-misa-translategemma-4b-it.json.txt"
    output_filename = "test.json"
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, input_filename)
    output_path = os.path.join(base_dir, output_filename)
    
    # Fallback nếu chạy trực tiếp cùng cấp thư mục với file
    if not os.path.exists(input_path):
        input_path = input_filename
        output_path = output_filename

    extract_test_data(input_path, output_path)
