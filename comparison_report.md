# Báo cáo So sánh Chất lượng Dịch thuật

**Baseline** (`output/benchmark.json`) vs **TranslateGemma 4B** (`output/gemma_results_new.json`)

> Lưu ý: Bỏ qua các vấn đề về cấu trúc HTML. Các case `vi-vi-*` (src == tgt) không được đưa vào đánh giá.

---

## Tổng quan

| Tiêu chí | Baseline | TranslateGemma 4B |
|---|---|---|
| Latency trung bình (vi→en) | ~1.5s | ~3.5s |
| Timeout / Error | 1 case (vi-km-007, 60s) | 0 |
| Tính nhất quán (cùng input) | ❌ Biến thiên | ✅ Ổn định |

---

## Chi tiết các test khác nhau

### 1. 🔴 Hallucination — `vi-en-001`

> Source: *"Tổng doanh thu năm 2025 đạt 12.500.000.000 đồng, tăng 15% so với cùng kỳ."*

| Model | Bản dịch |
|---|---|
| **Baseline** ✅ | *"Total revenue in 2025 **reached** 12,500,000,000 VND, an increase of 15%..."* |
| **Gemma** ❌ | *"Total revenue for 2025 **is projected to reach** 12,500,000,000 VND..."* |

**Nhận xét:** Gemma thêm "is projected to" — không có trong bản gốc, làm sai nghĩa tài chính (thực tế → dự báo).

---

### 2. 🟡 Tính nhất quán — `vi-en-003` (15 lần cùng input)

> Source: *"Quyết định giải thoát cho người yêu để họ tìm hạnh phúc mới, dù bản thân phải chịu đựng sự quên lãng và đau đớn khi lý trí và con tim mâu thuẫn."*

| Model | Kết quả |
|---|---|
| **Baseline** ❌ | Ra **3 bản dịch khác nhau** xen kẽ nhau trong 15 lần |
| **Gemma** ✅ | **Hoàn toàn giống nhau** cả 15 lần |

**Các biến thể của Baseline:**
- *"...despite the pain and conflict of having the **mind and heart** disagree."*
- *"...despite the pain and conflict between **reason and heart** that the speaker must endure."*
- *"...even though the speaker must endure the **pain of being forgotten** and the conflict between reason and heart."*

**Gemma** dịch nhất quán và sát nghĩa hơn: *"...even though it means enduring the **pain of being forgotten** and the conflict between reason and emotion."* — cụm "sự quên lãng" được giữ nguyên, trong khi một số biến thể của Baseline bỏ qua.

---

### 3. 🔴 Lỗi số liệu trong HTML — `vi-zh-002`

> Source: `<p>Quý 1: <b>120 tỷ</b>, Quý 2: <b>135 tỷ</b></p>`

| Model | Output |
|---|---|
| **Baseline** ✅ | `<b>120亿</b>，第二季度：<b>135亿</b>` — giữ nguyên dạng số |
| **Gemma** ❌ | `<b>十二十亿</b>...<b>十三亿五千万</b>` — chuyển sang chữ Hán, **sai cú pháp** (十二十亿 không hợp lệ trong tiếng Trung) |

**Nhận xét:** Khi dịch vi→zh có số liệu trong HTML, Baseline bảo toàn số tốt hơn đáng kể.

---

### 4. 🔴 Timeout — `vi-km-007`

> Source: *"Chính anh đã không biết cách, không biết cách yêu em. Để em phải buồn nhiều lần..."* (~50 từ)

| Model | Kết quả |
|---|---|
| **Baseline** ❌ | `null` + lỗi **504 Gateway Timeout** (60,035ms) |
| **Gemma** ✅ | Dịch thành công trong **6,603ms** |

**Nhận xét:** Baseline (dạng API) không ổn định với đoạn văn dài chứa nội dung cảm xúc.

---

### 5. 🟡 Chất lượng tiếng Nhật — `vi-ja-001`

> Source: *"Xin lỗi, tôi không hiểu câu hỏi. Bạn có thể nhắc lại không?"*

| Model | Bản dịch |
|---|---|
| **Baseline** | 申し訳ありません。質問の意味がわかりません。もう一度おっしゃっていただけますか？ |
| **Gemma** ✅ | 申し訳ありません**が**、質問の意味がわかりません。もう一度お尋ねいただけます**でしょうか**？ |

**Nhận xét:** Gemma dùng trợ từ **が** (liên kết tự nhiên hơn) và thể kính ngữ **でしょうか** (lịch sự hơn). Gemma nhỉnh hơn về độ tự nhiên.

---

## Lỗi chung của cả 2 model

| Test ID | Lỗi |
|---|---|
| `vi-zh-mix-001`, `vi-km-001` | **Sai thứ trong tuần**: "thứ Hai" (Monday) → 星期二 / Thursday |
| `vi-km-003` | Chèn từ tiếng Indonesia **"pertemuan"** vào output Khmer |
| `vi-lo-001` | Output lẫn lộn **Khmer + Thái + script khác** thay vì Lào thuần túy |
| `vi-si-001` | Đơn vị tiền tệ sai: Baseline ra "ඩොලර්" (dollar), Gemma ra "동" (tiếng Hàn) |

> **Kết luận chung cho nhóm ngôn ngữ Đông Nam Á hiếm (Km, Lo, My, Si):** Cả 2 model đều yếu — nên dùng model chuyên biệt như **NLLB-200** cho các ngôn ngữ này.

---

## Kết luận & Khuyến nghị

| Trường hợp sử dụng | Model nên dùng | Lý do |
|---|---|---|
| Vi → En/Ja/Ko (nội dung nghiệp vụ) | **Gemma** | Nhất quán, ít hallucination |
| Vi → Zh có số liệu | **Baseline** | Giữ nguyên dạng số tốt hơn |
| Latency là ưu tiên | **Baseline** | ~1.5s vs ~3.5s |
| Production cần output ổn định | **Gemma** | 100% deterministic trên cùng input |
| Vi → Km / Lo / My / Si | **NLLB-200** | Cả 2 model đều không đáng tin cậy |
