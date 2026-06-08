# TransMTL_OntoKG

Tóm tắt văn bản tiếng Việt + trích xuất keyphrase đa nhiệm (**TransMTL**),
tích hợp tuỳ chọn đồ thị tri thức bản thể học (**OntoKG / Neo4j**).

---

## 1. Bạn chỉ cần nhớ 3 file

| File | Vai trò | Bạn có sửa không? |
|------|---------|-------------------|
| **`pipeline_config.py`** | Bảng điều khiển: đường dẫn dữ liệu, Neo4j, bật/tắt OntoKG | ✅ Sửa **trước khi chạy** |
| **`conf.py`** | Hyperparameter của model (d_model, lr, batch_size…) | ✅ Sửa nếu muốn tinh chỉnh |
| **`run_ontokg.py` / `run_transmtl.py`** | 2 file để **chạy** (xem mục 4) | ❌ Chỉ chạy, không cần sửa |

> Trước đây bị "loạn" giữa `main.py` và `run_code.py`. Nay đã gọn lại:
> dùng **`run_ontokg.py`** (bước 1) và **`run_transmtl.py`** (bước 2).
> `main.py` chỉ là tuỳ chọn "chạy tất cả trong 1 lệnh" (mục 5) — **không bắt buộc**.

---

## 2. Cài đặt

```bash
pip install torch transformers underthesea rouge-score pandas numpy neo4j \
            pykeen umap-learn hdbscan sentencepiece
```
(Phần OntoKG cần thêm `transformers`, `pykeen`, `umap-learn`, `hdbscan`; nếu chỉ
chạy baseline thì không cần các gói này.)

---

## 3. Sửa `pipeline_config.py` (chỉ vài dòng)

Mở `pipeline_config.py` và chỉnh đúng 4 chỗ:

```python
RAW_DATA_CSV   = ".../tintuc_gen_final.csv"   # CSV gốc của bạn
PRETRAINED_VEC = ".../cc.vi.300.bin"          # FastText tiếng Việt

USE_ONTOKG     = False     # False = baseline | True = dùng OntoKG
NEO4J_PASSWORD = "password" # chỉ cần khi USE_ONTOKG = True
```

> CSV gốc cần có các cột: `title, summary, content, publish_time, topic, cleaned_keywords`.
>
> Không thích sửa file? Có thể truyền qua biến môi trường:
> `RAW_DATA_CSV=/duong/dan.csv python run_transmtl.py`

---

## 4. CÁCH CHẠY (chọn 1 trong 2 trường hợp)

### 🟢 Trường hợp A — Baseline (KHÔNG dùng OntoKG, không cần Neo4j)

1. Trong `pipeline_config.py`: đặt `USE_ONTOKG = False`.
2. Chạy đúng **một lệnh**:

```bash
python run_transmtl.py
```

File này tự: chia dữ liệu → train TransMTL → test. Xong.

---

### 🔵 Trường hợp B — Dùng OntoKG (TransMTL + tri thức)

Cần Neo4j đang chạy. Làm **2 bước**, theo đúng thứ tự:

1. Trong `pipeline_config.py`: đặt `USE_ONTOKG = True`.

2. **BƯỚC 1 — xây OntoKG (chạy MỘT LẦN, đây là phần tốn thời gian):**

```bash
python run_ontokg.py
```
> Tạo `data/kge/entity_embeddings.pt` và nạp KG vào Neo4j.
> Tự **bỏ qua** module đã chạy xong (resume) → lần sau chạy lại rất nhanh.
> Muốn dựng lại từ đầu: xoá thư mục `data/` rồi chạy lại.

3. **BƯỚC 2 — train + test TransMTL (chạy lại bao nhiêu lần tuỳ ý):**

```bash
python run_transmtl.py
```
> Train TransMTL có gắn embedding tri thức (truy vấn Neo4j đã dựng ở Bước 1),
> rồi test. **Không phải dựng lại OntoKG** mỗi lần train.

**Ý tưởng tách bước:** Bước 1 (OntoKG) nặng nhưng chỉ cần làm 1 lần. Sau đó bạn
có thể thử nhiều cấu hình TransMTL khác nhau (sửa `conf.py`) và chạy lại Bước 2
nhiều lần mà tái dùng KG đã có.

---

## 5. (Tuỳ chọn) Chạy tất cả trong 1 lệnh: `main.py`

Nếu muốn chạy nguyên pipeline một phát (split → ontokg → train → test):

```bash
python main.py --use-ontokg     # chạy hết, CÓ OntoKG
python main.py --no-ontokg      # chạy hết, baseline
python main.py                  # theo USE_ONTOKG trong pipeline_config.py
```

### Giải thích các flag (đều là tuỳ chọn — không truyền thì lấy từ config)

| Flag | Ý nghĩa | Mặc định |
|------|---------|----------|
| `--stage {all,split,ontokg,train,test}` | Chỉ chạy 1 giai đoạn | `all` |
| `--use-ontokg` / `--no-ontokg` | Bật / tắt OntoKG cho lần chạy này | theo `USE_ONTOKG` |
| `--skip-existing` | Bỏ qua bước đã có output (resume) | tắt |
| `--save-path ĐƯỜNG_DẪN` | Nơi lưu checkpoint | `Results_Score/BestModel.pt` |
| `--neo4j-pass MẬT_KHẨU` | Mật khẩu Neo4j | theo config |

Ví dụ chỉ chạy lại phần test:
```bash
python main.py --stage test --use-ontokg
```

> `run_ontokg.py` ≈ `main.py --stage ontokg --use-ontokg --skip-existing`
> `run_transmtl.py` ≈ `main.py --stage train` + `main.py --stage test`
> Hai file ngắn này có sẵn để bạn **khỏi phải nhớ flag**.

---

## 6. Kết quả nằm ở đâu?

| Bước | Output |
|------|--------|
| split | `data_split/{train,val,test,trainval}.csv` |
| ontokg | `data/kge/entity_embeddings.pt`, KG trong Neo4j |
| train | `Results_Score/BestModel.pt` (checkpoint tốt nhất) |
| test | log ROUGE-1/2/L + Keyphrase P/R/F1, file `Results_Score/BestModel_test_results.txt` |

> OntoKG được xây trên **trainval**; `get_loaders` tự tách held-out test bằng
> cùng seed=42 nên `test` đánh giá đúng phần dữ liệu model chưa từng train.
> Muốn test trên file `test.csv` riêng: đổi `TEST_DATA_CSV = TEST_CSV` trong
> `pipeline_config.py`.

---

## 7. OntoKG chạy chậm? Mẹo tăng tốc

Đã tối ưu sẵn: NER + embedding model chạy **GPU + fp16** (trước đây chạy CPU),
và chỉ tính POS cho `full_text`. Muốn nhanh hơn nữa:

- Chạy `run_ontokg.py` **một lần** rồi tái dùng (đã resume sẵn nhờ skip-existing).
- `OKG_DEVICE=cuda` (mặc định) — đảm bảo có GPU.
- Trong `OntoKG/module2_ner_concept.py`: đặt `use_phonlp=False` nếu không cần.
- Trong `OntoKG/module3_entity_linking.py`: đặt `use_wikidata=False` để bỏ truy
  vấn mạng Wikidata (nhanh hẳn, nhưng mất liên kết Wikidata).
- Trong `OntoKG/module7_kge_training.py`: giảm `n_epochs` (vd 200 → 100).

---

## 8. Lỗi thường gặp

- **`FileNotFoundError: trainval.csv`** → chưa chia dữ liệu. Chạy `run_transmtl.py`
  hoặc `run_ontokg.py` (cả hai tự chia), hoặc kiểm tra `RAW_DATA_CSV`.
- **Kết nối Neo4j thất bại** (khi `USE_ONTOKG=True`) → kiểm tra Neo4j đang chạy ở
  `NEO4J_URI` và `NEO4J_PASSWORD` đúng.
- **Out of memory khi train** → giảm `BATCH_SIZE` trong `conf.py`.
