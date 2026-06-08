# TransMTL_OntoKG

Tóm tắt văn bản tiếng Việt + trích xuất keyphrase đa nhiệm (TransMTL), tích hợp
tuỳ chọn đồ thị tri thức bản thể học (OntoKG / Neo4j).

## Chạy nhanh — CHỈ 1 LỆNH

Toàn bộ pipeline (chia dữ liệu → dựng OntoKG → train → test) chạy qua **`main.py`**:

```bash
python main.py                 # chạy hết theo cấu hình mặc định
python main.py --use-ontokg    # bật OntoKG cho lần chạy này (cần Neo4j)
python main.py --no-ontokg     # ép baseline TransMTL (không cần Neo4j)
python main.py --stage train   # chỉ 1 giai đoạn: split | ontokg | train | test
python main.py --skip-existing # bỏ qua bước đã có output (resume)
```

Train + test nhanh khi đã có sẵn `data_split/` (bỏ qua split & dựng KG):

```bash
python run_code.py
```

## Cấu hình — chỉ sửa 2 file

Không cần chỉnh path bên trong từng module nữa. Mọi thứ tập trung ở:

- **`pipeline_config.py`** — đường dẫn dữ liệu, thư mục output, Neo4j, bật/tắt
  OntoKG (`USE_ONTOKG`). Cần đổi tối thiểu:
  - `RAW_DATA_CSV` — CSV gốc (cột: `title, summary, content, publish_time,
    topic, cleaned_keywords`).
  - `PRETRAINED_VEC` — FastText `.bin` tiếng Việt (`cc.vi.300.bin`).
  - `NEO4J_PASSWORD` — nếu dùng OntoKG.
  - `USE_ONTOKG` — `True`/`False` (mặc định `False` = baseline, không cần Neo4j).
- **`conf.py`** — hyperparameter của model (d_model, num_layers, batch_size, lr…).

Mọi giá trị trong `pipeline_config.py` đều override được bằng biến môi trường
cùng tên, ví dụ: `RAW_DATA_CSV=/duong/dan/khac.csv python main.py`.

## Các giai đoạn pipeline

| Stage   | Việc làm                                            | Output chính |
|---------|------------------------------------------------------|--------------|
| split   | CSV gốc → train/val/test/trainval (+ `article_id`)   | `data_split/*.csv` |
| ontokg  | Module 1–7 (NER, linking, KG, ontology, KGE) + Module 8 nạp Neo4j | `data/kge/entity_embeddings.pt`, Neo4j |
| train   | Train TransMTL trên trainval (truy vấn KG qua Module 9 nếu bật) | `Results_Score/BestModel.pt` |
| test    | Đánh giá ROUGE-1/2/L + Keyphrase P/R/F1              | `*_test_results.txt` |

> OntoKG được xây trên **trainval**; `get_loaders` tự chia held-out test bằng
> cùng seed=42 nên `test` đánh giá đúng phần dữ liệu model chưa từng train.
> Muốn đánh giá trên file `test.csv` riêng: đổi `TEST_DATA_CSV = TEST_CSV`
> trong `pipeline_config.py`.

## Yêu cầu (khi bật OntoKG)

- Neo4j đang chạy ở `NEO4J_URI` (mặc định `bolt://localhost:7687`).
- Các module 1–7 dùng `underthesea`, `transformers`, `pykeen`, `umap`, `hdbscan`…
