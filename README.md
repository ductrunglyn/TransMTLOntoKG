# OntoKG‑TransMTL — Tài liệu kỹ thuật chi tiết

> Hệ thống **học đa nhiệm tiếng Việt** (tóm tắt văn bản + trích xuất từ khóa) **tích hợp đồ thị tri thức có bản thể học (OntoKG)**.
> Tài liệu này mô tả đầy đủ kiến trúc, vai trò + **sản phẩm (output)** của từng file/module, cách chạy cho từng trường hợp, định dạng dữ liệu và cách gỡ lỗi — đủ để người (hoặc một AI khác) tiếp quản và sửa chữa.

---

## 0. TL;DR — chạy nhanh

```bash
# Cấu hình: sửa pipeline_config.py (đường dẫn dữ liệu, Neo4j, USE_ONTOKG) + conf.py (hyperparameter)

# A) Baseline TransMTL (KHÔNG cần Neo4j): đặt USE_ONTOKG = False
python run_transmtl.py            # split -> train -> test

# B) Dùng OntoKG (cần Neo4j): đặt USE_ONTOKG = True
python run_ontokg.py              # BƯỚC 1: xây KG (chạy 1 lần, chậm)
python run_transmtl.py            # BƯỚC 2: train + test (chạy lại nhiều lần)

# C) Chạy tất cả 1 lệnh / từng giai đoạn
python main.py --use-ontokg                 # split->ontokg->train->test
python main.py --stage train --no-ontokg    # chỉ 1 giai đoạn
python main.py --skip-existing               # resume, bỏ bước đã có output
```

---

## 1. Tổng quan kiến trúc

Hệ thống gồm **2 phần lớn**:

```
            ┌─────────────────────── PHẦN 1: OntoKG (module 1→9) ───────────────────────┐
 CSV tin tức → preprocess → NER+concept → entity linking → relation → KG → ontology → KGE → Neo4j
            └───────────────────────────────────┬───────────────────────────────────────┘
                                                 │  (truy vấn subgraph theo article_id, module 9)
            ┌────────────────────────────────────▼──────────────────────────────────────┐
 CSV tin tức → Text Encoder ──► H_tok ──► Gated Fusion ◄── E_kg ◄── Graph Encoder(R‑GCN+GATv2)
            →                     │                                                       
            →                 H'_tok ──► MMoE ──► ┬─► Decoder + Copy ──► Tóm tắt          
            →                                     └─► CRF/BIOES        ──► Từ khóa         
            └────────────────────── PHẦN 2: TransMTL (model) ──────────────────────────────┘
```

- **Thiết kế "additive":** nếu `kg_batch=None` (hoặc `USE_ONTOKG=False`), nhánh đồ thị bị tắt và model chạy như **baseline TransMTL thuần** → phục vụ ablation công bằng.
- **Giao tiếp giữa 2 phần:** OntoKG nạp vào Neo4j; lúc train/test, `OntoKGBridge` truy vấn subgraph theo `article_id` của từng bài (module 9) → đưa vào model.

---

## 2. Cây thư mục & vai trò từng file

```
TransMTLOntoKG/
├── pipeline_config.py     ★ CẤU HÌNH TRUNG TÂM: đường dẫn dữ liệu, Neo4j, USE_ONTOKG, thư mục output
├── conf.py                ★ HYPERPARAMETER model: d_model, layers, lr, batch, MMoE...
│
├── run_ontokg.py          ★ BƯỚC 1: xây OntoKG (split + module 1‑8). Chạy 1 lần.
├── run_transmtl.py        ★ BƯỚC 2: train + test TransMTL. Chạy lại nhiều lần.
├── main.py                ★ Orchestrator 1 lệnh: --stage {all,split,ontokg,train,test}, --use-ontokg/--no-ontokg, --skip-existing
├── split_dataset.py       Chia CSV gốc -> data_split/{train,val,test,trainval}.csv (+ cột article_id)
│
├── TransMTL_v2.py         ★ KIẾN TRÚC MODEL: Encoder/Decoder, MMoE, Copy, CRF, fusion OntoKG, beam/greedy decode
├── train_v2.py            Vòng lặp huấn luyện: train_model(), train_one_epoch(), validate()
├── testing_v2.py          Đánh giá test: run_test() -> ROUGE + keyphrase P/R/F1 (+ OntoKG bridge)
├── data_utils.py          Dataset + DataLoader: MultiTaskDataset, CollateCPU, get_loaders() (trả cả article_id)
├── preprocessing.py       Tiền xử lý: FastText embedding, BPE tokenizer, BIOES, ids<->text, keyphrase
├── utils_v2.py            Hàm loss + tiện ích: summary loss, key loss, gộp trọng số đa nhiệm, load checkpoint
├── evaluation.py          Chỉ số đánh giá: keyphrase P/R/F1, ROUGE summary
│
├── ontokg_fusion.py       ★ CẦU NỐI MODEL↔KG: GraphEncoder (R‑GCN+GATv2), GatedFusion, encode_kg_batch
├── ontokg_data_bridge.py  ★ OntoKGBridge: truy vấn Neo4j theo article_id -> kg_batch cho model
│
├── OntoKG/                ★ PIPELINE XÂY KG (9 module, xem Mục 7)
│   ├── module1_preprocess.py        Chuẩn hóa văn bản, tách câu/từ, POS
│   ├── module2_ner_concept.py       NER (PhoBERT/ELECTRA) + trích khái niệm
│   ├── module3_entity_linking.py    Liên kết thực thể 4 tầng + giải đồng nghĩa (alias)
│   ├── module4_relation_extraction.py  Trích bộ ba (h,r,t) ràng buộc ontology
│   ├── module5_kg_construction.py   Dựng KG (RDF/Turtle, NetworkX, entity_index, pykeen TSV)
│   ├── module6_ontology_learning.py UMAP + HDBSCAN -> sinh lớp khái niệm
│   ├── module7_kge_training.py      KGE TransE (PyKEEN) + trộn PhoBERT -> entity_embeddings.pt
│   ├── module8_neo4j_loader.py      Nạp KG vào Neo4j
│   ├── module9_neo4j_retrieval.py   Truy vấn subgraph (dùng khi train/test)
│   └── aliases.json                 Từ điển alias thủ công (HN->Hà Nội, TP.HCM->...)
│
├── data_split/   (gitignored)  train/val/test/trainval.csv
├── data/         (gitignored)  toàn bộ artifact OntoKG (jsonl, kg/, kge/, ontology/)
└── Results_Score/(gitignored)  checkpoint model + *_test_results.txt
```

---

## 3. Cấu hình

### 3.1 `pipeline_config.py` — bảng điều khiển (sửa ở đây)
| Biến | Ý nghĩa |
|---|---|
| `RAW_DATA_CSV` | CSV gốc (cột: `title, summary, content, publish_time, topic, cleaned_keywords`) |
| `PRETRAINED_VEC` | FastText `.bin` tiếng Việt (cc.vi.300.bin) |
| `USE_ONTOKG` | **Công tắc thường** True/False (không phụ thuộc biến môi trường) |
| `NEO4J_URI/USER/PASSWORD/DATABASE` | Kết nối Neo4j (chỉ cần khi OntoKG bật) |
| `SPLIT_DIR / DATA_DIR / SAVE_PATH` | Thư mục output |
| `TRAIN_DATA_CSV / TEST_DATA_CSV` | Mặc định = `trainval.csv` (get_loaders tự chia held‑out theo seed=42) |
| `OKG_DEVICE` | cuda/cpu cho các module nặng |
| `ensure_dirs()`, `ontokg_env()` | tạo thư mục; xuất biến môi trường cho subprocess module 1‑8 |

> Mọi giá trị (trừ `USE_ONTOKG`) override được bằng biến môi trường cùng tên.

### 3.2 `conf.py` — hyperparameter model
`D_MODEL=300, NUM_LAYER=4, NUM_HEADS=6, DFF=1024, DROPOUT=0.3, BATCH_SIZE=24, NUM_EPOCHS=100, LR_BASE=5e-5, LABEL_SMOOTHING=0.15, SIZE_VOCAB=40000`, nhóm MMoE (`USE_MMOE, MMOE_NUM_EXPERTS=4,...`), `LABELS` (BIOES: O/B/I/E/S), `IGNORE_INDEX=-100`, `PAD_IDX=0`.

### 3.3 Biến môi trường đặc biệt (tăng tốc / điều khiển module 3)
| Env | Tác dụng |
|---|---|
| `OKG_EMB_MATCH=0` | Tắt so khớp embedding (Tầng 3) ở entity linking — nhanh nhất |
| `OKG_WD_ALIASES=0` | Tắt nạp alias Wikidata |
| `OKG_ALIAS_JSON=path` | Đổi file từ điển alias (mặc định `OntoKG/aliases.json`) |
| `OKG_DATA_DIR / OKG_INPUT_CSV / OKG_DEVICE` | main.py tự set cho module 1‑8 |

---

## 4. Luồng dữ liệu end‑to‑end (data flow)

```
RAW_DATA_CSV
  │ split_dataset.py
  ▼
data_split/{train,val,test,trainval}.csv         (thêm cột article_id = "<split>_000123")
  │ module1 (input = trainval.csv)
  ▼ data/preprocessed_articles.jsonl
  │ module2
  ▼ data/module2_ner_concept.jsonl
  │ module3  (+ data/entity_registry.pkl, data/wikidata_cache.json)
  ▼ data/module3_entity_linked.jsonl
  │ module4
  ▼ data/module4_triples.jsonl  (+ module4_enriched.jsonl)
  │ module5
  ▼ data/kg/{kg_global.ttl, kg_networkx.pkl, entity_index.pkl, pykeen_triples.tsv}
  │ module6                         module7
  ▼ data/ontology/*                 ▼ data/kge/{entity_embeddings.pt, entity_to_idx.json, idx_to_uri.json}
  │ module8 (nạp Neo4j từ kg/ + kge/ + module4_triples)
  ▼ Neo4j DB
  │ module9 (truy vấn lúc train/test, qua OntoKGBridge)
  ▼ kg_batch -> TransformerMTL
```

---

## 5. CÁCH CHẠY — từng trường hợp

### 5.1 Hai file runner đơn giản
| Lệnh | Làm gì |
|---|---|
| `python run_ontokg.py` | **Bước 1** — split (nếu chưa) + module 1‑8 (xây KG, nạp Neo4j). Tự `--skip-existing`. |
| `python run_transmtl.py` | **Bước 2** — split (nếu chưa) + train + test. Đọc `USE_ONTOKG` để bật/tắt KG. |

> Cả hai gọi `make_args(cfg_module=P)` nên **đổi `import pipeline_config` sang config khác** (vd `pipeline_config_baseline`) sẽ đi xuyên suốt mọi stage.

### 5.2 `main.py` — orchestrator linh hoạt
```bash
python main.py                       # theo USE_ONTOKG trong config
python main.py --use-ontokg          # ép bật KG cho lần chạy này
python main.py --no-ontokg           # ép baseline
python main.py --stage split         # chỉ chia dữ liệu
python main.py --stage ontokg --use-ontokg   # chỉ xây KG (module 1‑8)
python main.py --stage train         # chỉ train
python main.py --stage test          # chỉ test
python main.py --skip-existing       # resume (bỏ qua module/đầu ra đã có)
# Override đường dẫn:
python main.py --data-csv X.csv --pretrained-vec cc.vi.300.bin --save-path Results_Score/exp1.pt --neo4j-pass pw
```

### 5.3 Chạy thủ công từng module OntoKG (debug)
```bash
# Cần set OKG_DATA_DIR + OKG_INPUT_CSV (main.py tự set; chạy lẻ thì export tay)
export OKG_DATA_DIR=./data OKG_INPUT_CSV=./data_split/trainval.csv OKG_DEVICE=cuda
python OntoKG/module1_preprocess.py
python OntoKG/module2_ner_concept.py
... (đến module8)
```

### 5.4 Các kịch bản ablation phổ biến (qua env/config)
```bash
USE_ONTOKG=... # đặt trong pipeline_config.py
OKG_EMB_MATCH=0 python run_ontokg.py     # xây KG nhanh, bỏ so khớp embedding
OKG_WD_ALIASES=0 python run_ontokg.py    # không nạp alias Wikidata
# Baseline vs OntoKG: chỉ đổi USE_ONTOKG = False/True rồi chạy run_transmtl.py
```

---

## 6. Chi tiết các file MODEL / DATA / UTIL (Phần 2 — TransMTL)

### 6.1 `TransMTL_v2.py` — `class TransformerMTL`
Kiến trúc Transformer encoder–decoder đa nhiệm + tích hợp OntoKG.
- **Khối con:** `PositionalEncoding`, `MultiHeadAttention`, `FeedForward`, `EncoderLayer/Encoder`, `DecoderLayer/Decoder_Sum`, `CopyGate`, `MMoE`.
- **Khởi tạo quan trọng:** `emb_matrix` (FastText), `word2idx/idx2word`, `num_key_labels=5` (BIOES), `use_mmoe`, `use_copy=True`, `use_ontokg`, `kg_in_dim=768`, `kg_num_relations=9`. `cls_idx=<sos>`, `sep_idx=<eos>`.
- **Đầu ra (heads):**
  - Tóm tắt: `Decoder_Sum` → `summary_logits`; nếu copy bật → `summary_log_probs` (pointer‑generator qua `CopyGate` + `_apply_copy`).
  - Từ khóa: `final_key_proj` (d_model→5) → emissions → **CRF** (`crf_decoder`, thư viện torchcrf): train trả `key_nll`, infer trả `key_decoded`.
- **`_apply_ontokg_fusion(enc_out_shared, kg_batch, device)`** — chỉ chạy khi `use_ontokg` và có `kg_batch`: `encode_kg_batch` → `GatedFusion`. Nếu không → trả nguyên `enc_out_shared` (baseline).
- **`forward(inp, tar, labels, task="both", training, kg_batch=None)`** → dict gồm `summary_logits`/`summary_log_probs`, `key_nll`/`key_decoded`, (gates nếu MMoE).
- **Sinh chuỗi:** `greedy_decode_batch(inp, max_len, kg_batch)` và `beam_search_generate_batch(inp, max_len, beam_size, len_penalty, n_gram_block, kg_batch)` (auto‑regressive, có copy, n‑gram blocking).

### 6.2 `train_v2.py`
- **`train_model(...)`** (chữ ký dài, gọi từ main/run): tạo loader (`get_loaders`), nạp FastText, dựng `TransformerMTL`, khởi tạo **`OntoKGBridge`**, optimizer **theo nhóm tham số** (summary/shared/crf, lr nhân hệ số), `weight_logits` (trọng số đa nhiệm học được, optimizer riêng), scheduler `ReduceLROnPlateau`, PCGrad. Tham số OntoKG: `use_ontokg, entity_emb_path, entity_idx_path, neo4j_uri, neo4j_pass`.
- **`train_one_epoch(...)`** / **`validate(...)`**: unpack batch **8‑tuple** (có `article_ids`); `kg_batch = bridge.build_kg_batch(article_ids)`; gọi model với `kg_batch`. Validate tính ROUGE qua greedy decode.

### 6.3 `testing_v2.py` — `run_test(...)`
Tạo loader, dựng model (khớp `use_ontokg`), nạp checkpoint (`load_checkpoint_state`, `strict=False`), **bridge OntoKG**, lặp test → beam/greedy sinh tóm tắt → **ROUGE‑1/2/L** (rouge_score) + map subword→word → **keyphrase P/R/F1** (`evaluate_keyphrase_lists`). Lưu `<ckpt>_test_results.txt`, trả dict kết quả.

### 6.4 `data_utils.py`
- **`MultiTaskDataset`**: đọc CSV, BPE tokenize (vocab_subword), sinh nhãn **BIOES** cho từ khóa (ánh xạ keyword→span token), lưu **`article_ids`** (cột `article_id`, fallback `row_xxxxx`).
- **`CollateCPU`**: pad batch → trả **8 phần tử**: `src, summary_ids, attn, labels, raw_texts, token_maps, word_texts, article_ids`.
- **`get_loaders(data_path, len_in, len_out, num_workers, batch_size, val_ratio=0.2, test_ratio=0.2, seed=42, min_freq=3, vocab_size)`** → train/val/test loader + vocab + word2idx/idx2word + `ds` (chứa `.tokenizer`). **Tự chia 60/20/20** từ CSV truyền vào (seed cố định ⇒ test held‑out tái lập được).

### 6.5 `preprocessing.py`
`seed_everything`, `load_fasttext_bin_embeddings(word2idx, bin_path, d_model, pad_idx)` (**KHÔNG còn synonym**), BPE tokenizer (`ensure_tokenizer_for_csv`), ánh xạ keyword→span (`find_token_span_for_keyword`), `subword_labels_to_word_labels`, `ids_to_text`, `convert_tags_to_keyphrases`.

### 6.6 `utils_v2.py`
`compute_summary_loss_from_logits` (CE + label smoothing), `compute_summary_loss_from_logprobs` (NLL cho copy), `compute_key_loss_from_raw`, `compute_entropy_regularizer` (MMoE gate), **`combine_task_losses(loss_sum, key_nll, weight_logits, temp)`** (trọng số đa nhiệm học được), `ensure_decoder_sos`, `load_checkpoint_state`.

### 6.7 `evaluation.py`
`evaluate_keyphrase_lists` (P/R/F1 theo tập cụm từ, chuẩn hóa + unique), `evaluate_summaries` (ROUGE).

### 6.8 `ontokg_fusion.py` (cầu nối model↔KG)
- **`GraphEncoder(in_dim=768, d_model, num_relations=9, num_bases=4, dropout)`**: `Linear(768→d) → RGCNConv×2 → GATv2Conv`. Cần `torch_geometric`; nếu thiếu → fallback MLP. `forward(x, edge_index, edge_type) → (N, d_model)`.
- **`GatedFusion(d_model, num_heads, dropout)`**: cross‑attention (query=H_tok, key/value=E_kg) + **cổng sigmoid** `gate*attn_out` + residual + LayerNorm → `H'_tok`.
- **`encode_kg_batch(graph_encoder, kg_batch, d_model, device)`**: mã hóa list subgraph (mỗi sample 1 subgraph hoặc None) → tensor padded `E_kg (B,N_max,d)` + `padding_mask`.

### 6.9 `ontokg_data_bridge.py` — `class OntoKGBridge`
`__init__(uri, user, password, d_model, enabled)`; nếu `enabled` → tạo `Neo4jRetriever` (module9, dim=768). **`build_kg_batch(article_ids)`** → list subgraph torch (hoặc None nếu tắt/không có). `close()`.

---

## 7. Chi tiết PIPELINE OntoKG — INPUT → SẢN PHẨM từng module

> Các đường dẫn dưới đây tương đối với `DATA_DIR` (mặc định `./data`).

### Module 1 — `module1_preprocess.py` (Tiền xử lý)
- **Input:** `trainval.csv` (qua env `OKG_INPUT_CSV`).
- **Xử lý:** chuẩn hóa Unicode/HTML, tách câu, tách từ (underthesea), POS (**chỉ `full_text_pos`** — đã tối ưu), gộp `full_text` = title+summary+content.
- **Sản phẩm:** `preprocessed_articles.jsonl` (mỗi dòng = 1 bài: tokens, sentences, `full_text_pos`, topic_list, cleaned_keywords, `article_id`), `preprocess_errors.jsonl`.

### Module 2 — `module2_ner_concept.py` (NER + khái niệm)
- **Input:** `preprocessed_articles.jsonl`.
- **Xử lý:** NER bằng **PhoBERT/NlpHUST‑ELECTRA pipeline (GPU+fp16)** + underthesea, chuẩn hóa nhãn về `{PER,ORG,LOC,TIME,EVENT,MISC}`; trích khái niệm (noun phrase từ `full_text_pos` + so khớp embedding khái niệm, **GPU**).
- **Sản phẩm:** `module2_ner_concept.jsonl` (mỗi bài: `ner_entities[]` {surface,label}, `concept_mentions[]`), `module2_errors.jsonl`.

### Module 3 — `module3_entity_linking.py` (Liên kết + giải đồng nghĩa) ★
- **Input:** `module2_ner_concept.jsonl`.
- **Xử lý — 4 tầng (`_link_one`):** (0) chuẩn hóa **alias thủ công** (`aliases.json`) → (1) khớp bề mặt registry → (2) **Wikidata** (chỉ LOC/PER/ORG, có cache + nạp alias Wikidata) → (3) **so khớp embedding** cùng nhãn (vector hoá, chỉ proper‑noun) → (4) tạo URI mới. Gán URI duy nhất, gộp biến thể đồng nghĩa.
- **Sản phẩm:** `module3_entity_linked.jsonl` (entities có `uri/uri_source/wikidata_id`), `entity_registry.pkl` (URI→{surface_forms, label, embedding, count}), `wikidata_cache.json`.

### Module 4 — `module4_relation_extraction.py` (Quan hệ)
- **Input:** `module3_entity_linked.jsonl`.
- **Xử lý:** (A) metadata triple tự động `(article, rdf:type, Article)`, `(article, belongsTo, topic)`; (B) quan hệ ngữ nghĩa giữa thực thể **ràng buộc domain/range theo ontology** (loại liên kết phi logic). 9 loại quan hệ: `OCCURS_AT, OCCURS_ON, ORGANIZED_BY, PARTICIPATES_IN, LOCATED_IN, MANAGES, HAS_PART, CAUSED_BY, RELATED_TO`.
- **Sản phẩm:** `module4_triples.jsonl` (tất cả bộ ba), `module4_enriched.jsonl` (record gốc + `triples`).

### Module 5 — `module5_kg_construction.py` (Dựng KG)
- **Input:** `module3_entity_linked.jsonl` + `module4_triples.jsonl`.
- **Sản phẩm (thư mục `kg/`):** `kg_global.ttl` (RDF/Turtle), `kg_networkx.pkl` (MultiDiGraph), **`entity_index.pkl`** (uri→{embedding, info, stats}), **`pykeen_triples.tsv`** (head\trelation\ttail — cho module 7).

### Module 6 — `module6_ontology_learning.py` (Học ontology)
- **Input:** `kg/entity_index.pkl`.
- **Xử lý:** lọc entity (≥min_count) → **UMAP** giảm chiều → **HDBSCAN** phân cụm → gợi ý lớp khái niệm mới.
- **Sản phẩm (thư mục `ontology/`):** `ontology_v1.1.json` (cấu trúc lớp), `cluster_report.txt`, `cluster_matrix.npy` (toạ độ UMAP để visualize).

### Module 7 — `module7_kge_training.py` (KGE) ★
- **Input:** `kg/pykeen_triples.tsv` + `kg/entity_index.pkl`.
- **Xử lý:** huấn luyện **TransE (PyKEEN)** dim 256 → chiếu 768; **trộn với biểu diễn bề mặt PhoBERT** (`combine_alpha=0.5`).
- **Sản phẩm (thư mục `kge/`):** **`entity_embeddings.pt` (N×768) — ĐẦU VÀO CHO TRANSMTL**, `entity_to_idx.json`, `idx_to_uri.json`, `kge_model/` (checkpoint PyKEEN).

### Module 8 — `module8_neo4j_loader.py` (Nạp Neo4j)
- **Input:** `kg/entity_index.pkl`, `kge/entity_embeddings.pt`, `kge/entity_to_idx.json`, `module4_triples.jsonl`; creds qua env `NEO4J_*`.
- **Sản phẩm:** Neo4j DB gồm node `:Entity {uri, embedding[768], label, ...}`, `:Article {article_id}`, `:Topic {name}` và các quan hệ ngữ nghĩa + `HAS_ENTITY`. (Chạy `reset=True` xoá dữ liệu cũ.)

### Module 9 — `module9_neo4j_retrieval.py` (Truy vấn — dùng lúc train/test)
- **`Neo4jRetriever`** với `SEMANTIC_RELATIONS` (9 loại) + `RELATION_TO_ID`.
- **`get_article_subgraph(article_id)`** → {uris, node features, edges} ; **`subgraph_to_torch(sg)`** → dict `{x:(N,768), edge_index:(2,E), edge_type:(E,)}` cho GraphEncoder; thêm `link_entity_by_vector`, `get_entity_neighbors`.
- **Lưu ý:** chỉ bài có trong Neo4j (tức **train+val**) mới có subgraph; bài test (article_id `test_*`) chưa nạp ⇒ trả subgraph rỗng ⇒ model chạy như baseline cho bài test (xem Hạn chế).

---

## 8. Định dạng dữ liệu chuẩn

- **CSV gốc:** `title, summary, content, publish_time, topic, cleaned_keywords` (+ `article_id` do `split_dataset` thêm).
- **Batch (CollateCPU, 8‑tuple):** `src, summary_ids, attn, labels(BIOES id), raw_texts, token_maps, word_texts, article_ids`.
- **Nhãn từ khóa (BIOES):** `O=0,B=1,I=2,E=3,S=4` (conf.LABELS), `IGNORE_INDEX=-100`, `PAD_IDX=0`.
- **subgraph torch:** `{x:(N,768) float, edge_index:(2,E) long, edge_type:(E,) long∈[0,8]}`.

---

## 9. Hàm loss & vòng huấn luyện
- **Tóm tắt:** CrossEntropy (label smoothing) hoặc **NLL** khi copy bật.
- **Từ khóa:** **CRF negative log‑likelihood** (`key_nll`).
- **Gộp đa nhiệm:** `combine_task_losses(loss_sum, key_nll, weight_logits, temp)` — `weight_logits` là tham số **học được** (softmax), optimizer riêng (KHÔNG phải uncertainty‑loss Kendall).
- **(tùy chọn)** entropy regularizer cho gate MMoE; **PCGrad** chống xung đột gradient.
- Optimizer AdamW theo nhóm (summary/shared/crf, lr nhân hệ số), scheduler ReduceLROnPlateau, chọn checkpoint theo val.

---

## 10. Gỡ lỗi thường gặp (troubleshooting)
| Triệu chứng | Nguyên nhân & cách xử lý |
|---|---|
| `FileNotFoundError: trainval.csv` | Chưa split. Chạy `run_transmtl.py`/`run_ontokg.py` (tự split) hoặc kiểm tra `RAW_DATA_CSV`. |
| Chạy `run_transmtl.py` vẫn ra "TransMTL + OntoKG" dù muốn baseline | `USE_ONTOKG` là **công tắc thường** trong `pipeline_config.py`; nếu dùng config riêng, `make_args(cfg_module=P)` đã đi xuyên suốt. Đảm bảo import đúng config. |
| Kết nối Neo4j thất bại (USE_ONTOKG=True) | Kiểm tra Neo4j chạy ở `NEO4J_URI` + đúng `NEO4J_PASSWORD`. |
| Module 3 **rất chậm / số entity bùng nổ** | Đã sửa O(N²) (vector hoá + bucket theo nhãn + bỏ MISC). Có thể `OKG_EMB_MATCH=0` để nhanh tối đa; `--skip-existing` để resume. |
| Module 2 chậm | NER/embedding đã đưa lên GPU+fp16; cần `OKG_DEVICE=cuda`. Có thể `use_phonlp=False`. |
| `ImportError: torch_geometric` | GraphEncoder cần PyG (`pip install torch_geometric`). Thiếu thì fallback MLP (mất R‑GCN/GAT). |
| float16 overflow / AMP | Chạy thật trên GPU; kiểm tra masked_fill dùng finfo.min/2. |
| GitHub từ chối push (file >100MB) | `data_split/*.csv` lỡ commit. Đã có `.gitignore` chặn `data_split/`, `data/`, `*.csv`, `*.pt`...; xoá khỏi lịch sử bằng `git filter-branch`. |

---

## 11. HẠN CHẾ / CHƯA IMPLEMENT (quan trọng cho người tiếp quản)
Bản thảo bài báo mô tả vài thứ **CODE CHƯA CÓ** — đừng nhầm là đã có:
- **Streaming "Frozen Model – Evolving Graph" / `module10_streaming_inference.py`:** **chưa tồn tại** (chỉ được nhắc trong comment). Inductive Router / Knowledge Buffer chưa có.
- **EWC** (Elastic Weight Consolidation) và **tái huấn luyện KGE/online ontology theo chu kỳ:** chưa có; module 6/7 chạy **một lần (batch)**.
- **Topic Classifier** (đầu tác vụ thứ 3): không có; topic chỉ là node/quan hệ trong KG. Model chỉ có **2 đầu**: summary + keyphrase.
- **Copy trỏ thẳng vào E_kg / Semantic Typing Head:** copy hiện copy từ **token nguồn** (đã được làm giàu tri thức qua fusion), không copy trực tiếp nhãn KG.
- **synonym‑aware embedding:** đã **gỡ** khỏi `preprocessing.py` (giải đồng nghĩa nay nằm ở module 3).
- **Đánh giá test có OntoKG:** bài test chưa được nạp vào Neo4j ⇒ subgraph rỗng ⇒ KG không đóng góp ở test (cần liên kết thực thể bài test tới KG đã xây, hoặc làm module10).

---

## 12. Phụ thuộc
`torch, transformers, underthesea, rouge-score, pandas, numpy, neo4j, pykeen, umap-learn, hdbscan, sentencepiece, torchcrf, torch_geometric, rdflib, networkx`. (Baseline TransMTL chỉ cần nhóm đầu; phần OntoKG cần đủ.)
