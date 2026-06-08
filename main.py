# main.py
"""
==========================================================================
 PIPELINE END-TO-END: TransMTL + OntoKG
==========================================================================
Điều phối TOÀN BỘ quy trình trong 1 file, theo 4 giai đoạn:

  [A] CHIA DỮ LIỆU      : CSV gốc -> train/val/test/trainval
  [B] XÂY ONTOKG        : Module 1-7 trên trainval -> entity_embeddings.pt
                          Module 8 -> nạp vào Neo4j
  [C] TRAIN TransMTL    : trên trainval, truy vấn Neo4j qua Module 9
                          (đóng băng sau khi hội tụ)
  [D] TEST              : offline batch  + streaming (Module 10)

Cách dùng:
  # Chạy hết, CÓ OntoKG
  python main.py --stage all --use_ontokg --neo4j_pass password

  # Ablation baseline (KHÔNG OntoKG)
  python main.py --stage all

  # Chạy từng giai đoạn
  python main.py --stage split
  python main.py --stage ontokg --use_ontokg --neo4j_pass password
  python main.py --stage train  --use_ontokg --neo4j_pass password
  python main.py --stage test   --use_ontokg --neo4j_pass password

Yêu cầu thư mục làm việc chứa:
  module1_preprocess.py ... module10_streaming_inference.py
  TransMTL_v2.py, train_v2.py, testing_v2.py, data_utils.py,
  preprocessing.py, conf.py, ontokg_fusion.py, ontokg_data_bridge.py,
  split_dataset.py
==========================================================================
"""
import os
import sys
import time
import argparse
import subprocess

import conf as cfg
from split_dataset import split_dataset


# ──────────────────────────────────────────────────────────
# Tiện ích chạy script con (Module 1-8) tuần tự
# ──────────────────────────────────────────────────────────
def run_module(script: str, env: dict = None):
    """Chạy 1 module python, dừng pipeline nếu lỗi."""
    print(f"\n>>> Chạy {script} ...")
    t0 = time.time()
    result = subprocess.run([sys.executable, script], env=env)
    if result.returncode != 0:
        print(f"!!! {script} LỖI (returncode={result.returncode}). Dừng pipeline.")
        sys.exit(result.returncode)
    print(f"<<< {script} xong sau {time.time() - t0:.1f}s")


# ──────────────────────────────────────────────────────────
# [A] Chia dữ liệu
# ──────────────────────────────────────────────────────────
def stage_split(args):
    print("\n" + "=" * 64)
    print("[A] CHIA DỮ LIỆU TRAIN/VAL/TEST")
    print("=" * 64)
    split_dataset(args.data_csv, args.split_dir,
                  val_ratio=0.2, test_ratio=0.2, seed=42)


# ──────────────────────────────────────────────────────────
# [B] Xây OntoKG (Module 1-8) trên trainval
# ──────────────────────────────────────────────────────────
def stage_ontokg(args):
    print("\n" + "=" * 64)
    print("[B] XÂY ONTOKG (Module 1-8) trên trainval")
    print("=" * 64)

    # Truyền đường dẫn trainval cho Module 1 qua biến môi trường
    env = os.environ.copy()
    env["ONTOKG_INPUT_CSV"] = os.path.join(args.split_dir, "trainval.csv")
    env["NEO4J_PASSWORD"]   = args.neo4j_pass

    # Module 1-7: trích xuất + KGE
    for m in [
        "OntoKG/module1_preprocess.py",
        "OntoKG/module2_ner_concept.py",
        "OntoKG/module3_entity_linking.py",
        "OntoKG/module4_relation_extraction.py",
        "OntoKG/module5_kg_construction.py",
        "OntoKG/module6_ontology_learning.py",
        "OntoKG/module7_kge_training.py",
    ]:
        run_module(m, env)

    # Module 8: nạp Neo4j (chỉ khi dùng OntoKG)
    if args.use_ontokg:
        run_module("OntoKG/module8_neo4j_loader.py", env)
    else:
        print("(Bỏ qua Module 8 — chạy baseline không OntoKG)")


# ──────────────────────────────────────────────────────────
# [C] Train TransMTL
# ──────────────────────────────────────────────────────────
def stage_train(args):
    print("\n" + "=" * 64)
    print(f"[C] TRAIN TransMTL {'(+OntoKG)' if args.use_ontokg else '(baseline)'}")
    print("=" * 64)
    from train_v2 import train_model

    trainval_csv = os.path.join(args.split_dir, "trainval.csv")
    ontokg_kwargs = dict(
        use_ontokg      = args.use_ontokg,
        entity_emb_path = args.entity_emb if args.use_ontokg else None,
        entity_idx_path = args.entity_idx if args.use_ontokg else None,
        neo4j_uri       = args.neo4j_uri if args.use_ontokg else None,
        neo4j_pass      = args.neo4j_pass if args.use_ontokg else None,
    )

    train_model(
        trainval_csv, args.save_path, cfg.PAD_IDX, cfg.LABEL_SMOOTHING,
        args.pretrained_vec, cfg.NUM_LAYER, cfg.D_MODEL, cfg.NUM_HEADS, cfg.DFF,
        cfg.LEN_IN, cfg.LEN_OUT, cfg.DROPOUT, cfg.FREEZE_EMBEDDINGS,
        cfg.MMOE_NUM_EXPERTS, cfg.MMOE_EXPERT_HIDDEN, cfg.MMOE_GATE_HIDDEN,
        cfg.MMOE_DROPOUT, cfg.MMOE_USE_RESIDUAL, cfg.MMOE_GATE_TEMPERATURE,
        cfg.MMOE_RESIDUAL_SCALE, cfg.LR_BASE, cfg.WEIGHT_DECAY, cfg.NUM_EPOCHS,
        cfg.WARMUP_MMOE_EPOCHS, cfg.MMOE_ENTROPY_LAMBDA, cfg.CLIP_NORM,
        cfg.IGNORE_INDEX, cfg.NUM_WORKERS, cfg.BATCH_SIZE, cfg.USE_MMOE,
        cfg.DEVICE, cfg.SIZE_VOCAB,
        **ontokg_kwargs,
    )


# ──────────────────────────────────────────────────────────
# [D] Test offline + streaming
# ──────────────────────────────────────────────────────────
def stage_test(args):
    print("\n" + "=" * 64)
    print("[D] TEST OFFLINE (batch)")
    print("=" * 64)
    from testing_v2 import run_test

    test_csv = os.path.join(args.split_dir, "test.csv")
    ontokg_kwargs = dict(
        use_ontokg      = args.use_ontokg,
        entity_emb_path = args.entity_emb if args.use_ontokg else None,
        entity_idx_path = args.entity_idx if args.use_ontokg else None,
        neo4j_uri       = args.neo4j_uri if args.use_ontokg else None,
        neo4j_pass      = args.neo4j_pass if args.use_ontokg else None,
    )

    run_test(
        args.save_path, test_csv, cfg.LEN_IN, cfg.LEN_OUT, cfg.NUM_WORKERS,
        cfg.BATCH_SIZE, cfg.D_MODEL, cfg.PAD_IDX, args.pretrained_vec,
        cfg.NUM_LAYER, cfg.NUM_HEADS, cfg.DFF, cfg.DROPOUT, cfg.FREEZE_EMBEDDINGS,
        cfg.MMOE_NUM_EXPERTS, cfg.MMOE_EXPERT_HIDDEN, cfg.MMOE_GATE_HIDDEN,
        cfg.MMOE_DROPOUT, cfg.MMOE_USE_RESIDUAL, cfg.MMOE_GATE_TEMPERATURE,
        cfg.MMOE_RESIDUAL_SCALE, cfg.IGNORE_INDEX, cfg.DEVICE, cfg.USE_MMOE,
        cfg.SIZE_VOCAB,
        **ontokg_kwargs,
    )

    # Streaming test (Module 10) — chỉ khi có OntoKG
    if args.use_ontokg:
        print("\n" + "=" * 64)
        print("[D2] TEST STREAMING (Module 10)")
        print("=" * 64)
        print("Chạy: python OntoKG/module10_streaming_inference.py")
        print("(Cần plug model TransMTL đã train vào transmtl_infer_fn — xem Module 10)")


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "split", "ontokg", "train", "test"])
    ap.add_argument("--data_csv",
                    default="/home/hoangtrung/hdtrungoi/CoKhanh/Data/tintuc_gen_final.csv")
    ap.add_argument("--pretrained_vec",
                    default="/home/hoangtrung/hdtrungoi/CoKhanh/word_embedding_pretrain/cc.vi.300.bin")
    ap.add_argument("--split_dir", default="./data_split")
    ap.add_argument("--save_path", default="./Results_Score/BestModel.pt")
    # OntoKG
    ap.add_argument("--use_ontokg", action="store_true")
    ap.add_argument("--entity_emb", default="./data/kge/entity_embeddings.pt")
    ap.add_argument("--entity_idx", default="./data/kge/entity_to_idx.json")
    ap.add_argument("--neo4j_uri",  default="bolt://localhost:7687")
    ap.add_argument("--neo4j_pass", default="password")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    t_start = time.time()
    if args.stage in ("all", "split"):
        stage_split(args)
    if args.stage in ("all", "ontokg"):
        stage_ontokg(args)
    if args.stage in ("all", "train"):
        stage_train(args)
    if args.stage in ("all", "test"):
        stage_test(args)

    print("\n" + "=" * 64)
    print(f"PIPELINE HOÀN TẤT sau {time.time() - t_start:.1f}s")
    print("=" * 64)


if __name__ == "__main__":
    main()