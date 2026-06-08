# run_code.py
"""
Tiện ích NHANH: chỉ train rồi test (KHÔNG split, KHÔNG dựng OntoKG lại).
Dùng khi đã có sẵn data_split/ (và data/kge/ nếu bật OntoKG).

Tương đương:  python main.py --stage train  &&  python main.py --stage test
Mọi cấu hình lấy từ pipeline_config.py + conf.py.

Chạy toàn bộ pipeline (split -> ontokg -> train -> test) thì dùng:  python main.py
"""
import time
import argparse

import pipeline_config as P
from main import stage_train, stage_test


def _build_args():
    ap = argparse.ArgumentParser(description="Train + Test nhanh (TransMTL)")
    ap.add_argument("--pretrained-vec", dest="pretrained_vec", default=P.PRETRAINED_VEC)
    ap.add_argument("--save-path", dest="save_path", default=P.SAVE_PATH)
    ap.add_argument("--neo4j-pass", dest="neo4j_pass", default=P.NEO4J_PASSWORD)
    ap.add_argument("--use-ontokg", dest="use_ontokg", action="store_true", default=None)
    ap.add_argument("--no-ontokg", dest="use_ontokg", action="store_false")
    ap.set_defaults(use_ontokg=None)   # None => lấy từ pipeline_config.USE_ONTOKG
    args = ap.parse_args()
    if args.use_ontokg is None:
        args.use_ontokg = P.USE_ONTOKG
    return args


if __name__ == "__main__":
    args = _build_args()
    P.ensure_dirs()

    print("Starting training...")
    t0 = time.time()
    stage_train(args)
    print(f"Training completed in {time.time() - t0:.1f}s")

    print("\n=== Starting testing ===")
    stage_test(args)
