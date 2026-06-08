# run_code.py
import time

# from train import train_model
# from testing import run_test
from train_v2 import train_model
from testing_v2 import run_test
import conf as cfg

# path data
data_path = "/home/hoangtrung/hdtrungoi/CoKhanh/TransMTL_OntoKG/data_split/trainval.csv"
pretrained_vec_path = "/home/hoangtrung/hdtrungoi/CoKhanh/word_embedding_pretrain/cc.vi.300.bin"
synonym_path = "/home/hoangtrung/hdtrungoi/CoKhanh/Data/vietnamese_synonyms.json"

# path weight
save_score_path = './Results_Score/BestModel_NoMMoE_NoSyn.pt'

# Lấy các biến config
pad_idx = cfg.PAD_IDX
label_smoothing = cfg.LABEL_SMOOTHING
device = cfg.DEVICE
num_layers = cfg.NUM_LAYER
d_model = cfg.D_MODEL
num_heads = cfg.NUM_HEADS
dff = cfg.DFF
len_in = cfg.LEN_IN
len_out = cfg.LEN_OUT
dropout = cfg.DROPOUT
freeze_embeddings = cfg.FREEZE_EMBEDDINGS
mmoe_num_experts = cfg.MMOE_NUM_EXPERTS
mmoe_expert_hidden = cfg.MMOE_EXPERT_HIDDEN
mmoe_gate_hidden = cfg.MMOE_GATE_HIDDEN
mmoe_dropout = cfg.MMOE_DROPOUT
mmoe_use_residual = cfg.MMOE_USE_RESIDUAL
mmoe_gate_temperature = cfg.MMOE_GATE_TEMPERATURE
mmoe_residual_scale = cfg.MMOE_RESIDUAL_SCALE
lr=cfg.LR_BASE
weight_decay=cfg.WEIGHT_DECAY
num_epochs = cfg.NUM_EPOCHS
warmup_mmoe = cfg.WARMUP_MMOE_EPOCHS
entropy_lambda = cfg.MMOE_ENTROPY_LAMBDA
clip_norm = cfg.CLIP_NORM
ignore_index = cfg.IGNORE_INDEX
num_workers = cfg.NUM_WORKERS
batch_size = cfg.BATCH_SIZE
size_vocab = cfg.SIZE_VOCAB

# Sử dụng MMoE
use_mmoe = cfg.USE_MMOE

# Sử dụng từ đồng nghĩa
use_synonym = cfg.USE_SYNONYM

# Huấn luyện model
print("Starting training...")
start_time = time.time()
if __name__ == "__main__":
    train_model(data_path, save_score_path, pad_idx, label_smoothing, pretrained_vec_path, num_layers, 
                d_model, num_heads, dff, len_in, len_out, dropout, freeze_embeddings, mmoe_num_experts, 
                mmoe_expert_hidden, mmoe_gate_hidden, mmoe_dropout, mmoe_use_residual, mmoe_gate_temperature, 
                mmoe_residual_scale, lr, weight_decay, num_epochs, warmup_mmoe, entropy_lambda, clip_norm, 
                ignore_index, num_workers, batch_size, use_mmoe, device, use_synonym, synonym_path, size_vocab)
print("Training completed.")
end_time = time.time()
print("===========================================")
print("Thời gian huấn luyện xong:", end_time - start_time, "giây")


# Test model
print("===========================================")
print("Starting testing with Score weight ...")
if __name__ == "__main__":
    run_test(save_score_path, data_path, len_in, len_out, num_workers, batch_size, d_model, pad_idx, 
             pretrained_vec_path, num_layers, num_heads, dff, dropout, freeze_embeddings, mmoe_num_experts, 
             mmoe_expert_hidden, mmoe_gate_hidden, mmoe_dropout, mmoe_use_residual, mmoe_gate_temperature,
             mmoe_residual_scale, ignore_index, device, use_mmoe, use_synonym, synonym_path, size_vocab)
