import torch
from pathlib import Path

### Data
VOCAB_SIZE = 30_000
VOCAB_DIR = Path(__file__).parent/"bookcorpus_vocab"
MAX_LEN = 512

### Architecture
N_LAYERS = 6
N_HEADS = 6
HIDDEN_SIZE = 384
MLP_SIZE = 384 * 4

### Regularization
DROP_PROB = 0.1

### Optimizer
MAX_LR = 1e-4
BETA1 = 0.9
BETA2 = 0.999
WEIGHT_DECAY = 0


### Training
N_GPUS = torch.cuda.device_count()
if N_GPUS > 0:
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
N_WORKERS = 4
CKPT_DIR = Path(__file__).parent/"pretrain/checkpoints"
N_CKPT_SAMPLES = 100_000
### Masked Language Model
SELECT_PROB = 0.15
MASK_PROB = 0.8
RANDOMIZE_PROB = 0.1
