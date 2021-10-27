import tokenizers
import transformers
from transformers import AutoTokenizer
import torch

class config:
    MAX_LEN = 384
    STRIDE = 128
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 16
    EPOCHS = 5
    BERT_PATH = "bert_base_multilingual_uncased"
    MODEL_CONFIG = transformers.BertConfig.from_pretrained(BERT_PATH)
    MODEL_PATH = "pytorch_model.bin"
    DEVICE = torch.device("cuda")
    TRAINING_FILE = "../input/train.csv"
    TOKENIZER = AutoTokenizer.from_pretrained(BERT_PATH)
