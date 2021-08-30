from typing import Iterator
from encoder import Encoder
from decoder import Decoder
from torchtext.data import Field, BucketIterator
from torchtext import data
import random
import math
import time
import torch
import torch.nn as nn
import numpy as np
from utils import split_trg
from seq2seq import Seq2Seq
import pkbar
import dill
import pytorch_lightning as pl

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class PunctuationModel(pl.LightningModule):
    def __init__(self):
        super.__init__()
        HID_DIM = 256
        ENC_LAYERS = 3
        DEC_LAYERS = 3
        ENC_HEADS = 8
        DEC_HEADS = 8
        ENC_PF_DIM = 512
        DEC_PF_DIM = 512
        ENC_DROPOUT = 0.1
        DEC_DROPOUT = 0.1
        BATCH_SIZE = 128

        LEARNING_RATE = 0.0005
        N_EPOCHS = 8
        CLIP = 1

        self.enc = Encoder(INPUT_DIM, 
                HID_DIM, 
                ENC_LAYERS, 
                ENC_HEADS, 
                ENC_PF_DIM, 
                ENC_DROPOUT, 
                device)
