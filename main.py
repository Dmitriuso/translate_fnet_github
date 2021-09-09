import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator, TabularDataset
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

import numpy as np
import random
import math
import time
import pkbar

from encoder import Encoder
from decoder import Decoder
from seq2seq import Seq2Seq

from utils.pre_processing import SRC, TRG, train_data, valid_data, test_data
from inference import TranslationInference

work_dir = Path(__file__).parent.resolve()
models_dir = work_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_model_path = models_dir / "translate_fnet_de_en_trf.pt"

BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.src),
    sort_within_batch=True,
    device=device)

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 2048
DEC_PF_DIM = 2048
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM,
              HID_DIM,
              ENC_LAYERS,
              ENC_PF_DIM,
              ENC_DROPOUT,
              device)

dec = Decoder(OUTPUT_DIM,
              HID_DIM,
              DEC_LAYERS,
              DEC_PF_DIM,
              DEC_DROPOUT,
              device)

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


model.apply(initialize_weights)

LEARNING_RATE = 0.0005

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)


def train(model, iterator, optimizer, criterion, clip, kbar):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg[:, :-1])

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        # print('-' * 50)
        # print(f'trg : {trg.shape}')
        # print(f'output : {output.shape}')
        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]
        max_len = max([trg.shape[0], output.shape[0]])

        loss = criterion(nn.functional.pad(output, (0, max_len - output.shape[0])),
                         nn.functional.pad(trg, (0, max_len - trg.shape[0])))

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

        kbar.update(i, values=[("loss", epoch_loss / float(i + 1))])

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, kbar):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg[:, :-1])

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]
            max_len = max([trg.shape[0], output.shape[0]])

            loss = criterion(nn.functional.pad(output, (0, max_len - output.shape[0])),
                             nn.functional.pad(trg, (0, max_len - trg.shape[0])))

            epoch_loss += loss.item()

        kbar.add(1, values=[("val_loss", epoch_loss / len(iterator))])

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def main():
    N_EPOCHS = 50
    CLIP = 1

    best_valid_loss = float('inf')
    writer = SummaryWriter()

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        kbar = pkbar.Kbar(target=len(train_iterator), epoch=epoch, num_epochs=N_EPOCHS, width=8, always_stateful=False)

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP, kbar)
        valid_loss = evaluate(model, valid_iterator, criterion, kbar)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model, output_model_path)

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        writer.add_scalar("Train Loss", train_loss, epoch+1)
        writer.add_scalar("Train PPL", math.exp(train_loss), epoch+1)
        writer.add_scalar("Val. Loss", valid_loss, epoch+1)
        writer.add_scalar("Val. PPL", math.exp(valid_loss), epoch+1)

        model_eval_path = output_model_path
        model_eval = torch.load(model_eval_path)
        test_loss = evaluate(model_eval, valid_iterator, criterion, kbar)
        print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

        sentence = "I habe aber alles verstanden."
        translation = TranslationInference(
            model_path=model_eval_path,
            src_field=SRC,
            trg_field=TRG,
            max_len=50,
            device=device
        )
        print(f'prediction : {translation.inference(sentence)}')


if __name__ == '__main__':
    main()
