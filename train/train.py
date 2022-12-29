import torch
import torch.nn as nn
import config

from torch.optim import Adam

from transformer import Transformer
from train.load_data import train_iter, val_iter, en_vocab, fr_vocab
from optimizer import Optimizer

from utils.bleu import compute_bleu
from utils.utils import idx_to_sentence

# from label_smoothing import LabelSmoothing # TODO: add later


def iniatialize_weights(model):
    return nn.init.kaiming_uniform(model.weight.data)


def train(model, data_iterator, device, optimizer, criterion, clip):
    model.train()
    train_loss = 0
    for i, (fr_batch, en_batch) in enumerate(data_iterator):
        src = fr_batch.to(device)
        trg = en_batch.to(device)
        optimizer.zerograd()

        output = model(src, trg[:, :-1])
        reshaped_output = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(reshaped_output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        train_loss += loss.item()
        print("Iteration: ", i, "with loss: ", loss.item())
    return train_loss / len(data_iterator)


def validate(model, data_iterator, device, criterion, trg_vocab):
    model.eval()
    val_loss = 0
    batch_bleu_score = []
    with torch.no_grad():
        for _, (fr_batch, en_batch) in enumerate(data_iterator):
            src = fr_batch.to(device)
            trg = en_batch.to(device)

            output = model(src, trg[:, :-1])
            reshaped_output = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(reshaped_output, trg)
            val_loss += loss.item()

            bleu_scores = 0
            for j in range(len(fr_batch)):
                trg_sentence = idx_to_sentence(en_batch[j], trg_vocab)
                predict = output[j].max(dim=1)[1]
                predicted_sentence = idx_to_sentence(predict, trg_vocab)
                bleu_score = compute_bleu(
                    reference=trg_sentence, candidate=predicted_sentence
                )
                bleu_scores += bleu_score
            batch_bleu_score.append(bleu_scores / len(fr_batch))
        val_bleu_score = sum(batch_bleu_score) / len(batch_bleu_score)

    return val_loss, val_bleu_score


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_vocab_size = len(en_vocab)
    src_pad_idx = en_vocab["<pad>"]
    trg_vocab_size = len(fr_vocab)
    trg_pad_idx = fr_vocab["<pad>"]

    model = Transformer(
        src_vocab_size,
        src_pad_idx,
        trg_vocab_size,
        trg_pad_idx,
        config.d_model,
        config.max_seq_len,
        config.heads_num,
        config.forward_expansion,
        config.dropout,
        config.layers_num,
    )

    model.apply(iniatialize_weights)

    optimizer = Optimizer(
        config.d_model,
        Adam(
            params=model.parameters(),
            lr=config.init_learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        ),
    )

    criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)
