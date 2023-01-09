import torch
import torch.nn as nn
import config

from torch.optim import Adam

from transformer import Transformer
from utils.load_data import train_iter, val_iter, en_vocab, fr_vocab
from utils.optimizer import Optimizer

from utils.bleu import compute_bleu
from utils.map_idx_to_sentence import idx_to_sentence

# from label_smoothing import LabelSmoothing # TODO: add later


def iniatialize_weights(model):
    if hasattr(model, "weight") and model.weight.dim() > 1:
        nn.init.kaiming_uniform(model.weight.data)


def train(model, data_iterator, device, optimizer, criterion, clip):
    model.train()
    train_loss = 0
    for i, (fr_batch, en_batch) in enumerate(data_iterator):
        src = fr_batch.to(device)
        trg = en_batch.to(device)
        optimizer.optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        reshaped_output = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(reshaped_output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        train_loss += loss.item()
        print("Iteration: ", i + 1, "with loss: ", loss.item())
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


def run(
    model,
    train_data_iterator,
    val_data_iterator,
    device,
    optimizer,
    criterion,
    clip,
    trg_vocab,
):
    train_losses, val_losses, bleu_scores = [], [], []
    best_loss = float("Inf")
    for epoch in range(config.EPOCHS):
        train_loss = train(
            model, train_data_iterator, device, optimizer, criterion, clip
        )
        val_loss, bleu_score = validate(
            model, val_data_iterator, device, criterion, trg_vocab
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_loss:
            torch.save(model.state_dict(), f"saved/model_{val_loss}.pt")
            best_loss = val_loss

        f = open("result/train_loss.txt", "w")
        f.write(str(train_losses))
        f.close()

        f = open("result/bleu.txt", "w")
        f.write(str(bleu_scores))
        f.close()

        f = open("result/val_loss.txt", "w")
        f.write(str(val_losses))
        f.close()

        print(f"Train loss at epoch {epoch} is: {train_loss}")
        print(f"Val loss at epoch {epoch} is {val_loss}")
        print(f"Bleu score at epoch {epoch} is {bleu_score}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_vocab_size = len(fr_vocab)
    src_pad_idx = en_vocab["<pad>"]
    trg_vocab_size = len(en_vocab)
    trg_pad_idx = fr_vocab["<pad>"]

    model = Transformer(
        src_vocab_size,
        src_pad_idx,
        trg_vocab_size,
        trg_pad_idx,
        config.D_MODEL,
        config.MAX_SEQ_LEN,
        config.HEADS_NUM,
        config.FORWARD_EXPANSION,
        config.DROPOUT,
        config.LAYERS_NUM,
    )

    model.apply(iniatialize_weights)

    optimizer = Optimizer(
        config.D_MODEL,
        Adam(
            params=model.parameters(),
            lr=config.INIT_LEARNING_RATE,
            betas=(config.BETA1, config.BETA2),
            eps=config.EPS,
        ),
        config.WARM_UP,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)
    run(
        model,
        train_iter,
        val_iter,
        device,
        optimizer,
        criterion,
        clip=config.CLIP,
        trg_vocab=fr_vocab,
    )
