"""
Uses code from pytorch's tutorial page:
https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html
"""
import io
import torch

from collections import Counter
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torchtext.vocab import vocab
from torchtext.utils import download_from_url, extract_archive
from torch.nn.utils.rnn import pad_sequence


class Data:
    def __init__(self, url_base, train_urls, val_urls, test_urls):
        self.train_file_paths = [
            extract_archive(download_from_url(url_base + url))[0] for url in train_urls
        ]
        self.val_file_paths = [
            extract_archive(download_from_url(url_base + url))[0] for url in val_urls
        ]
        self.test_file_paths = [
            extract_archive(download_from_url(url_base + url))[0] for url in test_urls
        ]

    def get_tokenizer(self):
        self.en_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
        self.fr_tokenizer = get_tokenizer("spacy", language="fr_core_news_sm")

    def build_vocab(self, filepath, tokenizer):
        counter = Counter()
        with io.open(filepath, encoding="utf8") as f:
            for string_ in f:
                counter.update(tokenizer(string_))
        v = vocab(counter, specials=["<unk>", "<pad>", "<sos>", "<eos>"])
        v.set_default_index(-1)
        self.pad_idx = v["<pad>"]
        self.sos_idx = v["<sos>"]
        self.eos_idx = v["<eos>"]
        return v

    def preprocess_data(self, filepaths, en_vocab, fr_vocab):
        raw_fr_iter = iter(io.open(filepaths[0], encoding="utf8"))
        raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
        data = []
        for (raw_fr, raw_en) in zip(raw_fr_iter, raw_en_iter):
            fr_tensor_ = torch.tensor(
                [fr_vocab[token] for token in self.fr_tokenizer(raw_fr)],
                dtype=torch.long,
            )
            en_tensor_ = torch.tensor(
                [en_vocab[token] for token in self.en_tokenizer(raw_en)],
                dtype=torch.long,
            )
            data.append((fr_tensor_, en_tensor_))
        return data

    def generate_batches(self, data_batch):
        fr_batch, en_batch = [], []
        for (fr_item, en_item) in data_batch:
            fr_batch.append(
                torch.cat(
                    [
                        torch.tensor([self.sos_idx])
                        + fr_item
                        + torch.tensor([self.eos_idx])
                    ],
                    dim=0,
                )
            )
            en_batch.append(
                torch.cat(
                    [
                        torch.tensor([self.sos_idx])
                        + en_item
                        + torch.tensor([self.eos_idx])
                    ],
                    dim=0,
                )
            )

        fr_batch = pad_sequence(fr_batch, padding_value=self.pad_idx)
        en_batch = pad_sequence(en_batch, padding_value=self.pad_idx)
        return fr_batch, en_batch

    def load_data(self, data, batch_size):
        return DataLoader(
            data, batch_size=batch_size, shuffle=True, collate_fn=self.generate_batches
        )
