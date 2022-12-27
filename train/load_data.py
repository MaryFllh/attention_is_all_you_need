from data.data import Data
from config import BATCH_SIZE


url_base = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"
train_urls = ("train.fr.gz", "train.en.gz")
val_urls = ("val.fr.gz", "val.en.gz")
test_urls = ("test_2016_flickr.fr.gz", "test_2016_flickr.en.gz")


data = Data(url_base, train_urls, val_urls, test_urls)
data.get_tokenizer()
fr_vocab = data.build_vocab(data.train_file_paths[0], data.fr_tokenizer)
en_vocab = data.build_vocab(data.train_file_paths[1], data.en_tokenizer)

train_data = data.preprocess_data(data.train_file_paths, en_vocab, fr_vocab)
val_data = data.preprocess_data(data.val_file_paths, en_vocab, fr_vocab)
test_data = data.preprocess_data(data.test_file_paths, en_vocab, fr_vocab)

train_iter = data.load_data(train_data, BATCH_SIZE)
val_iter = data.load_data(val_data, BATCH_SIZE)
test_iter = data.load_data(test_data, BATCH_SIZE)
