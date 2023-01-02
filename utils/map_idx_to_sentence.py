def idx_to_sentence(indices, vocab):
    """
    Looks up the corresponding word of each
    index and joins them in a sentence

    Args:
        indices(list): list of indices
        vocab(vocab): the vocab object

    Returns:
        string: the sequence of corresponding words
    """
    words = []
    for i in indices:
        word = vocab.itos[i]
        if not word.startswith("<"):
            words.append(word)

    return " ".join(words)
