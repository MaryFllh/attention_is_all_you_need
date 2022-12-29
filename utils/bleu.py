import math
import numpy as np

from collections import Counter


def compute_bleu(reference, candidate):
    """
    Creates 1-4 grams of the reference and candidate
    and appends the precision of the candidate's to
    a list

    Args:
        reference(str): The ground truth phrase
        candidate(str): The candidate phrase

    Returns:
        bleu_score(int): bleu score across n-grams
    """
    precision = []
    reference_words = reference.split()
    candidate_words = candidate.split()
    for n in range(1, 5):
        reference_ngram = Counter(
            [
                " ".join(reference_words[i : i + n])
                for i in range(len(reference_words) + 1 - n)
            ]
        )
        candidate_ngram = Counter(
            [
                " ".join(candidate_words[i : i + n])
                for i in range(len(candidate_words) + 1 - n)
            ]
        )
        if not candidate_ngram or not reference_ngram:
            continue
        overlap = sum((reference_ngram & candidate_ngram).values())
        precision.append(overlap / sum(candidate_ngram.values()))

    brevity_penalty = (
        1
        if len(candidate) >= len(reference)
        else math.exp(1 - len(candidate) / len(reference))
    )
    bleu_score = brevity_penalty * np.mean(precision) * 100
    return bleu_score
