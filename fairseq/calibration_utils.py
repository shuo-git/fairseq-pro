from nltk.util import ngrams
from collections import Counter
import numpy as np
import itertools

def label_n_gram(reference, hypothesis, n, pad_idx=1):
    for i in range(len(reference)-1, -1, -1):
        if reference[i] == pad_idx:
            reference.pop(i)
        else:
            break
    labels = []
    ref_n_grams = Counter(ngrams(reference, n)) if len(reference) >= n else Counter()
    for i in range(0, len(hypothesis) - n + 1):
        pattern = tuple(hypothesis[i:i + n])
        if ref_n_grams[pattern] > 0:
            labels.append(1)
            ref_n_grams[pattern] -= 1
        else:
            labels.append(0)
    return labels