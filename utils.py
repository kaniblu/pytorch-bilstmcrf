import re
import collections


class Vocabulary(object):
    def __init__(self):
        self.f2i = {}
        self.i2f = {}

    def add(self, w, ignore_duplicates=True):
        if w in self.f2i:
            if not ignore_duplicates:
                raise ValueError(f"'{w}' already exists")

            return self.f2i[w]
        idx = len(self.f2i)
        self.f2i[w] = idx
        self.i2f[idx] = w

        return self.f2i[w]

    def remove(self, w):
        """
        Removes a word from the vocab. The indices are unchanged.
        """
        if w not in self.f2i:
            raise ValueError(f"'{w}' does not exist.")

        index = self.f2i[w]
        del self.f2i[w]
        del self.i2f[index]

    def reconstruct_indices(self):
        """
        Reconstruct word indices in case of word removals.
        Vocabulary does not handle empty indices when words are removed,
          hence it need to be told explicity about when to reconstruct them.
        """
        words = list(self.f2i.keys())
        del self.i2f, self.f2i
        self.f2i, self.i2f = {}, {}

        for i, w in enumerate(words):
            self.f2i[w] = i
            self.i2f[i] = w

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.i2f[item]
        elif isinstance(item, str):
            return self.f2i[item]
        elif hasattr(item, "__iter__"):
            return [self[ele] for ele in item]
        else:
            raise ValueError(f"Unknown type: {type(item)}")

    def __contains__(self, item):
        return item in self.f2i or item in self.i2f

    def __len__(self):
        return len(self.f2i)


def populate_vocab(words, vocab, cutoff=None):
    if cutoff is not None:
        counter = collections.Counter(words)
        words, _ = zip(*counter.most_common(cutoff))

    for w in words:
        vocab.add(w)

    return vocab


class PeriodChecker(object):
    PATTERN = re.compile(r"([0-9]+)(e|i|)")

    def __init__(self, s: str):
        m = self.PATTERN.match(s)
        assert m is not None, \
            f"String must be in the format of [integer]['e'/'i'/'s']."

        self.scalar = int(m.group(1))
        self.unit = m.group(2)

    def __call__(self, epochs=None, iters=None, steps=None):
        assert epochs or iters or steps, \
            "One of the items must be provided."

        if self.unit == "e":
            if epochs is not None:
                return epochs % self.scalar == 0
        elif self.unit == "i":
            if iters is not None:
                return iters % self.scalar == 0
        elif self.unit == "s":
            if steps is not None:
                return steps % self.scalar == 0
        else:
            raise ValueError(f"Unrecognized unit: {self.unit}")

        return False


class FileReader(object):
    def __init__(self, path):
        self.path = path

    def words(self):
        with open(self.path, "r") as f:
            for line in f:
                for w in line.strip().split():
                    yield w

    def sents(self):
        with open(self.path, "r") as f:
            for line in f:
                yield line.strip().split()