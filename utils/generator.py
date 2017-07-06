"""Text data reader.

This can handle variable-length text data."""

import os
import io
import glob
import random

import torch
import torch.utils.data as D


class TextFileReader(object):
    def __init__(self, path, shuffle_files=False):
        self.path = path
        self.shuffle = shuffle_files

    def __iter__(self):
        return self.generate()

    def generate(self):
        if os.path.isfile(self.path):
            filenames = [os.path.abspath(self.path)]
        elif os.path.isdir(self.path):
            filenames = glob.glob(os.path.join(self.path, "*.txt"))
        else:
            raise ValueError("Path does not exist: {}".format(self.path))

        if self.shuffle:
            random.shuffle(filenames)

        for filename in filenames:
            with io.open(filename, "r", encoding="utf-8") as f:
                for line in f:
                    yield line.strip()


class SentenceTargetGenerator(object):
    def __init__(self, sents, max_length):
        self.sents = sents
        self.max_length = max_length

    def __iter__(self):
        return self.generate()

    def generate(self):
        for line in self.sents:

            sent, target = line.split("\t")
            sent, target = sent.split(), target.split()

            if len(sent) > self.max_length or len(target) > self.max_length:
                continue

            yield sent, target


class SentenceTargetLabelGenerator(object):
    def __init__(self, sents, max_length):
        self.sents = sents
        self.max_length = max_length

    def __iter__(self):
        return self.generate()

    def generate(self):
        for line in self.sents:

            sent, target, label = line.split("\t")
            sent, target = sent.split(), target.split()

            if len(sent) > self.max_length or len(target) > self.max_length:
                continue

            yield sent, target, int(label)


def create_data_loader_label(sent_targets, batch_size, preprocessor,
                             shuffle=False, pin_memory=True,
                             add_input_noise=True):
    def _collate_fn(batch):
        sents, targets, labels = zip(*batch)
        sents, sents_lens = preprocessor(sents)
        targets, targets_lens = preprocessor(targets)
        labels = torch.LongTensor(labels)

        if add_input_noise:
            preprocessor.add_noise(sents, sents_lens)

        return sents, sents_lens, targets, targets_lens, labels

    data_loader = D.DataLoader(sent_targets, batch_size, shuffle, num_workers=2,
                               collate_fn=_collate_fn, pin_memory=pin_memory)

    return data_loader


def create_data_loader(sent_targets, batch_size, preprocessor,
                       shuffle=False, pin_memory=True, add_input_noise=True):
    def _collate_fn(batch):
        sents, targets = zip(*batch)
        sents, sents_lens = preprocessor(sents)
        targets, targets_lens = preprocessor(targets)

        if add_input_noise:
            preprocessor.add_noise(sents, sents_lens)

        return sents, sents_lens, targets, targets_lens

    data_loader = D.DataLoader(sent_targets, batch_size, shuffle, num_workers=2,
                               collate_fn=_collate_fn, pin_memory=pin_memory)

    return data_loader


class SentenceGenerator(object):
    """Simple Sentence Data Generator
    
    Generates sentences preprocessed by batches.
    """

    def __init__(self, sents, vocab, batch_size, max_length, preprocessor,
                 pin_memory=True, allow_residual=True):
        self.sents = sents
        self.vocab = vocab
        self.batch_size = batch_size
        self.max_length = max_length
        self.preprocessor = preprocessor
        self.pin_memory = pin_memory
        self.allow_residual = allow_residual

    def __iter__(self):
        return self.generate()

    def generate(self):
        batch = []

        for line in self.sents:
            sent = line.split()

            if len(sent) > self.max_length:
                continue

            batch.append(sent)

            if len(batch) < self.batch_size:
                continue

            batch_sents, sents_lens = self.preprocessor(batch)

            if self.pin_memory:
                batch_sents, sents_lens = \
                    batch_sents.pin_memory(), sents_lens.pin_memory()

            yield batch_sents, sents_lens

            del batch
            batch = []

        if self.allow_residual and batch:
            batch_sents, sents_lens = self.preprocessor(batch)

            if self.pin_memory:
                batch_sents, sents_lens = \
                    batch_sents.pin_memory(), sents_lens.pin_memory()

            yield batch_sents, sents_lens


class AutoencodingDataGenerator(object):
    """Sentence Data Generator
    
    Generates pairs sentences and their target predictions. The input data is
    an iterator over either (1) individual sentences or (2) a pair of sentence
    and its target sentence separated by a tab character. These can be mixed.
    The sentences must be a string of words separated by spaces.
    """

    def __init__(self, sents, vocab, batch_size, max_length, preprocessor,
                 pin_memory=True, add_input_noise=True, allow_residual=True):
        self.sents = sents
        self.vocab = vocab
        self.batch_size = batch_size
        self.max_length = max_length
        self.preprocessor = preprocessor
        self.pin_memory = pin_memory
        self.add_input_noise = add_input_noise
        self.allow_residual = allow_residual

    def __iter__(self):
        return self.generate()

    def generate(self):
        batch = []

        for line in self.sents:

            if "\t" in line:
                sent, target = line.split("\t")
                sent, target = sent.split(), target.split()
            else:
                line = line.split()
                sent, target = line, line

            if len(sent) > self.max_length or len(target) > self.max_length:
                continue

            batch.append((sent, target))

            if len(batch) < self.batch_size:
                continue

            batch_sents, batch_targets = zip(*batch)
            batch_sents, sents_lens = self.preprocessor(batch_sents)
            batch_targets, targets_lens = self.preprocessor(batch_targets)

            if self.add_input_noise:
                self.preprocessor.add_noise(batch_sents, sents_lens)

            if self.pin_memory:
                batch_sents, sents_lens = batch_sents.pin_memory(), sents_lens.pin_memory()
                batch_targets, targets_lens = batch_targets.pin_memory(), targets_lens.pin_memory()

            yield batch_sents, sents_lens, batch_targets, targets_lens

            del batch
            batch = []

        if self.allow_residual and batch:
            batch_sents, batch_targets = zip(*batch)
            batch_sents, sents_lens = self.preprocessor(batch_sents)
            batch_targets, targets_lens = self.preprocessor(batch_targets)

            if self.add_input_noise:
                self.preprocessor.add_noise(batch_sents, sents_lens)

            if self.pin_memory:
                batch_sents, sents_lens = batch_sents.pin_memory(), sents_lens.pin_memory()
                batch_targets, targets_lens = batch_targets.pin_memory(), targets_lens.pin_memory()

            yield batch_sents, sents_lens, batch_targets, targets_lens


class ContextDataGenerator(object):
    def __init__(self, sents, vocab, batch_size, max_length, preprocessor,
                 n_before=0, n_after=0, predict_self=True, pin_memory=True,
                 add_input_noise=True):
        assert n_before or n_after or predict_self

        self.sents = sents
        self.batch_size = batch_size
        self.max_length = max_length
        self.vocab = vocab
        self.preprocessor = preprocessor
        self.n_before = n_before
        self.n_after = n_after
        self.predict_self = predict_self
        self.pin_memory = pin_memory
        self.add_input_noise = add_input_noise

    def __iter__(self):
        return self.generate()

    def generate(self):
        n_bef, n_aft = self.n_before, self.n_after
        batch = []
        f_batch_size = self.batch_size + self.n_before + self.n_after

        for line in self.sents:
            words = line.split()

            if len(words) > self.max_length:
                continue

            batch.append(words)

            if len(batch) < f_batch_size:
                continue

            batch, lens = self.preprocessor(batch)

            if n_bef or n_aft:
                inp_data = batch[n_bef:len(batch) - n_aft].clone()
                inp_lens = lens[n_bef:len(batch) - n_aft].clone()
            else:
                inp_data = batch.clone()
                inp_lens = lens.clone()

            if self.add_input_noise:
                self.preprocessor.add_noise(inp_data, inp_lens)

            if n_bef or n_aft:
                _offset = n_bef + n_aft + 1
                out_data = [batch[i:i + _offset].unsqueeze(1)
                            for i in range(self.batch_size)]
                out_lens = [lens[i:i + _offset].unsqueeze(1)
                            for i in range(self.batch_size)]
                out_data = torch.cat(out_data, 1)
                out_lens = torch.cat(out_lens, 1)

                if not self.predict_self:
                    if n_bef and n_aft:
                        splits = out_data[:n_bef], out_data[-n_aft:]
                        splits_lens = out_lens[:n_bef], out_lens[-n_aft:]
                        out_data = torch.cat(splits, 0)
                        out_lens = torch.cat(splits_lens, 0)
                    elif n_bef:
                        out_data = out_data[:n_bef]
                        out_lens = out_lens[:n_bef]
                    elif n_aft:
                        out_data = out_data[-n_aft:]
                        out_lens = out_lens[-n_aft:]
            else:
                out_data = batch.unsqueeze(0)
                out_lens = lens.unsqueeze(0)

            inp_data, inp_lens = inp_data.contiguous(), inp_lens.contiguous()
            out_data, out_lens = out_data.contiguous(), out_lens.contiguous()

            if self.pin_memory:
                inp_data, inp_lens = inp_data.pin_memory(), inp_lens.pin_memory()
                out_data, out_lens = out_data.pin_memory(), out_lens.pin_memory()

            yield inp_data, inp_lens, out_data, out_lens

            del batch
            batch = []

        del batch