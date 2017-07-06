import torch
import torch.nn as nn
import torch.nn.init as I
import torch.nn.utils.rnn as R
import torch.nn.functional as F
from torch.autograd import Variable


def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def log_sum_exp(vec, dim=0):
    _, idx = torch.max(vec, dim)
    max = torch.gather(vec, dim, idx[..., 0].unsqueeze(-1))
    max_exp = max.expand_as(vec)
    return max + torch.log(torch.sum(torch.exp(vec - max_exp), dim))


class CRF(nn.Module):
    def __init__(self, vocab):
        super(CRF, self).__init__()

        self.vocab = vocab
        self.n_labels = n_labels = len(vocab)
        self.bos_idx = self.vocab.f2i[self.vocab.bos]
        self.eos_idx = self.vocab.f2i[self.vocab.eos]
        self.transitions = nn.Parameter(torch.randn(n_labels, n_labels))

    def reset_parameters(self):
        I.xavier_normal(self.transitions.data)

    def forward(self, logits, lens):
        """
        Arguments:
            logits: [batch_size, seq_len, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        """
        batch_size, seq_len, n_labels = logits.size()
        alpha = logits.data.new(batch_size, self.n_labels).fill_(-10000)
        alpha[:, self.bos_idx] = 0
        alpha = Variable(alpha)
        c_lens = lens.clone()

        logits_t = logits.transpose(1, 0)
        for logit in logits_t:
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   *self.transitions.size())
            alpha_exp = alpha.unsqueeze(1).expand(batch_size,
                                                  *self.transitions.size())
            trans_exp = self.transitions.unsqueeze(0).expand_as(alpha_exp)
            mat = logit_exp + trans_exp + alpha_exp
            alpha_nxt = log_sum_exp(mat, 2)

            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(alpha)
            alpha = mask * alpha_nxt + (1 - mask) * alpha
            c_lens -= 1

        return log_sum_exp(alpha, 1).squeeze(-1)

    def viterbi_decode(self, logits, lens):
        """Borrowed from pytorch tutorial

        Arguments:
            logits: [batch_size, seq_len, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        """
        scores, paths = [], []

        for logit, l in zip(logits, lens):
            backpointers = []

            # Initialize the viterbi variables in log space
            init_vvars = logits.data.new(1, self.n_labels).fill_(-10000.)
            init_vvars[0][self.bos_idx] = 0

            # forward_var at step i holds the viterbi variables for step i-1
            forward_var = Variable(init_vvars)
            for feat in logit[:l]:
                bptrs_t = []  # holds the backpointers for this step
                viterbivars_t = []  # holds the viterbi variables for this step

                for next_tag in range(self.n_labels):
                    # next_tag_var[i] holds the viterbi variable for tag i at the
                    # previous step, plus the score of transitioning
                    # from tag i to next_tag.
                    # We don't include the emission scores here because the max
                    # does not depend on them (we add them in below)
                    next_tag_var = forward_var + self.transitions[next_tag]
                    best_tag_id = argmax(next_tag_var)
                    bptrs_t.append(best_tag_id)
                    viterbivars_t.append(next_tag_var[0][best_tag_id])
                # Now add in the emission scores, and assign forward_var to the set
                # of viterbi variables we just computed
                forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
                backpointers.append(bptrs_t)

            # Transition to STOP_TAG
            terminal_var = forward_var + self.transitions[self.eos_idx]
            best_tag_id = argmax(terminal_var)
            path_score = terminal_var[0][best_tag_id]

            # Follow the back pointers to decode the best path.
            best_path = [best_tag_id]
            for bptrs_t in reversed(backpointers):
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)
            # Pop off the start tag (we dont want to return that to the caller)
            start = best_path.pop()
            assert start == self.bos_idx  # Sanity check
            best_path.reverse()

            paths.append(best_path)
            scores.append(path_score)

        return scores, paths

    def transition_score(self, labels, lens):
        """
        Arguments:
             labels: [batch_size, seq_len] LongTensor
             lens: [batch_size] LongTensor
        """
        batch_size, seq_len = labels.size()

        # pad labels with <beginning of labels> tag (TODO: is it necessary?)
        # labels_pad = Variable(labels.data.new(1).fill_(self.bos_idx))
        # labels_pad = labels_pad.unsqueeze(-1).expand(batch_size, 1)
        # labels = torch.cat([labels_pad, labels], dim=1)

        # transpose transition matrix to let make 1st dimen prev timestep
        transitions = self.transitions.t()

        # obtain transition vector for each label in batch and timestep
        # (except the last ones)
        transitions_exp = transitions.unsqueeze(0).expand(batch_size,
                                                          *transitions.size())
        labels_l = labels[:, :-1]
        labels_lexp = labels_l.unsqueeze(-1).expand(*labels_l.size(),
                                                    transitions.size(0))
        transition_rows = torch.gather(transitions_exp, 1, labels_lexp)

        # obtain transition score from the transition vector for each label
        # in batch and timestep (except the first ones)
        labels_rexp = labels[:, 1:].unsqueeze(-1)
        tran_scores = torch.gather(transition_rows, 2, labels_rexp)
        tran_scores = tran_scores.squeeze()

        mask = sequence_mask(lens - 1).float()
        tran_scores = tran_scores * mask
        score = tran_scores.sum(1).squeeze(-1)

        return score


class BiLSTMCRF(nn.Module):
    def __init__(self, word_vocabs, label_vocab, word_dims, hidden_dim,
                 dropout_prob):
        super(BiLSTMCRF, self).__init__()

        assert len(word_vocabs) == len(word_dims)

        self.n_feats = len(word_vocabs)
        self.word_dim = sum(word_dims)
        self.word_vocabs = word_vocabs
        self.label_vocab = label_vocab
        self.word_dims = word_dims
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.is_cuda = False

        self.n_labels = n_labels = len(label_vocab)

        for i, (word_vocab, word_dim) in enumerate(zip(word_vocabs, word_dims)):
            setattr(self, "embeddings_{}".format(i),
                    nn.Embedding(len(word_vocab), word_dim))

        self.input_layer = nn.Linear(self.word_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim * 2, n_labels)
        self.crf = CRF(label_vocab)
        self.lstm = nn.LSTM(input_size=hidden_dim,
                            hidden_size=hidden_dim,
                            num_layers=1,
                            bidirectional=True,
                            dropout=dropout_prob,
                            batch_first=True)

    def cuda(self, *args, **kwargs):
        ret = super(BiLSTMCRF, self).cuda(*args, **kwargs)
        self.is_cuda = True
        return ret

    def cpu(self, *args, **kwargs):
        ret = super(BiLSTMCRF, self).cpu(*args, **kwargs)
        self.is_cuda = False
        return ret

    def reset_parameters(self):
        for i in range(self.n_feats):
            embeddings = getattr(self, "embeddings_{}".format(i))
            I.xavier_normal(embeddings.weight.data)

        I.xavier_normal(self.input_layer.weight.data)
        self.crf.reset_parameters()
        self.lstm.reset_parameters()

    def _run_rnn_packed(self, cell, x, x_lens, h=None):
        x_packed = R.pack_padded_sequence(x, x_lens.data.tolist(),
                                          batch_first=True)

        if h is not None:
            output, h = cell(x_packed, h)
        else:
            output, h = cell(x_packed)

        output, _ = R.pad_packed_sequence(output, batch_first=True)

        return output, h

    def _embeddings(self, xs):
        """Takes raw feature sequences and produces a single word embedding

        Arguments:
            xs: [n_feats, batch_size, seq_len] LongTensor

        Returns:
            [batch_size, seq_len, word_dim] FloatTensor 
        """
        n_feats, batch_size, seq_len = xs.size()

        assert n_feats == self.n_feats

        res = []
        for i, x in enumerate(xs):
            embeddings = getattr(self, "embeddings_{}".format(i))

            x = embeddings(x)
            res.append(x)

        x = torch.cat(res, 2)

        return x

    def _forward_bilstm(self, xs, lens):
        n_feats, batch_size, seq_len = xs.size()

        x = self._embeddings(xs)
        x = x.view(-1, self.word_dim)
        x = F.tanh(self.input_layer(x))
        x = x.view(batch_size, seq_len, self.hidden_dim)

        o, h = self._run_rnn_packed(self.lstm, x, lens)

        o = o.contiguous()
        o = o.view(-1, self.hidden_dim * 2)
        o = self.output_layer(o)
        o = o.view(batch_size, seq_len, self.n_labels)
        return o

    def _bilstm_score(self, logits, y, lens):
        y_exp = y.unsqueeze(-1)
        scores = torch.gather(logits, 2, y_exp).squeeze(-1)
        mask = sequence_mask(lens).float()
        scores = scores * mask
        score = scores.sum(1).squeeze(-1)

        return score

    def score(self, xs, y, lens, logits=None):
        if logits is None:
            logits = self._forward_bilstm(xs, lens)

        transition_score = self.crf.transition_score(y, lens)
        bilstm_score = self._bilstm_score(logits, y, lens)

        score = transition_score + bilstm_score

        return score

    def loglik(self, xs, y, lens):
        logits = self._forward_bilstm(xs, lens)
        forward_score = self.crf(logits, lens)
        gold_score = self.score(xs, y, lens, logits=logits)
        loglik = gold_score - forward_score

        return loglik, logits


def sequence_mask(lens, max_len=None):
    batch_size = lens.size(0)

    if max_len is None:
        max_len = lens.max().data[0]

    ranges = torch.arange(0, max_len).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    ranges = Variable(ranges)

    if lens.data.is_cuda:
        ranges = ranges.cuda()

    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp

    return mask