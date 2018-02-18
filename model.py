import torch
import torch.nn as nn
import torch.nn.init as I
import torch.nn.utils.rnn as R
from torch.autograd import Variable


def log_sum_exp(vec, dim=0):
    max, idx = torch.max(vec, dim)
    max_exp = max.unsqueeze(-1).expand_as(vec)
    return max + torch.log(torch.sum(torch.exp(vec - max_exp), dim))


class CRF(nn.Module):
    def __init__(self, vocab_size):
        super(CRF, self).__init__()

        self.vocab_size = vocab_size
        self.n_labels = n_labels = vocab_size + 2
        self.start_idx = n_labels - 2
        self.stop_idx = n_labels - 1
        self.transitions = nn.Parameter(torch.randn(n_labels, n_labels))

    def reset_parameters(self):
        I.normal(self.transitions.data, 0, 1)

    def forward(self, logits, lens):
        """
        Arguments:
            logits: [batch_size, seq_len, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        """
        batch_size, seq_len, n_labels = logits.size()
        alpha = logits.data.new(batch_size, self.n_labels).fill_(-10000)
        alpha[:, self.start_idx] = 0
        alpha = Variable(alpha)
        c_lens = lens.clone()

        logits_t = logits.transpose(1, 0)
        for logit in logits_t:
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   *self.transitions.size())
            alpha_exp = alpha.unsqueeze(1).expand(batch_size,
                                                  *self.transitions.size())
            trans_exp = self.transitions.unsqueeze(0).expand_as(alpha_exp)
            mat = trans_exp + alpha_exp + logit_exp
            alpha_nxt = log_sum_exp(mat, 2).squeeze(-1)

            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(alpha)
            alpha = mask * alpha_nxt + (1 - mask) * alpha
            c_lens = c_lens - 1

        alpha = alpha + self.transitions[self.stop_idx].unsqueeze(0).expand_as(alpha)
        norm = log_sum_exp(alpha, 1).squeeze(-1)

        return norm

    def viterbi_decode(self, logits, lens):
        """Borrowed from pytorch tutorial

        Arguments:
            logits: [batch_size, seq_len, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        """
        batch_size, seq_len, n_labels = logits.size()
        vit = logits.data.new(batch_size, self.n_labels).fill_(-10000)
        vit[:, self.start_idx] = 0
        vit = Variable(vit)
        c_lens = lens.clone()

        logits_t = logits.transpose(1, 0)
        pointers = []
        for logit in logits_t:
            vit_exp = vit.unsqueeze(1).expand(batch_size, n_labels, n_labels)
            trn_exp = self.transitions.unsqueeze(0).expand_as(vit_exp)
            vit_trn_sum = vit_exp + trn_exp
            vt_max, vt_argmax = vit_trn_sum.max(2)

            vt_max = vt_max.squeeze(-1)
            vit_nxt = vt_max + logit
            pointers.append(vt_argmax.squeeze(-1).unsqueeze(0))

            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(vit_nxt)
            vit = mask * vit_nxt + (1 - mask) * vit

            mask = (c_lens == 1).float().unsqueeze(-1).expand_as(vit_nxt)
            vit += mask * self.transitions[ self.stop_idx ].unsqueeze(0).expand_as(vit_nxt)

            c_lens = c_lens - 1

        pointers = torch.cat(pointers)
        scores, idx = vit.max(1)
        idx = idx.squeeze(-1)
        paths = [idx.unsqueeze(1)]

        for argmax in reversed(pointers):
            idx_exp = idx.unsqueeze(-1)
            idx = torch.gather(argmax, 1, idx_exp)
            idx = idx.squeeze(-1)

            paths.insert(0, idx.unsqueeze(1))

        paths = torch.cat(paths[1:], 1)
        scores = scores.squeeze(-1)

        return scores, paths

    def transition_score(self, labels, lens):
        """
        Arguments:
             labels: [batch_size, seq_len] LongTensor
             lens: [batch_size] LongTensor
        """
        batch_size, seq_len = labels.size()

        # pad labels with <start> and <stop> indices
        labels_ext = Variable(labels.data.new(batch_size, seq_len + 2))
        labels_ext[:, 0] = self.start_idx
        labels_ext[:, 1:-1] = labels
        mask = sequence_mask(lens + 1, max_len=seq_len + 2).long()
        pad_stop = Variable(labels.data.new(1).fill_(self.stop_idx))
        pad_stop = pad_stop.unsqueeze(-1).expand(batch_size, seq_len + 2)
        labels_ext = (1 - mask) * pad_stop + mask * labels_ext
        labels = labels_ext

        trn = self.transitions

        # obtain transition vector for each label in batch and timestep
        # (except the last ones)
        trn_exp = trn.unsqueeze(0).expand(batch_size, *trn.size())
        lbl_r = labels[:, 1:]
        lbl_rexp = lbl_r.unsqueeze(-1).expand(*lbl_r.size(), trn.size(0))
        trn_row = torch.gather(trn_exp, 1, lbl_rexp)

        # obtain transition score from the transition vector for each label
        # in batch and timestep (except the first ones)
        lbl_lexp = labels[:, :-1].unsqueeze(-1)
        trn_scr = torch.gather(trn_row, 2, lbl_lexp)
        trn_scr = trn_scr.squeeze(-1)

        mask = sequence_mask(lens + 1).float()
        trn_scr = trn_scr * mask
        score = trn_scr.sum(1).squeeze(-1)

        return score


class LSTMCRF(nn.Module):
    def __init__(self, crf, vocab_sizes, word_dims, hidden_dim, layers,
                 dropout_prob, bidirectional=False):
        super(LSTMCRF, self).__init__()

        self.n_feats = len(word_dims)
        self.total_word_dim = sum(word_dims)
        self.word_dims = word_dims
        self.hidden_dim = hidden_dim
        self.lstm_layers = layers
        self.dropout_prob = dropout_prob
        self.is_cuda = False

        self.crf = crf
        self.bidirectional = bidirectional
        self.n_labels = n_labels = self.crf.n_labels
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, word_dim)
             for vocab_size, word_dim in zip(vocab_sizes, word_dims)]
        )

        self.output_hidden_dim = self.hidden_dim
        if bidirectional:
            self.output_hidden_dim *= 2

        self.tanh = nn.Tanh()
        self.input_layer = nn.Linear(self.total_word_dim, hidden_dim)
        self.output_layer = nn.Linear(self.output_hidden_dim, n_labels)
        self.lstm = nn.LSTM(input_size=hidden_dim,
                            hidden_size=hidden_dim,
                            num_layers=layers,
                            bidirectional=bidirectional,
                            dropout=dropout_prob,
                            batch_first=True)

    def reset_parameters(self):
        for emb in self.embeddings:
            I.xavier_normal(emb.weight.data)

        I.xavier_normal(self.input_layer.weight.data)
        I.xavier_normal(self.output_layer.weight.data)
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

        res = [emb(x) for emb, x in zip(self.embeddings, xs)]
        x = torch.cat(res, 2)

        return x

    def _forward_bilstm(self, xs, lens):
        n_feats, batch_size, seq_len = xs.size()

        x = self._embeddings(xs)
        x = x.view(-1, self.total_word_dim)
        x = self.tanh(self.input_layer(x))
        x = x.view(batch_size, seq_len, self.hidden_dim)

        o, h = self._run_rnn_packed(self.lstm, x, lens)

        o = o.contiguous()
        o = o.view(-1, self.output_hidden_dim)
        o = self.tanh(self.output_layer(o))
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

    def predict(self, xs, lens, return_scores=False):
        logits = self._forward_bilstm(xs, lens)
        scores, preds = self.crf.viterbi_decode(logits, lens)

        if return_scores:
            return preds, scores
        else:
            return preds

    def loglik(self, xs, y, lens, return_logits=False):
        logits = self._forward_bilstm(xs, lens)
        norm_score = self.crf(logits, lens)
        sequence_score = self.score(xs, y, lens, logits=logits)
        loglik = sequence_score - norm_score

        if return_logits:
            return loglik, logits
        else:
            return loglik


class TransparentDataParallel(nn.DataParallel):
    def __init__(self, *args, **kwargs):
        super(TransparentDataParallel, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        try:
            return super(TransparentDataParallel, self).__getattr__(item)
        except AttributeError:
            module = self.__dict__["_modules"]["module"]
            return module.__getattribute__(item)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)


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