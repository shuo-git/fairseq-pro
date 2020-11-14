# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq import calibration_utils
from fairseq.data import data_utils


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task, sentence_avg, label_smoothing):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.alpha = args.alpha
        self.beta = args.beta
        self.neg_lambda = args.neg_lambda
        self.eps_pos = args.eps_pos
        self.eps_neg = args.eps_neg
        self.gram1 = args.gram1
        self.gram2 = args.gram2
        self.gram3 = args.gram3
        self.gram4 = args.gram4
        self.pad_idx = task.tgt_dict.pad()
        self.eos_idx = task.tgt_dict.eos()

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--alpha', default=1.0, type=float,
                            help='loss = alpha x org_loss + beta x cali_loss')
        parser.add_argument('--beta', default=1.0, type=float,
                            help='loss = alpha x org_loss + beta x cali_loss')
        parser.add_argument('--neg-lambda', default=-1.0, type=float,
                            help='cali_loss = pos_loss + neg_lambda x neg_loss')
        parser.add_argument('--eps-pos', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing of correct events, 0 means no label smoothing')
        parser.add_argument('--eps-neg', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing of wrong events, 1 means entropy')
        parser.add_argument('--gram1', action='store_true',
                            help='include 1gram event')
        parser.add_argument('--gram2', action='store_true',
                            help='include 2gram event')
        parser.add_argument('--gram3', action='store_true',
                            help='include 3gram event')
        parser.add_argument('--gram4', action='store_true',
                            help='include 4gram event')
        # fmt: on

    def forward(self, model, sample, reduce=True, generator=None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if generator is None:
            net_output = model(**sample['net_input'])
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
            logging_output = {
                'loss': loss.data,
                'nll_loss': nll_loss.data,
                'ntokens': sample['ntokens'],
                'nsentences': sample['target'].size(0),
                'sample_size': sample_size,
            }
        else:
            net_output = model(**sample['net_input'])
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

            prev_hyps, hyps, labels = self.generate(model, sample, generator)
            model.train()
            net_output_hyp = model(src_tokens=sample['net_input']['src_tokens'],
                                   src_lengths=sample['net_input']['src_lengths'],
                                   prev_output_tokens=prev_hyps)
            lprobs_hyp = model.get_normalized_probs(net_output_hyp, log_probs=True)
            cali_loss = 0.
            for n in labels.keys():
                temp_loss, _ = self.compute_cali_loss(lprobs_hyp, hyps, labels[n], n, reduce=reduce)
                cali_loss += temp_loss

            loss = self.alpha * loss + self.beta * cali_loss

            sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
            logging_output = {
                'loss': loss.data,
                'nll_loss': nll_loss.data,
                'cali_loss': cali_loss if type(cali_loss) is float else cali_loss.data,
                'ntokens': sample['ntokens'],
                'nsentences': sample['target'].size(0),
                'sample_size': sample_size,
            }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    def generate(self, model, sample, generator):
        with torch.no_grad():
            predictions = generator.generate([model], sample)
            hyps = []
            labels = {}
            if self.gram1:
                labels[1] = []
            if self.gram2:
                labels[2] = []
            if self.gram3:
                labels[3] = []
            if self.gram4:
                labels[4] = []
            device = sample['id'].device
            for i, sample_id in enumerate(sample['id'].tolist()):
                hyps.append(utils.strip_pad(predictions[i][0]['tokens'], self.pad_idx))
                hyp = hyps[-1].int().tolist()
                ref = utils.strip_pad(sample['target'][i], self.pad_idx).int().tolist()
                for n in labels.keys():
                    labels[n].append(torch.tensor(calibration_utils.label_n_gram(ref, hyp, n),
                                                  dtype=torch.int, device=device))
            prev_hyps = data_utils.collate_tokens(hyps, pad_idx=self.pad_idx, eos_idx=self.eos_idx, left_pad=False, move_eos_to_beginning=True)
            hyps = data_utils.collate_tokens(hyps, pad_idx=self.pad_idx, eos_idx=self.eos_idx, left_pad=False, move_eos_to_beginning=False)
            for n in labels.keys():
                labels[n] = data_utils.collate_tokens(labels[n], pad_idx=0)
        return prev_hyps, hyps, labels

    def compute_cali_loss(self, input_lprobs, origin_target, origin_label=None, ngram=1, reduce=True):
        nll_loss = -input_lprobs.gather(dim=-1, index=origin_target.unsqueeze(dim=-1)).squeeze(dim=-1)      # BxL
        smooth_loss = -input_lprobs.sum(dim=-1)     # BxL
        ngram_nll_loss = torch.zeros_like(nll_loss) + nll_loss
        ngram_smooth_loss = torch.zeros_like(smooth_loss) + smooth_loss
        for i in range(1, ngram):
            ngram_nll_loss[:, i:] += nll_loss[:, :-i]
            ngram_smooth_loss[:, i:] += smooth_loss[:, :-i]
        ngram_nll_loss = ngram_nll_loss[:, ngram - 1:]
        ngram_smooth_loss = ngram_smooth_loss[:, ngram - 1:]
        target = origin_target[:, ngram - 1:]
        non_pad_mask = target.ne(self.padding_idx)
        ngram_nll_loss = ngram_nll_loss[non_pad_mask]
        ngram_smooth_loss = ngram_smooth_loss[non_pad_mask]
        if origin_label is not None:
            assert origin_label.size() == target.size()
            label = origin_label[non_pad_mask].to(dtype=ngram_nll_loss.dtype)
        else:
            label = 1.
        pos_ngram_nll_loss = label * ngram_nll_loss
        neg_ngram_nll_loss = (1. - label) * ngram_nll_loss
        pos_ngram_smooth_loss = label * ngram_smooth_loss
        neg_ngram_smooth_loss = (1. - label) * ngram_smooth_loss
        if reduce:
            pos_ngram_nll_loss = pos_ngram_nll_loss.sum()
            neg_ngram_nll_loss = neg_ngram_nll_loss.sum()
            pos_ngram_smooth_loss = pos_ngram_smooth_loss.sum()
            neg_ngram_smooth_loss = neg_ngram_smooth_loss.sum()
        vocab_size = input_lprobs.size(-1)
        pos_loss = (1. - self.eps_pos) * pos_ngram_nll_loss + self.eps_pos / vocab_size * pos_ngram_smooth_loss
        neg_loss = (1. - self.eps_neg) * neg_ngram_nll_loss + self.eps_neg / vocab_size * neg_ngram_smooth_loss
        loss = pos_loss + self.neg_lambda * neg_loss
        nll_loss = pos_ngram_nll_loss + self.neg_lambda * neg_ngram_nll_loss
        return loss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        cali_loss_sum = sum(log.get('cali_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('cali_loss', cali_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
