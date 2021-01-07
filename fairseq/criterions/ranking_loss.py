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
import numpy as np


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


@register_criterion('ranking_loss')
class RankingLossCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, alpha, ranking_lambda, ce_lambda):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.alpha = alpha
        self.ranking_lambda = ranking_lambda
        self.ce_lambda = ce_lambda
        self.pad_idx = task.tgt_dict.pad()
        self.eos_idx = task.tgt_dict.eos()

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--ranking-lambda', default=0.0, type=float,
                            help='total_loss = ce_lambda x cross_entropy + ranking_lambda x ranking_loss')
        parser.add_argument('--ce-lambda', default=1.0, type=float,
                            help='total_loss = ce_lambda x cross_entropy + ranking_lambda x ranking_loss')
        parser.add_argument('--alpha', default=1.0, type=float,
                            help='ranking_loss = alpha x (log s1 - log s2) - (log p1 - log p2)')
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

            prev_hyps, hyps, scores, lengths = self.generate(model, sample, generator)
            model.train()
            # `(batch, tgt_len, vocab)`
            net_output_hyp1 = model(src_tokens=sample['net_input']['src_tokens'],
                                   src_lengths=sample['net_input']['src_lengths'],
                                   prev_output_tokens=prev_hyps[0])
            net_output_hyp2 = model(src_tokens=sample['net_input']['src_tokens'],
                                    src_lengths=sample['net_input']['src_lengths'],
                                    prev_output_tokens=prev_hyps[1])
            lprobs_hyp1 = model.get_normalized_probs(net_output_hyp1, log_probs=True).gather(
                dim=-1, index=hyps[0].unsqueeze(dim=-1)
            ).squeeze(dim=-1) * hyps[0].ne(self.pad_idx)
            lprobs_hyp2 = model.get_normalized_probs(net_output_hyp2, log_probs=True).gather(
                dim=-1, index=hyps[1].unsqueeze(dim=-1)
            ).squeeze(dim=-1) * hyps[1].ne(self.pad_idx)

            lprobs_hyp1 = lprobs_hyp1.sum(dim=1) / (lengths[:, 0] + 1e-4)
            lprobs_hyp2 = lprobs_hyp2.sum(dim=1) / (lengths[:, 1] + 1e-4)

            cali_loss = self.alpha * (torch.log(scores[0]) - torch.log(scores[1])) - (lprobs_hyp1 - lprobs_hyp2)
            cali_loss = torch.max(torch.zeros_like(cali_loss), cali_loss) * lengths.mean(dim=1)
            if reduce:
                cali_loss = cali_loss.sum()

            loss = self.ce_lambda * loss + self.ranking_lambda * cali_loss

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
            hyps1 = []
            hyps2 = []
            scores1 = []
            scores2 = []
            lengths1 = []
            lengths2 = []
            list_size = generator.beam_size
            device = sample['id'].device
            for i, sample_id in enumerate(sample['id'].tolist()):
                idx1 = np.random.randint(list_size)
                while True:
                    idx2 = np.random.randint(list_size)
                    if idx2 != idx1:
                        break
                hyp1_tensor = utils.strip_pad(predictions[i][idx1]['tokens'], self.pad_idx)
                hyp2_tensor = utils.strip_pad(predictions[i][idx2]['tokens'], self.pad_idx)
                hyp1 = hyp1_tensor.int().tolist()
                hyp2 = hyp2_tensor.int().tolist()
                ref = utils.strip_pad(sample['target'][i], self.pad_idx).int().tolist()
                score1 = calibration_utils.sentence_bleu(ref, hyp1)
                score2 = calibration_utils.sentence_bleu(ref, hyp2)
                if score1 > score2:
                    hyps1.append(hyp1_tensor)
                    hyps2.append(hyp2_tensor)
                    scores1.append(score1)
                    scores2.append(score2)
                    lengths1.append(len(hyp1))
                    lengths2.append(len(hyp2))
                else:
                    hyps1.append(hyp2_tensor)
                    hyps2.append(hyp1_tensor)
                    scores1.append(score2)
                    scores2.append(score1)
                    lengths1.append(len(hyp2))
                    lengths2.append(len(hyp1))
            prev_hyps1 = data_utils.collate_tokens(hyps1, pad_idx=self.pad_idx, eos_idx=self.eos_idx, left_pad=False,
                                                   move_eos_to_beginning=True)
            prev_hyps2 = data_utils.collate_tokens(hyps2, pad_idx=self.pad_idx, eos_idx=self.eos_idx, left_pad=False,
                                                   move_eos_to_beginning=True)
            hyps1 = data_utils.collate_tokens(hyps1, pad_idx=self.pad_idx, eos_idx=self.eos_idx, left_pad=False,
                                             move_eos_to_beginning=False)
            hyps2 = data_utils.collate_tokens(hyps2, pad_idx=self.pad_idx, eos_idx=self.eos_idx, left_pad=False,
                                              move_eos_to_beginning=False)
            scores1 = torch.tensor(scores1, dtype=torch.float, device=device) + 1e-4
            scores2 = torch.tensor(scores2, dtype=torch.float, device=device) + 1e-4
        lengths = torch.tensor(list(zip(lengths1, lengths2)), dtype=torch.float, device=device)
        return (prev_hyps1, prev_hyps2), (hyps1, hyps2), (scores1, scores2), lengths

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
