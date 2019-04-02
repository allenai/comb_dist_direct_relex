from typing import Dict

import logging
from overrides import overrides
import torch
from torch import nn
import numpy as np
from allennlp.data import Vocabulary
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.training.metrics.average import Average
from allennlp.modules import TextFieldEmbedder

from relex.multilabel_average_precision_metric import MultilabelAveragePrecision
from relex.relation_instances_reader import RelationInstancesReader

log = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("comb_direct_dist_relex")
class CombDirectDistRelex(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 cnn_size: int = 100,
                 dropout_weight: float = 0.1,
                 with_entity_embeddings: bool = True ,
                 sent_loss_weight: float = 1,
                 attention_weight_fn: str = 'sigmoid',
                 attention_aggregation_fn: str = 'max') -> None:
        regularizer = None
        super().__init__(vocab, regularizer)
        self.output_alphas = False  # set to True at decoding
        self.num_classes = self.vocab.get_vocab_size("labels")

        self.text_field_embedder = text_field_embedder
        self.dropout_weight = dropout_weight
        self.with_entity_embeddings = with_entity_embeddings
        self.sent_loss_weight = sent_loss_weight
        self.attention_weight_fn = attention_weight_fn
        self.attention_aggregation_fn = attention_aggregation_fn

        # instantiate position embedder
        pos_embed_output_size = 5
        pos_embed_input_size = 2 * RelationInstancesReader.max_distance + 1
        self.pos_embed = nn.Embedding(pos_embed_input_size, pos_embed_output_size)
        pos_embed_weights = np.array([range(pos_embed_input_size)] * pos_embed_output_size).T
        self.pos_embed.weight = nn.Parameter(torch.Tensor(pos_embed_weights))

        d = cnn_size
        sent_encoder = CnnEncoder  # TODO: should be moved to the config file
        cnn_output_size = d
        embedding_size = 300  # TODO: should be moved to the config file

        # instantiate sentence encoder 
        self.cnn = sent_encoder(embedding_dim=(embedding_size + 2 * pos_embed_output_size), num_filters=cnn_size,
                                ngram_filter_sizes=(2, 3, 4, 5),
                                conv_layer_activation=torch.nn.ReLU(), output_dim=cnn_output_size)

        # dropout after word embedding
        self.dropout = nn.Dropout(p=self.dropout_weight)

        #  given a sentence, returns its unnormalized attention weight
        self.attention_ff = nn.Sequential(
                nn.Linear(cnn_output_size, d),
                nn.ReLU(),
                nn.Linear(d, 1)
        )

        self.ff_before_alpha = nn.Sequential(
                nn.Linear(1, 50),
                nn.ReLU(),
                nn.Linear(50, 1),
        )

        ff_input_size = cnn_output_size
        if self.with_entity_embeddings:
            ff_input_size += embedding_size

        # output layer
        self.ff = nn.Sequential(
                nn.Linear(ff_input_size, d),
                nn.ReLU(),
                nn.Linear(d, self.num_classes)
        )

        self.loss = torch.nn.BCEWithLogitsLoss()  # sigmoid + binary cross entropy
        self.metrics = {}
        self.metrics['ap'] = MultilabelAveragePrecision()  # average precision = AUC
        self.metrics['bag_loss'] = Average()  # to display bag-level loss

        if self.sent_loss_weight > 0:
            self.metrics['sent_loss'] = Average()  # to display sentence-level loss

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                mentions: Dict[str, torch.LongTensor],
                positions1: torch.LongTensor,
                positions2: torch.LongTensor,
                is_direct_supervision_bag: torch.LongTensor,
                sent_labels: torch.LongTensor,  # sentence-level labels 
                labels: torch.LongTensor  # bag-level labels
                ) -> Dict[str, torch.Tensor]:

        # is all instances in this batch directly or distantly supervised
        is_direct_supervision_batch = bool(is_direct_supervision_bag['tokens'].shape[1] - 1)

        if is_direct_supervision_bag['tokens'].shape[1] != 1:
            direct_supervision_bags_count = sum(is_direct_supervision_bag['tokens'][:, -1] != 0).item()
            # is it a mix of both ? this affects a single batch because of the sorting_keys in the bucket iterator 
            if direct_supervision_bags_count != len(is_direct_supervision_bag['tokens'][:, -1] != 0):
                log.error("Mixed batch with %d supervised bags. Treated as dist. supervised", direct_supervision_bags_count)

        tokens = mentions['tokens']
        assert tokens.dim() == 3
        batch_size = tokens.size(0)
        padded_bag_size = tokens.size(1)
        padded_sent_size = tokens.size(2)
        mask = util.get_text_field_mask(mentions, num_wrapping_dims=1)

        # embed text
        t_embd = self.text_field_embedder(mentions)

        # embed position information
        p1_embd = self.pos_embed(positions1)
        p2_embd = self.pos_embed(positions2)

        # concatinate position emebddings to the word embeddings
        # x.shape: batch_size x padded_bag_size x padded_sent_size x (text_embedding_size + 2 * position_embedding_size)
        x = torch.cat([t_embd, p1_embd, p2_embd], dim=3)

        if self.dropout_weight > 0:
            x = self.dropout(x)

        # merge the first two dimensions becase sentence encoder doesn't support the 4d input
        x = x.view(batch_size * padded_bag_size, padded_sent_size, -1) 
        mask = mask.view(batch_size * padded_bag_size, -1)

        # call sequence encoder
        x = self.cnn(x, mask)  # (batch_size * padded_bag_size) x cnn_output_size

        # separate the first two dimensions back
        x = x.view(batch_size, padded_bag_size, -1)
        mask = mask.view(batch_size, padded_bag_size, -1)
       
        # compute unnormalized attention weights, one scaler per sentence
        alphas = self.attention_ff(x)

        if self.sent_loss_weight > 0:
            # compute sentence-level loss on the directly supervised data (if any)
            sent_labels = sent_labels.unsqueeze(-1)
            # `sent_labels != 2`: directly supervised examples and distantly supervised negative examples
            sent_labels_mask = ((sent_labels != 2).long() * mask[:, :, [0]]).float()
            sent_labels_masked_pred = sent_labels_mask * torch.sigmoid(alphas)
            sent_labels_masked_goal = sent_labels_mask * sent_labels.float()
            sent_loss = torch.nn.functional.binary_cross_entropy(sent_labels_masked_pred, sent_labels_masked_goal)

        # apply a small FF to the attention weights
        alphas = self.ff_before_alpha(alphas)

        # normalize attention weights based on the selected weighting function 
        if self.attention_weight_fn == 'uniform':
            alphas = mask[:, :, 0].float()
        elif self.attention_weight_fn == 'softmax':
            alphas = util.masked_softmax(alphas.squeeze(-1), mask[:, :, 0].float())
        elif self.attention_weight_fn == 'sigmoid':
            alphas = torch.sigmoid(alphas.squeeze(-1)) * mask[:, :, 0].float()
        elif self.attention_weight_fn == 'norm_sigmoid':  # equation 7 in https://arxiv.org/pdf/1805.02214.pdf
            alphas = torch.sigmoid(alphas.squeeze(-1)) * mask[:, :, 0].float()
            alphas = alphas / alphas.sum(dim=-1, keepdim=True)
        else:
            assert False

        # Input: 
        #   `x`: sentence encodings
        #   `alphas`: attention weights
        #   `attention_aggregation_fn`: aggregation function
        # Output: bag encoding
        if self.attention_aggregation_fn == 'max':
            x = alphas.unsqueeze(-1) * x  # weight sentences
            x = x.max(dim=1)[0]  # max pooling
        elif self.attention_aggregation_fn == 'avg':
            x = torch.bmm(alphas.unsqueeze(1), x).squeeze(1)  # average pooling
        else:
            assert False

        if self.with_entity_embeddings:
            # actual bag_size (not padded_bag_size) for each instance in the batch
            bag_size = mask[:, :, 0].sum(dim=1, keepdim=True).float()

            e1_mask = (positions1 == 0).long() * mask
            e1_embd = torch.matmul(e1_mask.unsqueeze(2).float(), t_embd)
            e1_embd_sent_sum = e1_embd.squeeze(dim=2).sum(dim=1)
            e1_embd_sent_avg = e1_embd_sent_sum / bag_size

            e2_mask = (positions2 == 0).long() * mask
            e2_embd = torch.matmul(e2_mask.unsqueeze(2).float(), t_embd)
            e2_embd_sent_sum = e2_embd.squeeze(dim=2).sum(dim=1)
            e2_embd_sent_avg = e2_embd_sent_sum / bag_size

            e1_e2_mult = e1_embd_sent_avg * e2_embd_sent_avg
            x = torch.cat([x, e1_e2_mult], dim=1)

        logits = self.ff(x)  # batch_size x self.num_classes
        output_dict = {'logits': logits}  # sigmoid is applied in the loss function and the metric class, not here

        if self.output_alphas:  # for prediction
            if alphas is not None:
                output_dict['alphas'] = alphas.data.cpu().numpy().tolist()

        if labels is not None:  # Training and evaluation
            w = self.sent_loss_weight / (self.sent_loss_weight + 1)
            one_minus_w = 1 - w  # weight of the bag-level loss

            if is_direct_supervision_batch and self.sent_loss_weight > 0:
                one_minus_w = 0

            loss = self.loss(logits, labels.squeeze(-1).type_as(logits)) * self.num_classes  # scale the loss to be more readable
            loss *= one_minus_w
            self.metrics['bag_loss'](loss.item())
            self.metrics['ap'](logits, labels.squeeze(-1))

            if self.sent_loss_weight > 0:
                sent_loss *= w
                self.metrics['sent_loss'](sent_loss.item())
                loss += sent_loss
            output_dict['loss'] = loss
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        prob_thr = 0.6  # to ignore predicted labels with low prob. 
        self.output_alphas = True  # TODO: this is not working for the first call

        probs = torch.sigmoid(output_dict['logits'])
        labels_count = 0
        for i, p in enumerate(probs.squeeze().cpu().data.numpy()):
            if p > prob_thr:  # ignore predictions with low prob.
                output_dict[self.vocab.get_token_from_index(i, namespace="labels")] = torch.Tensor([float(p)])
                labels_count += 1
        output_dict['labels_count'] = torch.Tensor([labels_count])
        del output_dict['logits']
        del output_dict['loss']
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
