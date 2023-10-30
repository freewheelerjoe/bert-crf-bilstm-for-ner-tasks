from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import  TokenClassifierOutput
from torch import nn
from torch.nn import CrossEntropyLoss
import torch
from torchcrf import CRF

max_number_of_pos_tags = 22

class BertEmbeddings(nn.Module):
    def __init__(self, config,ner_ratio):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
     
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
       
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
      
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 定义drop
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
   
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        self.ner_ratio = ner_ratio

    def forward(
            self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None,
            past_key_values_length=0):

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

       
        seq_length = input_shape[1]

        # 位置向量映射
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

 
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

  
        embeddings = inputs_embeds + token_type_embeddings
    
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

    
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertForTokenClassification(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
      
        self.bert = BertModel(config, add_pooling_layer=False)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
       
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

       
        sequence_output = outputs[0]

       
        sequence_output = self.dropout(sequence_output)


        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertCRF(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
      
        self.num_labels = config.num_labels

     
        self.bert = BertModel(config,add_pooling_layer=False)
  
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
        self.bilstm = nn.LSTM(config.hidden_size, (config.hidden_size) // 2, batch_first=True,
                              bidirectional=True)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        lstm_output, hc = self.bilstm(sequence_output)
        logits = self.classifier(lstm_output)


        loss = None
        if labels is not None:
            log_likelihood, tags = self.crf(logits, labels), self.crf.decode(logits)
            loss = 0 - log_likelihood
        else:
            tags = self.crf.decode(logits)
        tags = torch.Tensor(tags)

    
        if not return_dict:
            output = (tags,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return loss, tags
