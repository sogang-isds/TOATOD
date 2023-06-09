import torch
from torch import nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config
import os

class T5Gen_Model(nn.Module):
    def __init__(self, model_path, tokenizer, dropout=0.1, is_training=True):
        super().__init__()
        self.tokenizer = tokenizer # tokenizer with extended vocabulary
        self.pad_token_id, self.sos_d_token_id, self.eos_d_token_id = self.tokenizer.convert_tokens_to_ids(['<_PAD_>', '<sos_d>', '<eos_d>'])

        if is_training:
            print ('Initializing Huggingface T5 model...')
            t5_config = T5Config.from_pretrained(model_path)
            t5_config.__dict__["dropout"] = dropout
            self.model = T5ForConditionalGeneration.from_pretrained(model_path, config=t5_config, resume_download=True)
        else:
            print('Loading Model from pretrained ckpt...')
            self.model = torch.load(os.path.join(model_path, "model.pt"))
        print ('Resizing Token Embeddings...')
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.tgt_sos_token_id = self.tokenizer.convert_tokens_to_ids(['<sos_d>'])[0]
        self.tgt_eos_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_d>'])[0]

    def forward(self, src_input, src_mask, tgt_input, tgt_output):
        src_mask = src_mask.type(src_input.type())
        outputs = self.model(input_ids=src_input, attention_mask=src_mask, decoder_input_ids=tgt_input, labels=tgt_output)
        loss = outputs[0]#.mean()
        return loss

    def parse_batch_text(self, batch_pred_ids):
        res_text_list = []
        for predicted_ids in batch_pred_ids:
            one_pred_ids = []
            for one_id in predicted_ids:
                if one_id in [self.pad_token_id, self.sos_d_token_id, self.eos_d_token_id]:
                    pass
                else:
                    one_pred_ids.append(one_id)
            one_res_text = self.tokenizer.decode(one_pred_ids)
            res_text_list.append(one_res_text)
        return res_text_list

    def batch_prediction(self, src_input, src_mask):
        #outputs = self.model.generate(input_ids = src_input, attention_mask = src_mask, decoder_start_token_id = self.sos_b_token_id,
        #    pad_token_id = self.pad_token_id, eos_token_id = self.eos_b_token_id, max_length = 64)
        outputs = self.model.generate(input_ids = src_input, attention_mask = src_mask, decoder_start_token_id = self.tgt_sos_token_id,
            pad_token_id = self.pad_token_id, eos_token_id = self.tgt_eos_token_id, max_length = 64)
        return self.parse_batch_text(outputs)

    def save_model(self, ckpt_save_path):
        if not os.path.exists(ckpt_save_path):
            os.mkdir(ckpt_save_path)
        # save model
        torch.save(self.model, os.path.join(ckpt_save_path, 'model.pt'))
        # save tokenizer
        self.tokenizer.save_pretrained(ckpt_save_path)