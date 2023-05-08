import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn.functional as F
from torch import nn
from transformers import T5Tokenizer
from t5adapter import set_task_for_inference, set_task_for_train

class T5ForReinforce(nn.Module):
    def __init__(self, model_path, evaluator, special_token_list, alpha=0.7, beta=0.5):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = torch.load(os.path.join(model_path, 'model.pt'),map_location='cpu')
        self.evaluator = evaluator
        self.special_token_list = special_token_list
        self.add_special_decoder_token = True
        self.pad_token_id, self.sos_b_token_id, self.eos_b_token_id, self.sos_a_token_id, self.eos_a_token_id, \
        self.sos_r_token_id, self.eos_r_token_id = self.tokenizer.convert_tokens_to_ids(['<_PAD_>', '<sos_b>',
                                                                                         '<eos_b>', '<sos_a>',
                                                                                         '<eos_a>', '<sos_r>',
                                                                                         '<eos_r>'])
        self.alpha = alpha
        self.beta = beta
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.rewards = []

    def forward(self, batch, mode, dial_id=None,dials=None,ver='2.1'):
        loss = 0
        beta = self.beta
        if mode == 'nlg':
            start_token, end_token, start_token_id, end_token_id = '<sos_r>', '<eos_r>', self.sos_r_token_id, self.eos_r_token_id

            pack = []
            need_key = ["bspn","dspn","pointer"]

            src_input, src_mask, tgt_input, tgt_output = batch
            outputs = self.model(input_ids=src_input, attention_mask=src_mask, labels=tgt_output)
            session_loss, logits = outputs.loss, outputs.logits
            prob = F.softmax(logits, dim=-1)
            loss += session_loss.mean()
            batch_size = src_input.size(0)
            loss_tensor = torch.zeros(batch_size).to(src_input.device)
            for i in range(batch_size):
                prediction = self.tokenized_decode(prob[i, :, :].argmax(dim=-1). \
                                                   tolist()).strip()
                prediction = prediction.split(start_token)[-1].split(end_token)[0].strip()
                preds = []
                for token in prediction.split():
                    if token == '<_PAD_>':
                        continue
                    else:
                        preds.append(token)
                prediction = ' '.join(preds).strip()

                golden = tgt_output[i, :].tolist()
                golden = golden[:golden.index(-100) if -100 in golden else len(golden)]
                gt = self.tokenized_decode(golden).strip()
                gt = gt.split(start_token)[-1].split(end_token)[0].strip()

                gs = []
                for token in gt.split():
                    if token == '<_PAD_>':
                        continue
                    else:
                        gs.append(token)
                gt = ' '.join(gs)


                dic = {}
                for key in need_key:
                    if not isinstance(dials[i][key],str):
                        v = self.tokenized_decode(dials[i][key])
                    else:
                        v = dials[i][key]
                    if key in ["bspn"]:
                        dic[f"{key}_gen"] = v
                    else:
                        dic[key] = v
                dic.update({'dial_id': dial_id[i], 'turn_num': i, 'resp': gt, 'resp_gen': prediction})
                pack.append(dic)

                p = prob[i, :, :].max(dim=-1).values.prod() + 1e-10
                log_prob = torch.log(p)
                loss_tensor[i] = log_prob


                bleu, success, match = self.evaluator.validation_metric(pack)
            # else:
            #     results = self.evaluator.e.evaluate(pack)
            #     match, success, bleu = results['success']['inform']['total'], results['success']['success']['total'], \
            #                            results['bleu']['mwz22']
            # print(prediction)
            # print(bleu)
            combined_score = 0.5 * (success + match) + bleu
            reward = beta * success + (1 - beta) * bleu + 1 # 1 is for avoiding zero reward
            loss_tensor = -(loss_tensor * reward / 100) # 100 is for normalization for balancing with categorical cross entropy loss
            loss_tensor = loss_tensor.mean()
            policy_loss = loss_tensor

            loss = self.alpha * policy_loss + (1 - self.alpha) * loss

            return loss, \
                   torch.Tensor([reward]).to(loss.device), \
                   torch.Tensor([match]).to(loss.device), \
                   torch.Tensor([success]).to(loss.device), \
                   torch.Tensor([bleu]).to(loss.device), \
                   torch.Tensor([combined_score]).to(loss.device)

        elif mode == 'dst':
            start_token, end_token, start_token_id, end_token_id = '<sos_b>', '<eos_b>', self.sos_b_token_id, self.eos_b_token_id
            src_input, src_mask, tgt_input, tgt_output = batch
            outputs = self.model(input_ids=src_input, attention_mask=src_mask, decoder_input_ids=tgt_input,
                                 labels=tgt_output)
            loss, logits = outputs.loss, outputs.logits
            prob = F.softmax(logits, dim=-1)

            batch_size = src_input.size(0)

            loss_tensor = torch.zeros(batch_size).to(loss.device)
            reward_tensor = torch.zeros(batch_size).to(loss.device)
            for i in range(batch_size):
                prediction = self.tokenized_decode(prob[i, :, :].argmax(dim=-1).tolist()).strip()
                prediction = prediction.split(start_token)[-1].split(end_token)[0].strip()

                preds = []
                for token in prediction.split():
                    if token == '<_PAD_>':
                        continue
                    else:
                        preds.append(token)
                prediction = ' '.join(preds)


                golden = tgt_output[i, :].tolist()
                golden = golden[:golden.index(-100) if -100 in golden else len(golden)]
                gt = self.tokenized_decode(golden).strip()
                gt = gt.split(start_token)[-1].split(end_token)[0].strip()

                gs = []
                for token in gt.split():
                    if token == '<_PAD_>':
                        continue
                    else:
                        gs.append(token)
                gt = ' '.join(gs)

                if "<eos_b>" in prediction:
                    prediction = prediction[:prediction.index("<eos_b>")]
                if "<eos_b>" in gt:
                    gt = gt[:gt.index("<eos_b>")]

                pack = [{"dial_id": "0", "turn_num": 0, "bspn_gen": "", "bspn": ""}
                    , {"dial_id": "0", "turn_num": str(i + 1), "bspn_gen": prediction, "bspn": gt}]
                rew, f1, acc, _, _ = self.evaluator.dialog_state_tracking_eval(pack, eval_dial_list=["0.json"])
                reward = rew + 1  # add 1 to avoid zero reward
                p = prob[i, :, :].max(dim=-1).values.prod() + 1e-10

                log_prob = torch.log(p)

                policy_loss = - (log_prob * reward)
                loss_tensor[i] = policy_loss
                reward_tensor[i] = rew

            r = reward_tensor.mean()
            loss_tensor = loss_tensor.mean()
            loss = self.alpha * loss_tensor + (1 - self.alpha) * loss
            return loss, r

    def tokenized_decode(self, token_id_list):
        pred_tokens = self.tokenizer.convert_ids_to_tokens(token_id_list)
        res_text = ''
        curr_list = []
        for token in pred_tokens:
            if token in self.special_token_list + ['<s>', '</s>', '<pad>']:
                if len(curr_list) == 0:
                    res_text += ' ' + token + ' '
                else:
                    curr_res = self.tokenizer.convert_tokens_to_string(curr_list)
                    res_text = res_text + ' ' + curr_res + ' ' + token + ' '
                    curr_list = []
            else:
                curr_list.append(token)
        if len(curr_list) > 0:
            curr_res = self.tokenizer.convert_tokens_to_string(curr_list)
            res_text = res_text + ' ' + curr_res + ' '
        res_text_list = res_text.strip().split()
        res_text = ' '.join(res_text_list).strip()
        return res_text

    def batch_generate(self, src_input, src_mask, generate_mode, max_decode_len):
        '''
            This function deals with batch generation. In order to fully take advantage of batch inference,
            in each batch, we only generate one type of output. e.g. Given a batch of dialogue history, we
            generate the corresponding belief state/dialogue action/system response for the given batch. The
            specific type of output is decided by the input argument "generate_mode"
        '''
        if self.add_special_decoder_token:
            if generate_mode == 'bs':
                start_token, end_token, start_token_id, end_token_id = '<sos_b>', '<eos_b>', self.sos_b_token_id, self.eos_b_token_id
            elif generate_mode == 'da':
                start_token, end_token, start_token_id, end_token_id = '<sos_a>', '<eos_a>', self.sos_a_token_id, self.eos_a_token_id
            elif generate_mode == 'nlg':
                start_token, end_token, start_token_id, end_token_id = '<sos_r>', '<eos_r>', self.sos_r_token_id, self.eos_r_token_id
            else:
                raise Exception('Wrong Generate Mode!!!')
        else:
            start_token, end_token = '<pad>', '</s>'
            start_token_id, end_token_id = \
                self.tokenizer.convert_tokens_to_ids([start_token])[0], \
                self.tokenizer.convert_tokens_to_ids([end_token])[0]

        outputs = self.model.generate(input_ids=src_input, attention_mask=src_mask,
                                      decoder_start_token_id=start_token_id,
                                      pad_token_id=self.pad_token_id, eos_token_id=end_token_id,
                                      max_length=max_decode_len)

        res_text_list = []
        for predicted_ids in outputs:
            one_res_text = self.tokenized_decode(predicted_ids)
            # print (one_res_text)
            one_res_text = one_res_text.split(start_token)[-1].split(end_token)[0].strip()

            final_res_list = []
            for token in one_res_text.split():
                if token == '<_PAD_>':
                    continue
                else:
                    final_res_list.append(token)
            one_res_text = ' '.join(final_res_list).strip()

            res_text_list.append(one_res_text)
        return res_text_list

    def save_model(self, ckpt_save_path):
        if not os.path.exists(ckpt_save_path):
            os.mkdir(ckpt_save_path)
        # save model
        torch.save(self.model, os.path.join(ckpt_save_path, 'model.pt'))
        # save tokenizer
        self.tokenizer.save_pretrained(ckpt_save_path)
