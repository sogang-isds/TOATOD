from modelling.T5Model import T5Gen_Model
import torch
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
from t5adapter import set_task_for_inference, set_task_for_train

def get_n_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

path = './ckpt/nlg_reinforce_dst_reinforce'
import os
path = os.path.join(path,os.listdir(path)[0])


tokenizer = T5Tokenizer.from_pretrained(path)
toatod = T5Gen_Model(path, tokenizer, None, dropout=0., add_special_decoder_token=False, is_training=False)
model = T5ForConditionalGeneration.from_pretrained("../checkpoints/base")
model.train()
for params in model.parameters():
    params.requires_grad = True
base_model = T5ForConditionalGeneration.from_pretrained("t5-base")

print(get_n_params(toatod))
toatod.model = set_task_for_train(toatod.model,task='dst')
print(get_n_params(toatod))
toatod.model = set_task_for_inference(toatod.model,task='nlg')
print(get_n_params(toatod))
toatod.model = set_task_for_inference(toatod.model,task='dst')
print(get_n_params(toatod))
toatod = toatod.eval()
print(get_n_params(toatod))
print(get_n_params(model))
print(get_n_params(base_model))

# print(model)
# print(base_model)