import argparse
import os

import torch
import progressbar
from transformers import T5Tokenizer

from eval import MultiWozEvaluator
from mwzeval.metrics import Evaluator
from modelling.T5Model import T5Gen_Model
from modelling.reinforce import T5ForReinforce
from dataclass import MultiWozData
from config import Config
from inference_utils import batch_generate
from t5adapter import copy_weight

def parse_config():
    parser = argparse.ArgumentParser()
    # dataset configuration
    parser.add_argument('--data_path_prefix', type=str, help='The path where the data stores.')
    parser.add_argument('--use_db_as_input', type=str, default='True',
        help="True or False, whether includes db result as part of the input when generating response.")

    parser.add_argument('--add_special_decoder_token', default='True', type=str, help='Whether we discriminate the decoder start and end token for different tasks.')
    parser.add_argument('--model_name',default='t5-small',type=str)
    parser.add_argument('--pretrained_path', type=str, help='the path that stores pretrained checkpoint.')
    parser.add_argument("--batch_size", type=int, default=4, help='Batch size')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="gradient accumulation step.")
    parser.add_argument("--model_type", type=str, default="reinforce", help="reinforce or gen")
    parser.add_argument("--dataset", type=str, default="test", help="dev or test")
    parser.add_argument("--mode", type=str, default="dst", help="dst or nlg")
    parser.add_argument("--ref_model_path",type=str,default="")
    parser.add_argument("--ref_task",type=str,default="dst")
    return parser.parse_args()

args = parse_config()

folder_name = os.listdir(args.pretrained_path)
assert len(folder_name) == 1, "there are multiple file or dir"
assert folder_name[0].startswith("epoch"), "It is not model dir"

args.pretrained_path = os.path.join(args.pretrained_path,f'{folder_name[0]}')

if 'multiwoz22' in args.data_path_prefix:
    ver = '2.2'
else:
    ver = '2.1'

if args.use_db_as_input == 'True':
    use_db_as_input = True
elif args.use_db_as_input == 'False':
    use_db_as_input = False

if args.add_special_decoder_token == 'True':
    add_special_decoder_token = True
elif args.add_special_decoder_token == 'False':
    add_special_decoder_token = False

tokenizer = T5Tokenizer.from_pretrained(args.pretrained_path)
cfg = Config(args.data_path_prefix)
data = MultiWozData(args.model_name, tokenizer, cfg, args.data_path_prefix, shuffle_mode=False,
        data_mode='test', use_db_as_input=use_db_as_input, add_special_decoder_token=add_special_decoder_token)

print ('Data loaded')
n_gpu = torch.cuda.device_count()
evaluator = MultiWozEvaluator(data.reader, cfg)
if args.model_type == 'reinforce':
    model = T5ForReinforce(args.pretrained_path,evaluator,data.special_token_list)
else:
    model = T5Gen_Model(args.pretrained_path, data.tokenizer, data.special_token_list, dropout=0.,
                        add_special_decoder_token=add_special_decoder_token, is_training=False)

if args.ref_model_path != "":
    args.ref_model_path = os.path.join(args.ref_model_path,os.listdir(args.ref_model_path)[0])
    if args.model_type == 'reinforce':
        ref_model = T5ForReinforce(args.ref_model_path,evaluator,data.special_token_list)
    else:
        ref_model = T5Gen_Model(args.ref_model_path, data.tokenizer, data.special_token_list, dropout=0.,
                            add_special_decoder_token=add_special_decoder_token, is_training=False)
    model.model = copy_weight(model.model,ref_model.model,args.ref_task)
    del ref_model

model = model.cuda()

if n_gpu > 1:
    model = torch.nn.DataParallel(model)


with torch.no_grad():
    ref_bs, ref_act, ref_db = False, False, False
    input_contain_db = use_db_as_input
    dev_batch_list = data.build_all_evaluation_batch_list(ref_bs, ref_act, ref_db, input_contain_db,
                                                          eva_batch_size=n_gpu * args.batch_size,
                                                          eva_mode=args.dataset)
    dev_batch_num_per_epoch = len(dev_batch_list)
    p = progressbar.ProgressBar(dev_batch_num_per_epoch)
    print('Number of evaluation batches is %d' % dev_batch_num_per_epoch)
    p.start()
    all_dev_result = []
    for p_dev_idx in range(dev_batch_num_per_epoch):
        p.update(p_dev_idx)
        one_inference_batch = dev_batch_list[p_dev_idx]
        dev_batch_parse_dict = batch_generate(model, one_inference_batch, ref_bs, ref_act, ref_db,
                                              input_contain_db, data, args.mode)
        for item in dev_batch_parse_dict:
            all_dev_result.append(item)
    p.finish()

    if ver == '2.1':
        all_dev_result_wrap = evaluator.wrap_evaluation_result(all_dev_result)

        jga,f1, acc, _, _ = evaluator.dialog_state_tracking_eval(all_dev_result_wrap)
        print('JGA: %.4f, F1: %.4f, ACC: %.4f' % (jga, f1, acc))
        one_dev_str = 'jga_{:.2f}_f1_{:.2f}_acc_{:.2f}.json'.format(jga, f1, acc)

        if args.mode != 'dst':
            dev_bleu, dev_success, dev_match = evaluator.validation_metric(all_dev_result)

            dev_score = 0.5 * (dev_success + dev_match) + dev_bleu
            print('Inform: %2.2f  Success: %2.2f  BLEU: %2.2f    Score: %.2f' % (dev_match, dev_success, dev_bleu, dev_score))
            one_dev_str2 = 'bleu_{:.2f}_success_{:.2f}_match_{:.2f}_score_{:.2f}'.format(dev_bleu, dev_success, dev_match, dev_score)
            one_dev_str = one_dev_str2 + '_' + one_dev_str
    else:
        from utils import mwz22_format_change

        parse_dict = mwz22_format_change(all_dev_result,evaluator.reader)
        evaluator = Evaluator(True, True, True, True)
        results = evaluator.evaluate(parse_dict)

        dst_result = results['dst']
        jga, f1, precision, recall = dst_result['joint_accuracy'], dst_result['slot_f1'],\
                                              dst_result['slot_precision'], dst_result['slot_recall']
        print('JGA: %.4f, F1: %.4f, PRECISION: %.4f RECALL: %.4f' % (jga, f1, precision, recall))
        one_dev_str = 'jga_{:.2f}_f1_{:.2f}_precision_{:.2f}.json'.format(jga, f1, precision)
        if args.mode != 'dst':
            print(results)
            dev_bleu, dev_success, dev_match = results['bleu']["mwz22"], results['success']['success']['total'],\
                                               results['success']['inform']['total']
            dev_score = 0.5 * (dev_success + dev_match) + dev_bleu
            print('Inform: %2.2f  Success: %2.2f  BLEU: %2.2f    Score: %.2f' % (
            dev_match, dev_success, dev_bleu, dev_score))
            one_dev_str2 = 'bleu_{:.2f}_success_{:.2f}_match_{:.2f}_score_{:.2f}'.format(dev_bleu, dev_success,
                                                                                         dev_match, dev_score)
            one_dev_str = one_dev_str2 + '_' + one_dev_str

    import json
    with open(args.pretrained_path + f'/{args.dataset}_{ver}_' + one_dev_str, 'w') as f:
        json.dump(all_dev_result, f, indent=2)