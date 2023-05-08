import os
import sys
import json
import random
import torch
from torch import nn
import torch.nn.functional as F
import operator
from operator import itemgetter
import progressbar
import argparse
from eval import MultiWozEvaluator
from t5adapter import add_adapter, set_task_for_train, set_task_for_inference
from t5adapter import AdapterLayer
from mwzeval.metrics import Evaluator
from utils import mwz22_format_change
import wandb


def parse_config():
    parser = argparse.ArgumentParser()
    # dataset configuration
    parser.add_argument('--data_path_prefix', type=str, help='The path where the data stores.')

    parser.add_argument('--shuffle_mode', type=str, default='shuffle_session_level', 
        help="shuffle_session_level or shuffle_turn_level, it controls how we shuffle the training data.")

    parser.add_argument('--use_db_as_input', type=str, default='True', 
        help="True or False, whether includes db result as part of the input when generating response.")

    parser.add_argument('--add_special_decoder_token', default='True', type=str, help='Whether we discriminate the decoder start and end token for different tasks.')

    parser.add_argument('--train_data_ratio', type=float, default=1.0, help='the ratio of training data used for training the model')

    # model configuration
    parser.add_argument('--model_name', type=str, help='t5-small, t5-base or t5-large')

    parser.add_argument('--pretrained_path', type=str, help='the path that stores pretrained checkpoint.')

    # training configuration
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--epoch_num", default=30, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", type=int, default=4, help='Batch size')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation step.")
    parser.add_argument("--ckpt_save_path", type=str, help="directory to save the model parameters.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--dst",action='store_true', help="whether to train dst")
    parser.add_argument("--nlg",action='store_true', help="whether to train nlg")
    parser.add_argument("--policy",action='store_true', help="whether to train policy")
    parser.add_argument("--lr",type=float,default=0.0001, help="learning rate")
    parser.add_argument("--weight_path",type=str, default='', help="path to load the fine-tuned model")
    parser.add_argument('--wandb_name', type=str, help='wandb run name')
    parser.add_argument('--devide_num_adapter_units', type=int, default=2)
    return parser.parse_args()


def get_optimizers(model, args):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    from transformers.optimization import Adafactor
    optimizer = Adafactor(
        optimizer_grouped_parameters,
        lr=args.lr,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False
        )
    scheduler = None
    return optimizer, scheduler

def set_seed(seed):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

import argparse
if __name__ == '__main__':
    args = parse_config()

    if args.wandb_name:
        wandb_name = args.wandb_name
        wandb.init(
            project="adapter",
            name=wandb_name
        )
    else:
        wandb.init()

    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()

    wandb.config.update(args)
    set_seed(args.seed)
    device = torch.device('cuda')

    assert args.model_name.startswith('t5'), f"{args.model_name}"
    from transformers import T5Tokenizer
    print ('Loading Pretrained Tokenizer...')
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_path)

    if args.use_db_as_input == 'True':
        use_db_as_input = True
    elif args.use_db_as_input == 'False':
        use_db_as_input = False
    else:
        raise Exception('Wrong Use DB Mode!!!')

    if args.add_special_decoder_token == 'True':
        add_special_decoder_token = True
    elif args.add_special_decoder_token == 'False':
        add_special_decoder_token = False
    else:
        raise Exception('Wrong Add Special Token Mode!!!')

    print ('Start loading data...')
    from dataclass import MultiWozData
    from config import Config
    cfg = Config(args.data_path_prefix)

    data = MultiWozData(args.model_name, tokenizer, cfg, args.data_path_prefix, shuffle_mode=args.shuffle_mode, 
        data_mode='train', use_db_as_input=use_db_as_input, add_special_decoder_token=add_special_decoder_token, 
        train_data_ratio=args.train_data_ratio)
    print ('Data loaded')
    evaluator = MultiWozEvaluator(data.reader, cfg)

    print ('Start loading model...')
    from modelling.T5Model import T5Gen_Model
    model = T5Gen_Model(args.pretrained_path, data.tokenizer, data.special_token_list, dropout=args.dropout, 
        add_special_decoder_token=add_special_decoder_token, is_training=True)
    dim = model.model.config.d_model
    down_dim = int(dim // args.devide_num_adapter_units)
    print(f"Model dimension: {dim}, Adapter down dimension: {down_dim}")
    model.model = add_adapter(model.model, AdapterLayer, {'dim':dim, 'down_dim':down_dim},['dst', 'policy', 'nlg'])
    if args.weight_path != '':
        folder = os.listdir(args.weight_path)
        assert len(folder) == 1 and folder[0].startswith('epoch')
        model.model = torch.load(os.path.join(args.weight_path, folder[0], 'model.pt'))
    model = model.to(device)
    wandb.watch(model)
    print ('Model loaded')
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        multi_gpu_training = True
    else:
        multi_gpu_training = False

    if multi_gpu_training:
        model = torch.nn.DataParallel(model)
    optimizer, _ = get_optimizers(model, args)
    optimizer.zero_grad()

    min_dev_loss = 1e10
    max_dev_jga, max_dev_score, max_dev_str = 0.,0., ''
    dst_step, policy_step, nlg_step = 0, 0, 0
    if 'multiwoz22' in args.data_path_prefix:
        ver = '2.2'
    else:
        ver = '2.1'
    ref_bs, ref_act, ref_db = False, False, False  # we only consider e2e evaluation
    input_contain_db = use_db_as_input
    dev_batch_list = data.build_all_evaluation_batch_list(ref_bs, ref_act, ref_db, input_contain_db,
                                                          eva_batch_size=n_gpu * args.batch_size, eva_mode='dev')
    dev_batch_num_per_epoch = len(dev_batch_list)
    if args.dst:
        e = Evaluator(success=False, bleu=False, richness=False, dst=True)
        for epoch in range(args.epoch_num):
            model.train()
            # turn on dst adapter and turn off the others
            if multi_gpu_training:
                model.module.model = set_task_for_train(model.module.model, "dst")
            else:
                model.model = set_task_for_train(model.model, "dst")
            # --- training --- #
            print ('-----------------------------------------')
            print ('Start training DST Adapter at epoch %d' % epoch)
            train_iterator = data.build_iterator(batch_size=n_gpu * args.batch_size, mode='train',task="dst")
            train_batch_num_per_epoch = int(data.train_num / (n_gpu * args.batch_size))
            p = progressbar.ProgressBar(train_batch_num_per_epoch)
            p.start()
            p_train_idx = 0
            epoch_step, train_loss = 0, 0.
            for _, train_batch in enumerate(train_iterator):
                p.update(p_train_idx)
                p_train_idx += 1
                one_train_input_batch, one_train_output_batch = train_batch
                if len(one_train_input_batch) == 0 or len(one_train_output_batch) == 0: break
                for i in range(len(train_batch)):
                    input_batch = train_batch[0]
                    output_batch = train_batch[1]

                train_batch_src_tensor, train_batch_src_mask, train_batch_input, train_batch_labels = \
                data.parse_batch_tensor(train_batch)
                if cuda_available:
                    train_batch_src_tensor = train_batch_src_tensor.to(device)
                    train_batch_src_mask = train_batch_src_mask.to(device)
                    train_batch_input = train_batch_input.to(device)
                    train_batch_labels = train_batch_labels.to(device)

                loss = model(train_batch_src_tensor, train_batch_src_mask, train_batch_input, train_batch_labels)
                loss = loss.mean()
                loss.backward()
                train_loss += loss.item()
                wandb.log({'train_dst_loss': loss.item(),"dst_step":dst_step})
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                epoch_step += 1
                dst_step += 1
                if (epoch_step+1) % args.gradient_accumulation_steps == 0 or (epoch_step + 1) == train_batch_num_per_epoch:
                    optimizer.step()
                    optimizer.zero_grad()

            p.finish()
            train_loss = train_loss / train_batch_num_per_epoch
            wandb.log({'train_dst_loss_per_epoch': train_loss,"epoch":epoch})
            print ('At epoch %d, total update steps is %d, the total training loss is %5f' % (epoch, epoch_step, train_loss))
            print ('++++++++++++++++++++++++++++++++++++++++++')

            if args.train_data_ratio <= 0.01:
                if args.pretrained_path == 'None':
                    if epoch < 60:  # first train 10 epoches
                        continue
                else:
                    if epoch < 50:  # first train 5 epoches
                        continue
            elif args.train_data_ratio <= 0.2:
                if epoch < 10:  # first train 3 epoches
                    continue
            else:
                pass
            # **********************************************************************
            # --- evaluation --- #
            from inference_utils import batch_generate
            print ('Start evaluation at epoch %d' % epoch)
            model.eval()
            #set_task_for_inference(model,"dst")
            with torch.no_grad():

                p = progressbar.ProgressBar(dev_batch_num_per_epoch)
                print ('Number of evaluation batches is %d' % dev_batch_num_per_epoch)
                p.start()
                all_dev_result = []
                for p_dev_idx in range(dev_batch_num_per_epoch):
                    p.update(p_dev_idx)
                    one_inference_batch = dev_batch_list[p_dev_idx]

                    dev_batch_parse_dict = batch_generate(model, one_inference_batch, ref_bs, ref_act, ref_db,
                        input_contain_db, data, 'dst')

                    for item in dev_batch_parse_dict:
                        all_dev_result.append(item)
                p.finish()
                if ver == '2.1':
                    all_dev_result = evaluator.wrap_evaluation_result(all_dev_result)
                    jg, slot_f1, slot_acc, slot_cnt, slot_corr = evaluator.dialog_state_tracking_eval(all_dev_result)
                    wandb.log({'dev_jga': jg, 'dev_slot_f1': slot_f1, 'dev_slot_acc': slot_acc, "epoch": epoch})
                    print('JGA: %2.2f  F1: %2.2f  ACC: %2.2f' % (jg, slot_f1, slot_acc))
                    one_dev_str = 'dev_e2e_evaluation_jga_{}_f1_{}_acc_{}'.format(round(jg, 2),
                                                                                  round(slot_f1, 2), round(slot_acc, 2))

                elif ver == '2.2':
                    parse_dict = mwz22_format_change(all_dev_result,data.reader)
                    dst_result = e.evaluate(parse_dict)
                    dst_result = dst_result['dst']
                    jg, slot_f1, slot_prec, slot_recall = dst_result['joint_accuracy'], dst_result['slot_f1'], dst_result['slot_precision'], dst_result['slot_recall']
                    wandb.log({'dev_jga': jg, 'dev_slot_f1': slot_f1, 'dev_slot_prec': slot_prec, 'dev_slot_recall': slot_recall, "epoch": epoch})
                    print('JGA: %2.2f  F1: %2.2f  PRECISION: %2.2f  RECALL: %2.2f' % (jg, slot_f1, slot_prec, slot_recall))
                    one_dev_str = 'dev_e2e_evaluation_jga_{}_f1_{}'.format(round(jg, 2),
                                                                           round(slot_f1, 2))

                if jg > max_dev_jga:
                    max_dev_str = one_dev_str
                    max_dev_jga = jg
                    print ('Saving Model...')
                    model_save_path = args.ckpt_save_path + '/epoch_' + str(epoch) + '_best_ckpt'
                    print('Model Saved')

                    import os
                    if os.path.exists(model_save_path):
                        pass
                    else: # recursively construct directory
                        os.makedirs(model_save_path, exist_ok=True)

                    if multi_gpu_training:
                        model.module.save_model(model_save_path)
                    else:
                        model.save_model(model_save_path)

                    pkl_save_path = model_save_path + '/' + one_dev_str + '.json'
                    with open(pkl_save_path, 'w') as outfile:
                        json.dump(all_dev_result, outfile, indent=4)
                    print ('Validation result saved.')
                    # --------------------------------------------------------------------------------------------- #
                    # removing extra checkpoints...
                    # only save 1 checkpoints
                    import os
                    from operator import itemgetter
                    fileData = {}
                    test_output_dir = args.ckpt_save_path
                    for fname in os.listdir(test_output_dir):
                        if fname.startswith('epoch'):
                            fileData[fname] = os.stat(test_output_dir + '/' + fname).st_mtime
                        else:
                            pass
                    sortedFiles = sorted(fileData.items(), key=itemgetter(1))
                    max_save_num = 1
                    if len(sortedFiles) < max_save_num:
                        pass
                    else:
                        delete = len(sortedFiles) - max_save_num
                        for x in range(0, delete):
                            one_folder_name = test_output_dir + '/' + sortedFiles[x][0]
                            print (one_folder_name)
                            os.system('rm -r ' + one_folder_name)
                    print ('-----------------------------------')
                    # --------------------------------------------------------------------------------------------- #

                print ('Current Result: ' + one_dev_str)
                print ('Best Result: ' + max_dev_str)
                print ('dev evaluation finished.')
            print ('-----------------------------------------')
            model.train()
            optimizer.zero_grad()

    if args.policy:
        for epoch in range(args.epoch_num):
            model.train()
            if multi_gpu_training:
                model.module.model = set_task_for_train(model.module.model, "policy")
            else:
                model.model = set_task_for_train(model.model, "policy")
            # turn on policy adapter and turn off the others
            train_iterator = data.build_iterator(batch_size=n_gpu * args.batch_size, mode='train',
                                                 task="policy")
            print('-----------------------------------------')
            print('Start training Policy Adapter at epoch %d' % epoch)
            train_batch_num_per_epoch = int(data.train_num / (n_gpu*args.batch_size))
            p = progressbar.ProgressBar(train_batch_num_per_epoch)
            p.start()
            p_train_idx = 0
            epoch_step, train_loss = 0, 0.
            for _, train_batch in enumerate(train_iterator):
                p.update(p_train_idx)
                p_train_idx += 1
                one_train_input_batch, one_train_output_batch = train_batch
                if len(one_train_input_batch) == 0 or len(one_train_output_batch) == 0: break
                train_batch_src_tensor, train_batch_src_mask, train_batch_input, train_batch_labels = \
                    data.parse_batch_tensor(train_batch)
                if cuda_available:
                    train_batch_src_tensor = train_batch_src_tensor.to(device)
                    train_batch_src_mask = train_batch_src_mask.to(device)
                    train_batch_input = train_batch_input.to(device)
                    train_batch_labels = train_batch_labels.to(device)

                loss = model(train_batch_src_tensor, train_batch_src_mask, train_batch_input, train_batch_labels)
                loss = loss.mean()
                loss.backward()
                train_loss += loss.item()
                wandb.log({'train_policy_loss': loss.item(),"polcy_step":policy_step})
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                epoch_step += 1
                policy_step += 1
                if (epoch_step + 1) % args.gradient_accumulation_steps == 0 or (
                        epoch_step + 1) == train_batch_num_per_epoch:
                    optimizer.step()
                    optimizer.zero_grad()

            p.finish()
            train_loss = train_loss / train_batch_num_per_epoch
            wandb.log({'train_policy_loss_per_epoch': train_loss,"epoch":epoch})
            print('At epoch %d, total update steps is %d, the total training loss is %5f' % (epoch, epoch_step, train_loss))
            print('++++++++++++++++++++++++++++++++++++++++++')

            if args.train_data_ratio <= 0.01:
                if args.pretrained_path == 'None':
                    if epoch < 40:  # first train 10 epoches
                        continue
                else:
                    if epoch < 50:  # first train 3 epoches
                        continue
            elif args.train_data_ratio <= 0.2:
                if epoch < 10:  # first train 3 epoches
                    continue
            else:
                pass
            # **********************************************************************
            # --- evaluation --- #
            from inference_utils import batch_generate
            print ('Start evaluation at epoch %d' % epoch)
            model.eval()
            #set_task_for_inference(model,"dst")
            with torch.no_grad():
                p = progressbar.ProgressBar(dev_batch_num_per_epoch)
                print ('Number of evaluation batches is %d' % dev_batch_num_per_epoch)
                p.start()
                all_dev_result = []
                for p_dev_idx in range(dev_batch_num_per_epoch):
                    p.update(p_dev_idx)
                    one_inference_batch = dev_batch_list[p_dev_idx]
                    dev_batch_parse_dict = batch_generate(model, one_inference_batch, ref_bs, ref_act, ref_db,
                        input_contain_db, data,mode='policy')
                    for item in dev_batch_parse_dict:
                        all_dev_result.append(item)

                p.finish()
                dev_bleu, dev_success, dev_match = evaluator.validation_metric(all_dev_result)

                dev_score = 0.5 * (dev_success + dev_match) + dev_bleu
                wandb.log({'dev_bleu': dev_bleu, 'dev_success': dev_success, 'dev_match': dev_match, 'dev_score': dev_score,"epoch":epoch})
                print ('Inform: %2.2f  Success: %2.2f  BLEU: %2.2f    Score: %.2f' % (dev_match, dev_success, dev_bleu, dev_score))
                one_dev_str = 'dev_e2e_evaluation_inform_{}_success_{}_bleu_{}_combine_score_{}'.format(round(dev_match, 2),
                    round(dev_success,2), round(dev_bleu,2), round(dev_score,2))

                if dev_score > max_dev_score:
                    max_dev_str = one_dev_str
                    max_dev_score = dev_score
                    print ('Saving Model...')
                    model_save_path = args.ckpt_save_path + '/epoch_' + str(epoch) + '_best_ckpt'
                    print('Model Saved')

                    import os
                    if os.path.exists(model_save_path):
                        pass
                    else: # recursively construct directory
                        os.makedirs(model_save_path, exist_ok=True)

                    if cuda_available and torch.cuda.device_count() > 1:
                        model.module.save_model(model_save_path)
                    else:
                        model.save_model(model_save_path)

                    pkl_save_path = model_save_path + '/' + one_dev_str + '.json'
                    with open(pkl_save_path, 'w') as outfile:
                        json.dump(all_dev_result, outfile, indent=4)
                    print ('Validation result saved.')
                    # --------------------------------------------------------------------------------------------- #
                    # removing extra checkpoints...
                    # only save 1 checkpoints
                    import os
                    from operator import itemgetter
                    fileData = {}
                    test_output_dir = args.ckpt_save_path
                    for fname in os.listdir(test_output_dir):
                        if fname.startswith('epoch'):
                            fileData[fname] = os.stat(test_output_dir + '/' + fname).st_mtime
                        else:
                            pass
                    sortedFiles = sorted(fileData.items(), key=itemgetter(1))
                    max_save_num = 1
                    if len(sortedFiles) < max_save_num:
                        pass
                    else:
                        delete = len(sortedFiles) - max_save_num
                        for x in range(0, delete):
                            one_folder_name = test_output_dir + '/' + sortedFiles[x][0]
                            print (one_folder_name)
                            os.system('rm -r ' + one_folder_name)
                    print ('-----------------------------------')
                    # --------------------------------------------------------------------------------------------- #

                print ('Current Result: ' + one_dev_str)
                print ('Best Result: ' + max_dev_str)
                print ('dev evaluation finished.')
            print ('-----------------------------------------')
            model.train()

    if args.nlg:
        for epoch in range(args.epoch_num):
            print('-----------------------------------------')
            print('Start training NLG Adapter at epoch %d' % epoch)
            model.train()
            if multi_gpu_training:
                model.module.model = set_task_for_train(model.module.model, "nlg")
            else:
                model.model = set_task_for_train(model.model, "nlg")
            # turn on policy adapter and turn off the others
            train_iterator = data.build_iterator(batch_size=n_gpu * args.batch_size, mode='train',
                                                 task="nlg")
            train_batch_num_per_epoch = int(data.train_num / (n_gpu * args.batch_size))
            p = progressbar.ProgressBar(train_batch_num_per_epoch)
            p.start()
            p_train_idx = 0
            epoch_step, train_loss = 0, 0.
            for _, train_batch in enumerate(train_iterator):
                p.update(p_train_idx)
                p_train_idx += 1
                one_train_input_batch, one_train_output_batch = train_batch
                if len(one_train_input_batch) == 0 or len(one_train_output_batch) == 0: break
                train_batch_src_tensor, train_batch_src_mask, train_batch_input, train_batch_labels = \
                    data.parse_batch_tensor(train_batch)
                if cuda_available:
                    train_batch_src_tensor = train_batch_src_tensor.to(device)
                    train_batch_src_mask = train_batch_src_mask.to(device)
                    train_batch_input = train_batch_input.to(device)
                    train_batch_labels = train_batch_labels.to(device)

                loss = model(train_batch_src_tensor, train_batch_src_mask, train_batch_input, train_batch_labels)
                loss = loss.mean()
                loss.backward()
                train_loss += loss.item()
                wandb.log({'train_nlg_loss': loss.item(),"nlg_step":nlg_step})
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                epoch_step += 1
                nlg_step += 1
                if (epoch_step + 1) % args.gradient_accumulation_steps == 0 or (
                        epoch_step + 1) == train_batch_num_per_epoch:
                    optimizer.step()
                    optimizer.zero_grad()

            p.finish()
            train_loss = train_loss / train_batch_num_per_epoch
            wandb.log({'train_nlg_loss_per_epoch': train_loss,"epoch":epoch})
            print('At epoch %d, total update steps is %d, the total training loss is %5f' % (epoch, epoch_step, train_loss))
            print('++++++++++++++++++++++++++++++++++++++++++')
            # **********************************************************************
            # for few-shot learning, we let the model to first train for several epochs

            if args.train_data_ratio <= 0.01:
                if args.pretrained_path == 'None':
                    if epoch < 40:  # first train 10 epoches
                        continue
                else:
                    if epoch < 50:  # first train 3 epoches
                        continue
            elif args.train_data_ratio <= 0.2:
                if epoch < 10:  # first train 3 epoches
                    continue
            else:
                pass
            # **********************************************************************
            # --- evaluation --- #
            from inference_utils import batch_generate
            print ('Start evaluation at epoch %d' % epoch)
            model.eval()
            #set_task_for_inference(model,"dst")
            with torch.no_grad():
                p = progressbar.ProgressBar(dev_batch_num_per_epoch)
                print ('Number of evaluation batches is %d' % dev_batch_num_per_epoch)
                p.start()
                all_dev_result = []
                for p_dev_idx in range(dev_batch_num_per_epoch):
                    p.update(p_dev_idx)
                    one_inference_batch = dev_batch_list[p_dev_idx]
                    dev_batch_parse_dict = batch_generate(model, one_inference_batch, ref_bs, ref_act, ref_db,
                        input_contain_db, data,mode='nlg')
                    for item in dev_batch_parse_dict:
                        all_dev_result.append(item)
                p.finish()
                if 'multiwoz22' in args.data_path_prefix:
                    ver = '2.2'
                else:
                    ver = '2.1'
                dev_bleu, dev_success, dev_match = evaluator.validation_metric(all_dev_result,ver=ver)

                dev_score = 0.5 * (dev_success + dev_match) + dev_bleu
                wandb.log({'dev_bleu': dev_bleu, 'dev_success': dev_success, 'dev_match': dev_match, 'dev_score': dev_score,"epoch":epoch})
                print ('Inform: %2.2f  Success: %2.2f  BLEU: %2.2f    Score: %.2f' % (dev_match, dev_success, dev_bleu, dev_score))
                one_dev_str = 'dev_e2e_evaluation_inform_{}_success_{}_bleu_{}_combine_score_{}'.format(round(dev_match, 2),
                    round(dev_success,2), round(dev_bleu,2), round(dev_score,2))

                if dev_score > max_dev_score:
                    max_dev_str = one_dev_str
                    max_dev_score = dev_score
                    print ('Saving Model...')
                    model_save_path = args.ckpt_save_path + '/epoch_' + str(epoch) + '_best_ckpt'
                    print('Model Saved')

                    import os
                    if os.path.exists(model_save_path):
                        pass
                    else: # recursively construct directory
                        os.makedirs(model_save_path, exist_ok=True)

                    if cuda_available and torch.cuda.device_count() > 1:
                        model.module.save_model(model_save_path)
                    else:
                        model.save_model(model_save_path)

                    pkl_save_path = model_save_path + '/' + one_dev_str + '.json'
                    with open(pkl_save_path, 'w') as outfile:
                        json.dump(all_dev_result, outfile, indent=4)
                    print ('Validation result saved.')
                    # --------------------------------------------------------------------------------------------- #
                    # removing extra checkpoints...
                    # only save 1 checkpoints
                    import os
                    from operator import itemgetter
                    fileData = {}
                    test_output_dir = args.ckpt_save_path
                    for fname in os.listdir(test_output_dir):
                        if fname.startswith('epoch'):
                            fileData[fname] = os.stat(test_output_dir + '/' + fname).st_mtime
                        else:
                            pass
                    sortedFiles = sorted(fileData.items(), key=itemgetter(1))
                    max_save_num = 1
                    if len(sortedFiles) < max_save_num:
                        pass
                    else:
                        delete = len(sortedFiles) - max_save_num
                        for x in range(0, delete):
                            one_folder_name = test_output_dir + '/' + sortedFiles[x][0]
                            print (one_folder_name)
                            os.system('rm -r ' + one_folder_name)
                    print ('-----------------------------------')
                    # --------------------------------------------------------------------------------------------- #

                print ('Current Result: ' + one_dev_str)
                print ('Best Result: ' + max_dev_str)
                print ('dev evaluation finished.')
            print ('-----------------------------------------')
            model.train()
