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
from t5adapter import add_adapter, set_task_for_train, set_task_for_inference, copy_weight
from t5adapter import AdapterLayer
import wandb
from inference_utils import batch_generate
from utils import mwz22_format_change
from mwzeval.metrics import Evaluator


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
    parser.add_argument('--ref_model_path',type=str,default='')
    parser.add_argument('--ref_task',type=str,default='dst',help='dst or nlg')
    # training configuration
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--epoch_num", default=15, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", type=int, default=4, help='Batch size')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="gradient accumulation step.")
    parser.add_argument("--ckpt_save_path", type=str, help="directory to save the model parameters.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--mode", type=str, default="dst", help="dst or nlg")
    parser.add_argument("--alpha", type=float, default=0.7, help="the weight of the reward")
    parser.add_argument("--beta", type=float, default=0.5, help="the weight of the success")
    parser.add_argument('--wandb_name', type=str, help='wandb run name')

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
           name = wandb_name
        )
    else:
        wandb.init()

    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()

    folder = os.listdir(args.pretrained_path)
    assert len(folder) == 1 and folder[0].startswith('epoch')
    args.pretrained_path = os.path.join(args.pretrained_path, folder[0])

    wandb.config.update(args)
    set_seed(args.seed)
    device = torch.device('cuda')
    print(args.model_name)
    assert args.model_name.startswith('t5')
    from transformers import T5Tokenizer
    print ('Loading Pretrained Tokenizer...')
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_path)
    n_gpu = torch.cuda.device_count()
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
    from modelling.reinforce import T5ForReinforce

    model = T5ForReinforce(args.pretrained_path,evaluator,data.special_token_list,alpha=args.alpha, beta=args.beta)

    if args.ref_model_path != '':
        folder = os.listdir(args.ref_model_path)
        assert len(folder) == 1 and folder[0].startswith('epoch')
        args.ref_model_path = os.path.join(args.ref_model_path, folder[0])
        print ('Loading reference model from {}'.format(args.ref_model_path))
        ref_model = T5ForReinforce(args.ref_model_path,evaluator,data.special_token_list,alpha=args.alpha, beta=args.beta)
        model.model = copy_weight(model.model, ref_model.model, args.ref_task)
        del ref_model


    else:
        pass
    wandb.watch(model)
    print ('Model loaded')

    if n_gpu > 1:
        multi_gpu_training = True
    else:
        multi_gpu_training = False

    if multi_gpu_training:
        model = model.cuda()
        model = torch.nn.DataParallel(model)
    else:
        model = model.cuda()
    optimizer, _ = get_optimizers(model, args)
    optimizer.zero_grad()


    max_dev_score, max_dev_str = 0., ''
    nlg_step = 0

    if args.mode == 'nlg':
        return_level = 'session'

    else:
        return_level = 'turn'
    ref_bs, ref_act, ref_db = False, False, False  # we only consider e2e evaluation
    input_contain_db = use_db_as_input
    # if n_gpu > 4:
    #     batch_gpu = 4
    # else:
    #     batch_gpu = n_gpu
    dev_batch_list = data.build_all_evaluation_batch_list(ref_bs, ref_act, ref_db, input_contain_db,
                                                          eva_batch_size=n_gpu * (args.batch_size if \
                                                                                          return_level != 'session' else 32),
                                                          eva_mode='dev')
    for epoch in range(args.epoch_num):
        print(f'Start training {args.mode.upper()} Adapter with REINFORCE at epoch %d' % epoch)
        model.train()
        if multi_gpu_training:
            model.module.model = set_task_for_train(model.module.model, args.mode)
        else:
            model.model = set_task_for_train(model.model, args.mode)
        if return_level == 'session':
            data.train_num = 8433
            if 'multiwoz22' in args.data_path_prefix :
                data.train_num = 8426
        train_iterator = data.build_iterator(batch_size=n_gpu * args.batch_size, mode='train',
                                             task=args.mode, return_level=return_level)
        train_batch_num_per_epoch = int(data.train_num / (n_gpu * args.batch_size))
        print("train batch num per epoch: ", train_batch_num_per_epoch)
        print("train batch size: ", n_gpu * args.batch_size)
        print("train data num: ", data.train_num)
        p = progressbar.ProgressBar(train_batch_num_per_epoch)
        p.start()
        p_train_idx = 0
        epoch_step, train_loss, train_reward = 0, 0., 0.
        match, success, bleu, combined_score = 0., 0., 0., 0.

        for step, train_batch in enumerate(train_iterator):
            p.update(p_train_idx)
            p_train_idx += 1
            dial_id = None
            if args.mode == 'dst':
                one_train_input_batch, one_train_output_batch = train_batch
            elif args.mode == 'nlg':
                one_train_input_batch, one_train_output_batch, dial_ids, one_dial_batch = train_batch
            if len(one_train_input_batch) == 0 or len(one_train_output_batch) == 0: break


            train_batch_src_tensor, train_batch_src_mask, train_batch_input, train_batch_labels = \
                data.parse_batch_tensor([one_train_input_batch, one_train_output_batch])

            if cuda_available:
                train_batch_src_tensor = train_batch_src_tensor.to(device)
                train_batch_src_mask = train_batch_src_mask.to(device)
                train_batch_input = train_batch_input.to(device)
                train_batch_labels = train_batch_labels.to(device)
                batch = [train_batch_src_tensor, train_batch_src_mask, train_batch_input, train_batch_labels]

            if args.mode == 'nlg':
                if '22' in args.data_path_prefix:
                    ver = '2.2'
                else:
                    ver = '2.1'

                loss, reward, m, s, b,c = model(batch=batch,
                                     mode=args.mode,
                                     dial_id=dial_ids,
                                     dials=one_dial_batch,
                                     ver=ver)
                m = m.mean().item()
                s = s.mean().item()
                b = b.mean().item()
                c = c.mean().item()

            else:
                loss, reward = model(batch=batch,
                                    mode=args.mode,
                                    dial_id=dial_id,)

            loss = loss.mean()
            loss.backward()

            train_loss += loss.item()
            reward = reward.mean().item()
            train_reward += reward
            if args.mode == 'nlg':
                wandb.log({f'train_{args.mode}_reward': reward, "loss": loss.item(), "match": m, "success": s, "bleu": b,
                            'combined_score':c,args.mode:nlg_step})
                match += m
                success += s
                bleu += b
                combined_score += c
            else:
                wandb.log({f'train_{args.mode}_reward': reward ,"loss":loss.item(),args.mode:nlg_step})
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            epoch_step += 1
            nlg_step += 1
            if (epoch_step + 1) % args.gradient_accumulation_steps == 0 or (
                    epoch_step + 1) == train_batch_num_per_epoch:
                optimizer.step()
                optimizer.zero_grad()

        p.finish()
        train_loss = train_loss / step
        train_reward = train_reward / step
        if args.mode == 'nlg':
            match = match / step
            success = success / step
            bleu = bleu / step
            combined_score = combined_score / step
            wandb.log({f"match_per_epoch": match, "success_per_epoch": success, "bleu_per_epoch": bleu,"combined_score_per_epoch":combined_score,"epoch":epoch})
        wandb.log({f'train_{args.mode}_loss_per_epoch': train_loss,"train_reward_per_epoch":train_reward,"epoch":epoch})
        print('At epoch %d, total update steps is %d, the total training loss is %5f' % (epoch, epoch_step, train_loss))
        print('++++++++++++++++++++++++++++++++++++++++++')

        # --- evaluation --- #

        print ('Start evaluation at epoch %d' % epoch)

        model.eval()

        with torch.no_grad():

            dev_batch_num_per_epoch = len(dev_batch_list)
            p = progressbar.ProgressBar(dev_batch_num_per_epoch)
            print('Number of evaluation batches is %d' % dev_batch_num_per_epoch)
            p.start()
            all_dev_result = []
            if args.mode == 'nlg':
                for p_dev_idx in range(dev_batch_num_per_epoch):
                    p.update(p_dev_idx)
                    one_inference_batch = dev_batch_list[p_dev_idx]
                    dev_batch_parse_dict = batch_generate(model, one_inference_batch, ref_bs, ref_act, ref_db,
                                                          input_contain_db, data, mode="nlg")
                    for item in dev_batch_parse_dict:
                        all_dev_result.append(item)
                p.finish()
                if '22' in args.data_path_prefix:
                    ver = '2.2'
                else:
                    ver = '2.1'
                dev_bleu, dev_success, dev_match = evaluator.validation_metric(all_dev_result,ver=ver)

                dev_score = 0.5 * (dev_success + dev_match) + dev_bleu
                print('INFORM: %2.2f, SUCCESS: %2.2f, BLEU: %2.2f, COMBINED: %2.2f' % (dev_match, dev_success, dev_bleu,\
                                                                                       dev_score))
                one_dev_str = 'inform_{:.2f}_success{:.2f}_bleu_{:.2f}_combined{:.2f}'.format(dev_match, dev_success,\
                                                                                             dev_bleu, dev_score)
                wandb.log({f"dev_{args.mode}_match": dev_match, f"dev_{args.mode}_success": dev_success,
                            f"dev_{args.mode}_bleu": dev_bleu, f"dev_{args.mode}_combined_score": dev_score,
                            f"dev_{args.mode}_epoch": epoch})
                if dev_score > max_dev_score:
                    max_dev_str = one_dev_str
                    max_dev_score = dev_score
                    print('Saving Model...')
                    model_save_path = args.ckpt_save_path + '/epoch_' + str(epoch) + '_' + one_dev_str
                    # model_save_path = args.ckpt_save_path + '/epoch_' + str(epoch) + '_best_ckpt'

                    import os

                    if os.path.exists(model_save_path):
                        pass
                    else:  # recursively construct directory
                        os.makedirs(model_save_path, exist_ok=True)

                    if cuda_available and torch.cuda.device_count() > 1:
                        model.module.save_model(model_save_path)
                    else:
                        model.save_model(model_save_path)

                    pkl_save_path = model_save_path + '/' + one_dev_str + '.json'
                    with open(pkl_save_path, 'w') as outfile:
                        json.dump(all_dev_result, outfile, indent=4)
                    print('Validation result saved.')
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
                            print(one_folder_name)
                            os.system('rm -r ' + one_folder_name)
                    print('-----------------------------------')
                    # --------------------------------------------------------------------------------------------- #
            elif args.mode == 'dst':
                e = Evaluator(success=False, bleu=False, richness=False, dst=True)
                for p_dev_idx in range(dev_batch_num_per_epoch):
                    p.update(p_dev_idx)
                    one_inference_batch = dev_batch_list[p_dev_idx]
                    dev_batch_parse_dict = batch_generate(model, one_inference_batch, ref_bs, ref_act, ref_db,
                                                          input_contain_db, data, mode="dst")
                    for item in dev_batch_parse_dict:
                        all_dev_result.append(item)
                p.finish()
                if '22' in args.data_path_prefix:
                    ver = '2.2'
                else:
                    ver = '2.1'

                if ver == '2.1':
                    all_dev_result = evaluator.wrap_evaluation_result(all_dev_result)
                    jga, f1, acc, _, _ = evaluator.dialog_state_tracking_eval(all_dev_result)
                    wandb.log({'dev_jga': jga, 'dev_f1': f1, 'dev_acc': acc, "epoch": epoch})
                    print('JGA: %2.2f, F1: %2.2f, ACC: %2.2f' % (jga, f1, acc))
                    one_dev_str = 'dev_jga{:.2f}_f1{:.2f}_acc{:.2f}'.format(jga, f1, acc)

                elif ver == '2.2':
                    parse_dict = mwz22_format_change(all_dev_result, data.reader)
                    dst_result = e.evaluate(parse_dict)
                    dst_result = dst_result['dst']
                    jga, slot_f1 = dst_result['joint_accuracy'], dst_result['slot_f1']
                    wandb.log({'dev_jga': jga, 'dev_slot_f1': slot_f1, "epoch": epoch})
                    print('JGA: {:.2f}  F1: {:.2f}'.format(jga, slot_f1))
                    one_dev_str = 'dev_e2e_evaluation_jga_{}_f1_{}'.format(round(jga, 2),
                                                                           round(slot_f1, 2))

                if jga > max_dev_score:
                    max_dev_str = one_dev_str
                    max_dev_score = jga
                    print('Saving Model...')
                    # model_save_path = args.ckpt_save_path + '/epoch_' + str(epoch) + '_' + one_dev_str
                    model_save_path = args.ckpt_save_path + '/epoch_' + str(epoch) + '_best_ckpt'

                    import os

                    if os.path.exists(model_save_path):
                        pass
                    else:  # recursively construct directory
                        os.makedirs(model_save_path, exist_ok=True)

                    if cuda_available and torch.cuda.device_count() > 1:
                        model.module.save_model(model_save_path)
                    else:
                        model.save_model(model_save_path)

                    pkl_save_path = model_save_path + '/' + one_dev_str + '.json'
                    with open(pkl_save_path, 'w') as outfile:
                        json.dump(all_dev_result, outfile, indent=4)
                    print('Validation result saved.')
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
                            print(one_folder_name)
                            os.system('rm -r ' + one_folder_name)
                    print('-----------------------------------')
                    # --------------------------------------------------------------------------------------------- #

            print('Current Result: ' + one_dev_str)
            print('Best Result: ' + max_dev_str)
            print('dev evaluation finished.')
        print('-----------------------------------------')
        model.train()

