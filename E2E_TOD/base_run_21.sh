# MultiWOZ2.1 Supervised Learning (base)
CUDA_VISIBLE_DEVICES="0,1,2,3" python learn.py --data_path_prefix ../data/multiwoz21 --model_name t5-base --pretrained_path ../checkpoints/base --batch_size 16 --ckpt_save_path ./ckpt/dst_base --dst --lr 1e-4 --epoch_num 15
CUDA_VISIBLE_DEVICES="0,1,2,3" python learn.py --data_path_prefix ../data/multiwoz21 --model_name t5-base --pretrained_path ../checkpoints/base --weight_path ./ckpt/dst_base --batch_size 16 --ckpt_save_path ./ckpt/nlg_base --nlg --lr 1e-4 --epoch_num 15

# Evaluation: Result will be saved in ckpt/nlg_base
python evaluation.py --data_path_prefix ../data/multiwoz21 --model_name t5-base --pretrained_path ./ckpt/nlg_base --batch_size 64 --ckpt_save_path ./ckpt/nlg_base --mode nlg

# MultiWOZ2.1 Reinforcement Learning (base) DST
python reinforce.py --data_path_prefix ../data/multiwoz21 --model_name t5-base --pretrained_path ./ckpt/dst_base --batch_size 32 --ckpt_save_path ./ckpt/dst_base_reinforce --lr 1e-5 --mode dst --epoch_num 10 --alpha 1.0

# MultiWOZ2.1 Reinforcement Learning (base) NLG
python reinforce.py --data_path_prefix ../data/multiwoz21 --model_name t5-base --pretrained_path ./ckpt/nlg_base --batch_size 4 --ckpt_save_path ./ckpt/nlg_base_reinforce --lr 1e-6 --mode nlg --epoch_num 3 --alpha 0.5 --beta 0.7

# Evaluation: Result will be saved in ckpt/nlg_base_reinforce
python evaluation.py --data_path_prefix ../data/multiwoz21 --model_name t5-base --pretrained_path ./ckpt/nlg_base_reinforce --batch_size 64 --mode nlg
