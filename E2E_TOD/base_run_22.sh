# MultiWOZ2.2 Supervised Learning (base)
CUDA_VISIBLE_DEVICES="0,1,2,3" python learn.py --data_path_prefix ../data/multiwoz22 --model_name t5-base --pretrained_path ../checkpoints/base --batch_size 16 --ckpt_save_path ./ckpt/dst22_base --dst --lr 1e-4 --epoch_num 15

# Evaluation: Result will be saved in ckpt/nlg_base
python evaluation.py --data_path_prefix ../data/multiwoz22 --model_name t5-base --pretrained_path ./ckpt/nlg_base --ref_model_path ./ckpt/dst22_base --batch_size 64 --ckpt_save_path ./ckpt/nlg22_base --mode nlg

# MultiWOZ2.2 Reinforcement Learning (base) DST
python reinforce.py --data_path_prefix ../data/multiwoz22 --model_name t5-base --pretrained_path ./ckpt/dst22_base --batch_size 32 --ckpt_save_path ./ckpt/dst22_base_reinforce --lr 1e-5 --mode dst --epoch_num 10 --alpha 1.0

# Evaluation: Result will be saved in ckpt/nlg_base_reinforce
python evaluation.py --data_path_prefix ../data/multiwoz22 --model_name t5-base --pretrained_path ./ckpt/nlg_base_reinforce --batch_size 64 --ref_model_path ./ckpt/dst22_base_reinforce --ref_task dst --mode nlg
