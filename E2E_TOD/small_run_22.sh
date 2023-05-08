# MultiWOZ2.2 Supervised Learning (small)
CUDA_VISIBLE_DEVICES="0,1,2,3" python learn.py --data_path_prefix ../data/multiwoz22 --model_name t5-small --pretrained_path ../checkpoints/small --batch_size 16 --ckpt_save_path ./ckpt/dst22_small --dst --lr 1e-4 --epoch_num 15

# Evaluation: Result will be saved in ckpt/nlg_small
python evaluation.py --data_path_prefix ../data/multiwoz22 --model_name t5-small --pretrained_path ./ckpt/nlg_small --batch_size 64 --ref_model_path ./ckpt/dst22_small --ref_task dst --mode nlg

# MultiWOZ2.2 Reinforcement Learning (small) DST
python reinforce.py --data_path_prefix ../data/multiwoz22 --model_name t5-small --pretrained_path ./ckpt/dst22_small --batch_size 32 --ckpt_save_path ./ckpt/dst22_small_reinforce --lr 1e-5 --mode dst --epoch_num 10

# Evaluation: Result will be saved in ckpt/nlg_small_reinforce
python evaluation.py --data_path_prefix ../data/multiwoz22 --model_name t5-small --pretrained_path ./ckpt/nlg_small_reinforce --batch_size 64 --ref_model_path ./ckpt/dst22_small_reinforcem --ref_task dst --mode nlg