# MultiWOZ2.1 Supervised Learning (small)
CUDA_VISIBLE_DEVICES="0,1,2,3" python learn.py --data_path_prefix ../data/multwoz21 --model_name t5-small --pretrained_path ../checkpoints/small --batch_size 16 --ckpt_save_path ./ckpt/dst_small --dst --lr 1e-4 --epoch_num 15
CUDA_VISIBLE_DEVICES="0,1,2,3" python learn.py --data_path_prefix ../data/multwoz21 --model_name t5-small --pretrained_path ../checkpoints/small --weight_path ./ckpt/dst_small --batch_size 16 --ckpt_save_path ./ckpt/nlg_small --nlg --lr 1e-4 --epoch_num 15

# Evaluation: Result will be saved in ckpt/nlg_small
python evaluation.py --data_path_prefix ../data/multwoz21 --model_name t5-small --pretrained_path ./ckpt/nlg_small --batch_size 64 --ckpt_save_path ./ckpt/nlg_small --mode nlg

# MultiWOZ2.1 Reinforcement Learning (small) DST
python reinforce.py --data_path_prefix ../data/multwoz21 --model_name t5-small --pretrained_path ./ckpt/dst_small --batch_size 32 --ckpt_save_path ./ckpt/dst_small_reinforce --lr 1e-5 --mode dst --epoch_num 10

# MultiWOZ2.1 Reinforcement Learning (small) NLG
python reinforce.py --data_path_prefix ../data/multwoz21 --model_name t5-small --pretrained_path ./ckpt/nlg_small --batch_size 4 --ckpt_save_path ./ckpt/nlg_small_reinforce --lr 1e-5 --mode nlg --epoch_num 3

# Evaluation: Result will be saved in ckpt/nlg_small_reinforce
python evaluation.py --data_path_prefix ../data/multwoz21 --model_name t5-small --pretrained_path ./ckpt/nlg_small_reinforce --batch_size 64 --ckpt_save_path ./ckpt/nlg_small_reinforce --mode nlg