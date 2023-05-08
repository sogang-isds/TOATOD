python learn.py\
    --data_prefix ../data/banking77\
    --datapoints_per_intent 10000\
    --num_train_epochs 150\
    --model_name t5-base\
    --pretrained_path ../E2E_TOD/checkpoints/base\
    --batch_size_per_gpu 64\
    --number_of_gpu 1\
    --lr 0.15\
    --save_path model_save/base/full_training_0.15

python learn.py\
    --data_prefix ../data/banking77\
    --datapoints_per_intent 10000\
    --num_train_epochs 150\
    --model_name t5-small\
    --pretrained_path ../E2E_TOD/checkpoints/small\
    --batch_size_per_gpu 64\
    --number_of_gpu 1\
    --lr 0.001\
    --save_path model_save/small/full_training_0.001

python learn.py\
    --data_prefix ../data/clinc150\
    --datapoints_per_intent 10000\
    --num_train_epochs 150\
    --model_name t5-base\
    --pretrained_path ../E2E_TOD/checkpoints/base\
    --batch_size_per_gpu 64\
    --number_of_gpu 1\
    --lr 0.15\
    --save_path model_save/base/full_training_0.15/ \

python learn.py\
    --data_prefix ../data/clinc150\
    --datapoints_per_intent 10000\
    --num_train_epochs 150\
    --model_name t5-small\
    --pretrained_path ../E2E_TOD/checkpoints/small\
    --batch_size_per_gpu 64\
    --number_of_gpu 1\
    --lr 0.001\
    --save_path model_save/small/full_training_0.001/ \

python learn.py\
    --data_prefix ../data/hwu64\
    --datapoints_per_intent 10000\
    --num_train_epochs 150\
    --model_name t5-base\
    --pretrained_path ../E2E_TOD/checkpoints/base\
    --batch_size_per_gpu 64\
    --number_of_gpu 1\
    --lr 0.1\
    --save_path model_save/base/full_training_0.1/ \

python learn.py\
    --data_prefix ../data/hwu64\
    --datapoints_per_intent 10000\
    --num_train_epochs 150\
    --model_name t5-small\
    --pretrained_path ../E2E_TOD/checkpoints/small\
    --batch_size_per_gpu 64\
    --number_of_gpu 1\
    --lr 0.01\
    --save_path model_save/small/full_training_0.01/ \