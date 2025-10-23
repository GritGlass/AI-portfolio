#train
python ./src/run.py --input ./data --test_id 7 --early_stop 30 --epoch 1000 --test_cycle 20 --run_type train --output ./result --try_id 1 --resume checkpoint_file.ckpt

#fine tuning
# python ./src/run.py --input ./data --test_id 7 --early_stop 30 --epoch 1000 --test_cycle 20 --run_type valid --output ./result --try_id 1 --model_weights checkpoint_file.ckpt

#test
# python ./src/run.py --input ./data --test_id 7 --test_cycle 20 --run_type test --output ./result --try_id 1 --model_weights checkpoint_file.ckpt
