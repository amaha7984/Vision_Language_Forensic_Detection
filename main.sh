python main.py \
    --model 'LASTED' \
    --train_file 'annotation/Train.txt' \
    --val_ratio 0.10\
    --test_file 'annotation/Test.txt' \
    --isTrain 1 \
    --lr 0.0001 \
    --data_size 448 \
    --batch_size 48 \
    --gpu '0,1,2,3' \
    2>&1 | tee weights/log.log