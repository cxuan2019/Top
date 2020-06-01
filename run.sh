#!/bin/bash

#'lotv' 'compact' 'sketch'
#
#for VAR in 'compact' 'sketch'
#do
##  python run.py experiement --train-data=train_$VAR.tsv --dev-data=eval_$VAR.tsv --test-data=test_$VAR.tsv --max-epoch=50  --save-to=model_$VAR.bin --seed=4 --dropout=0.5 --cuda
#  python run.py test --train-data=train_$VAR.tsv --dev-data=eval_$VAR.tsv --test-data=test_$VAR.tsv --save-to=model_$VAR.bin --cuda
#done

#python run.py experiement --train-data=train_lotv.tsv --dev-data=eval_lotv.tsv --test-data=test_lotv.tsv --max-epoch=50  --save-to=model_lotv.bin --seed=0 --dropout=0.3 --cuda
#python run.py test --train-data=train_lotv.tsv --dev-data=eval_lotv.tsv --test-data=test_lotv.tsv --save-to=model_lotv.bin --cuda

#python run.py experiement --train-data=train_compact.tsv --dev-data=eval_compact.tsv --test-data=test_compact.tsv --max-epoch=50  --save-to=model_compact_copy.bin --seed=4 --dropout=0.5 --use-copy --cuda
python run.py test --train-data=train_compact.tsv --dev-data=eval_compact.tsv --test-data=test_compact.tsv --save-to=model_compact_copy.bin --use-copy --cuda

#python run.py experiement --train-data=train_compact.tsv --dev-data=eval_compact.tsv --test-data=test_compact.tsv --max-epoch=50  --save-to=model_compact_pos.bin --seed=4 --dropout=0.5 --use-pos-embed --cuda
#python run.py test --train-data=train_compact.tsv --dev-data=eval_compact.tsv --test-data=test_compact.tsv --save-to=model_compact_pos.bin --use-pos-embed --cuda

## for hyperparameter tuning
#for VAR in '1' '2' '3' '4' '5'
#do
#  python run.py experiement --train-data=train_compact.tsv --dev-data=eval_compact.tsv --test-data=test_compact.tsv --max-epoch=50  --save-to=model_compact_copy_$VAR.bin --seed=$VAR --use-copy --cuda
##  python run.py experiement --train-data=train_compact.tsv --dev-data=eval_compact.tsv --test-data=test_compact.tsv --max-epoch=50  --save-to=model_compact_pos_$VAR.bin --seed=$VAR --use-pos-embed --cuda
##  python run.py experiement --train-data=train_compact.tsv --dev-data=eval_compact.tsv --test-data=test_compact.tsv --max-epoch=50  --save-to=model_compact_$VAR.bin --seed=$VAR --cuda
###  python run.py test --train-data=train_$VAR.tsv --dev-data=eval_$VAR.tsv --test-data=test_$VAR.tsv --save-to=model_$VAR.bin --cuda
#done

#for VAR in 48
#do
##  for VAR2 in '0.5'
##  do
#  python run.py experiement --train-data=train_compact.tsv --dev-data=eval_compact.tsv --test-data=test_compact.tsv --max-epoch=50  --save-to=model_compact_copy_"$VAR".bin --batch-size="$VAR" --seed=4 --dropout=0.5 --use-copy --cuda
##  python run.py experiement --train-data=train_compact.tsv --dev-data=eval_compact.tsv --test-data=test_compact.tsv --max-epoch=50  --save-to=model_compact_pos_$VAR.bin --seed=$VAR --use-pos-embed --cuda
##  python run.py experiement --train-data=train_compact.tsv --dev-data=eval_compact.tsv --test-data=test_compact.tsv --max-epoch=50  --save-to=model_compact_$VAR.bin --seed=$VAR --cuda
###  python run.py test --train-data=train_$VAR.tsv --dev-data=eval_$VAR.tsv --test-data=test_$VAR.tsv --save-to=model_$VAR.bin --cuda
##  done
#done



# --cuda