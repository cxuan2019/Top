#!/bin/bash
# python run.py experiement [opts]: both training and testing
# python run.py test [opts]: testing only

# for 'compact' 'sketch' reprenseation (with vanilla seq2seq)
#for VAR in 'compact' 'sketch'
#do
##  python run.py experiement --train-data=train_$VAR.tsv --dev-data=eval_$VAR.tsv --test-data=test_$VAR.tsv --max-epoch=50  --save-to=model_$VAR.bin --seed=4 --dropout=0.5 --cuda
#  python run.py test --train-data=train_$VAR.tsv --dev-data=eval_$VAR.tsv --test-data=test_$VAR.tsv --save-to=model_$VAR.bin --cuda
#done

# for 'lotv' represenation (with vanilla seq2seq)
#python run.py experiement --train-data=train_lotv.tsv --dev-data=eval_lotv.tsv --test-data=test_lotv.tsv --max-epoch=50  --save-to=model_lotv.bin --seed=0 --dropout=0.3 --cuda
#python run.py test --train-data=train_lotv.tsv --dev-data=eval_lotv.tsv --test-data=test_lotv.tsv --save-to=model_lotv.bin --cuda

# for 'compact' represenation (with copy-augmented seq2seq)
#python run.py experiement --train-data=train_compact.tsv --dev-data=eval_compact.tsv --test-data=test_compact.tsv --max-epoch=50  --save-to=model_compact_copy.bin --seed=4 --dropout=0.5 --use-copy --cuda
#python run.py test --train-data=train_compact.tsv --dev-data=eval_compact.tsv --test-data=test_compact.tsv --save-to=model_compact_copy.bin --use-copy --cuda
