## Character-level Language Modeling

### Overview

In character-level language modeling tasks, each sequence is broken into elements by characters. 
Therefore, in a character-level language model, at each time step the model is expected to predict
the next coming character. We evaluate the temporal convolutional network as a character-level
language model on the PennTreebank dataset and the text8 dataset.

### Data

- **PennTreebank**: When used as a character-level lan-
guage corpus, PTB contains 5,059K characters for training,
396K for validation, and 446K for testing, with an alphabet
size of 50. PennTreebank is a well-studied (but relatively
small) language dataset.

- **text8**: text8 is about 20 times larger than PTB, with 
about 100M characters from Wikipedia (90M for training, 5M 
for validation, and 5M for testing). The corpus contains 27 
unique alphabets.

See `data_generator` in `utils.py`. We download the language corpus using [observations](#) package 
in python.

### Note

- Just like in a recurrent network implementation where it is common to repackage 
hidden units when a new sequence begins, we pass into TCN a sequence `T` consisting 
of two parts: 1) effective history `L1`, and 2) valid sequence `L2`:

```
Sequence [---------T---------] = [--L1-- -----L2-----]
```

In the forward pass, the whole sequence is passed into TCN, but only the `L2` portion is used for 
training. This ensures that the training data are also provided with sufficient history. The size
of `T` and `L2` can be adjusted via flags `seq_len` and `validseqlen`.

- The choice of dataset to use can be specified via the `--dataset` flag. For instance, running

```
python char_cnn_test.py --dataset ptb
```

would (download if no data found, and) train on the PennTreebank (PTB) dataset.

- Empirically, we found that Adam works better than SGD on the text8 dataset.

# Runs

```
python char_cnn_test.py --dataset quora --levels 3 --ksize 5 --nhid 600

| End of epoch   6 | test loss 1.083 | test bpc    1.563
```

```
python char_cnn_test.py --dataset quora --levels 3 --ksize 5 --nhid 500 --optim='Adam' --lr 2e-3

| End of epoch  51 | valid loss 1.057 | valid bpc    1.525
| End of epoch  51 | test loss 1.058 | test bpc    1.527
```

```
python char_cnn_test.py --dataset quora --levels 3 --ksize 5 --nhid 700 --optim='Adam' --lr 2e-3

| End of epoch  33 | valid loss 1.035 | valid bpc    1.493
| End of epoch  33 | test loss 1.036 | test bpc    1.494
```

```
python char_cnn_test.py --dataset quora --levels 4 --ksize 5 --nhid 1000 --optim='Adam' --lr 2e-3

-----------------------------------------------------------------------------------------
| Epoch  13 | valid aux  loss 1.032 | bpc    1.489
| Epoch  13 | valid main loss 0.148 | bpc    0.214 | F1 0.366
-----------------------------------------------------------------------------------------
| Epoch  13 | test  aux  loss 1.034 | bpc    1.491
| Epoch  13 | test  main loss 0.149 | bpc    0.215 | F1 0.262
-----------------------------------------------------------------------------------------
```

## Jan 7, 2018

```
Train positions: 9632207
Answer positions: 130614
Yes answers: 8085
```

1000 hidden units and 10x boost for main loss:
```
python char_cnn_test.py --dataset quora --levels 4 --ksize 5 --nhid 1000 --optim='Adam' --lr 5e-4 --main 10 --gpu 1
```
This peaks after Epoch 11:
```
-----------------------------------------------------------------------------------------
| epoch  11 | valid aux    loss 1.017 | bpc    1.467
| epoch  11 | valid main   loss 0.002 | scaled 0.019 | comb loss 1.035
| epoch  11 | valid answer loss 0.136 | bpc    0.197 | F1 0.582
-----------------------------------------------------------------------------------------
| epoch  11 | test  aux    loss 1.018 | bpc    1.469
| epoch  11 | test  main   loss 0.002 | scaled 0.019 | comb loss 1.036
| epoch  11 | test  answer loss 0.137 | bpc    0.198 | F1 0.574
-----------------------------------------------------------------------------------------
```
Language model keeps improving after that, but main loss degrades:
```
-----------------------------------------------------------------------------------------
| epoch  24 | valid aux    loss 0.990 | bpc    1.428
| epoch  24 | valid main   loss 0.003 | scaled 0.025 | comb loss 1.015
| epoch  24 | valid answer loss 0.186 | bpc    0.269 | F1 0.573
-----------------------------------------------------------------------------------------
| epoch  24 | test  aux    loss 0.991 | bpc    1.430
| epoch  24 | test  main   loss 0.003 | scaled 0.025 | comb loss 1.017
| epoch  24 | test  answer loss 0.188 | bpc    0.271 | F1 0.566
-----------------------------------------------------------------------------------------
```

Go to 5 levels for bigger receptive field:
```
python char_cnn_test.py --dataset quora --levels 5 --ksize 4 --nhid 1000 --optim='Adam' --lr 1e-3 --main 10 --gpu 0 --seq_len 800 --validseqlen 640

-----------------------------------------------------------------------------------------
| epoch   1 | valid aux    loss 1.189 | bpc    1.715
| epoch   1 | valid main   loss 0.002 | scaled 0.021 | comb loss 1.210
| epoch   1 | valid answer loss 0.152 | bpc    0.220 | F1 0.504
-----------------------------------------------------------------------------------------
| epoch   1 | test  aux    loss 1.190 | bpc    1.716
| epoch   1 | test  main   loss 0.002 | scaled 0.021 | comb loss 1.210
| epoch   1 | test  answer loss 0.153 | bpc    0.220 | F1 0.497
-----------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------
| epoch   2 | valid aux    loss 1.113 | bpc    1.605
| epoch   2 | valid main   loss 0.002 | scaled 0.018 | comb loss 1.131
| epoch   2 | valid answer loss 0.134 | bpc    0.193 | F1 0.560
-----------------------------------------------------------------------------------------
| epoch   2 | test  aux    loss 1.114 | bpc    1.608
| epoch   2 | test  main   loss 0.002 | scaled 0.018 | comb loss 1.132
| epoch   2 | test  answer loss 0.134 | bpc    0.194 | F1 0.556
-----------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------
| epoch   3 | valid aux    loss 1.083 | bpc    1.562
| epoch   3 | valid main   loss 0.002 | scaled 0.017 | comb loss 1.100
| epoch   3 | valid answer loss 0.128 | bpc    0.185 | F1 0.582
-----------------------------------------------------------------------------------------
| epoch   3 | test  aux    loss 1.084 | bpc    1.564
| epoch   3 | test  main   loss 0.002 | scaled 0.017 | comb loss 1.102
| epoch   3 | test  answer loss 0.129 | bpc    0.186 | F1 0.571
-----------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------
| epoch   4 | valid aux    loss 1.064 | bpc    1.536
| epoch   4 | valid main   loss 0.002 | scaled 0.017 | comb loss 1.082
| epoch   4 | valid answer loss 0.128 | bpc    0.185 | F1 0.593
-----------------------------------------------------------------------------------------
| epoch   4 | test  aux    loss 1.066 | bpc    1.538
| epoch   4 | test  main   loss 0.002 | scaled 0.017 | comb loss 1.083
| epoch   4 | test  answer loss 0.129 | bpc    0.186 | F1 0.583
-----------------------------------------------------------------------------------------
```
Peaks after Epoch 9:
```
-----------------------------------------------------------------------------------------
| epoch   9 | valid aux    loss 1.027 | bpc    1.482
| epoch   9 | valid main   loss 0.002 | scaled 0.017 | comb loss 1.044
| epoch   9 | valid answer loss 0.126 | bpc    0.181 | F1 0.610
-----------------------------------------------------------------------------------------
| epoch   9 | test  aux    loss 1.029 | bpc    1.484
| epoch   9 | test  main   loss 0.002 | scaled 0.017 | comb loss 1.046
| epoch   9 | test  answer loss 0.126 | bpc    0.182 | F1 0.604
-----------------------------------------------------------------------------------------
```
Language model keeps improving after that:
```
-----------------------------------------------------------------------------------------
| epoch  14 | valid aux    loss 1.012 | bpc    1.461
| epoch  14 | valid main   loss 0.002 | scaled 0.019 | comb loss 1.031
| epoch  14 | valid answer loss 0.136 | bpc    0.197 | F1 0.602
-----------------------------------------------------------------------------------------
| epoch  14 | test  aux    loss 1.014 | bpc    1.463
| epoch  14 | test  main   loss 0.002 | scaled 0.019 | comb loss 1.033
| epoch  14 | test  answer loss 0.137 | bpc    0.197 | F1 0.595
-----------------------------------------------------------------------------------------
```

## Jan 8, 2018

Try main loss boost of 5 rather than 10:
```
python char_cnn_test.py --dataset quora --levels 5 --ksize 4 --nhid 1000 --optim='Adam' --lr 1e-3 --main 5 --gpu 1 --seq_len 800 --validseqlen 640
```
F1 score meanders before peaking:
```
-----------------------------------------------------------------------------------------
| epoch  27 | valid aux    loss 0.989 | bpc    1.427
| epoch  27 | valid main   loss 0.002 | scaled 0.010 | comb loss 0.999
| epoch  27 | valid answer loss 0.142 | bpc    0.205 | F1 0.573
-----------------------------------------------------------------------------------------
| epoch  27 | test  aux    loss 0.991 | bpc    1.430
| epoch  27 | test  main   loss 0.002 | scaled 0.010 | comb loss 1.001
| epoch  27 | test  answer loss 0.144 | bpc    0.207 | F1 0.561
-----------------------------------------------------------------------------------------
```
Thereafter it is supplanted by the language model:
```
-----------------------------------------------------------------------------------------
| epoch  31 | valid aux    loss 0.986 | bpc    1.423
| epoch  31 | valid main   loss 0.002 | scaled 0.010 | comb loss 0.996
| epoch  31 | valid answer loss 0.150 | bpc    0.216 | F1 0.567
-----------------------------------------------------------------------------------------
| epoch  31 | test  aux    loss 0.988 | bpc    1.425
| epoch  31 | test  main   loss 0.002 | scaled 0.010 | comb loss 0.998
| epoch  31 | test  answer loss 0.151 | bpc    0.217 | F1 0.560
-----------------------------------------------------------------------------------------
```

Try main loss boost of 2 rather than 10:
```
python char_cnn_test.py --dataset quora --levels 5 --ksize 4 --nhid 1000 --optim='Adam' --lr 1e-3 --main 2 --gpu 0 --seq_len 800 --validseqlen 640
```
This got as far as:
```
-----------------------------------------------------------------------------------------
| epoch  12 | valid aux    loss 1.012 | bpc    1.461
| epoch  12 | valid main   loss 0.002 | scaled 0.004 | comb loss 1.016
| epoch  12 | valid answer loss 0.137 | bpc    0.198 | F1 0.542
-----------------------------------------------------------------------------------------
| epoch  12 | test  aux    loss 1.014 | bpc    1.463
| epoch  12 | test  main   loss 0.002 | scaled 0.004 | comb loss 1.018
| epoch  12 | test  answer loss 0.139 | bpc    0.201 | F1 0.528
-----------------------------------------------------------------------------------------
```
I killed this because convergence was just too slow.

Try shuffling Quora questions first:
```
python char_cnn_test.py --dataset quora-large --levels 5 --ksize 4 --nhid 1000 --optim='Adam' --lr 1e-3 --main 10 --gpu 0 --seq_len 800 --validseqlen 640

-----------------------------------------------------------------------------------------
| epoch   1 | valid aux    loss 1.191 | bpc    1.719
| epoch   1 | valid main   loss 0.002 | scaled 0.021 | comb loss 1.212
| epoch   1 | valid answer loss 0.154 | bpc    0.221 | F1 0.516
-----------------------------------------------------------------------------------------
| epoch   1 | test  aux    loss 1.189 | bpc    1.716
| epoch   1 | test  main   loss 0.002 | scaled 0.021 | comb loss 1.210
| epoch   1 | test  answer loss 0.155 | bpc    0.224 | F1 0.519
-----------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------
| epoch   2 | valid aux    loss 1.115 | bpc    1.609
| epoch   2 | valid main   loss 0.002 | scaled 0.018 | comb loss 1.133
| epoch   2 | valid answer loss 0.132 | bpc    0.190 | F1 0.580
-----------------------------------------------------------------------------------------
| epoch   2 | test  aux    loss 1.113 | bpc    1.606
| epoch   2 | test  main   loss 0.002 | scaled 0.018 | comb loss 1.131
| epoch   2 | test  answer loss 0.133 | bpc    0.192 | F1 0.578
-----------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------
| epoch   3 | valid aux    loss 1.082 | bpc    1.562
| epoch   3 | valid main   loss 0.002 | scaled 0.017 | comb loss 1.100
| epoch   3 | valid answer loss 0.128 | bpc    0.184 | F1 0.598
-----------------------------------------------------------------------------------------
| epoch   3 | test  aux    loss 1.081 | bpc    1.559
| epoch   3 | test  main   loss 0.002 | scaled 0.017 | comb loss 1.098
| epoch   3 | test  answer loss 0.128 | bpc    0.185 | F1 0.597
-----------------------------------------------------------------------------------------
```
Peaks after Epoch 8:
```
-----------------------------------------------------------------------------------------
| epoch   8 | valid aux    loss 1.032 | bpc    1.489
| epoch   8 | valid main   loss 0.002 | scaled 0.017 | comb loss 1.049
| epoch   8 | valid answer loss 0.122 | bpc    0.176 | F1 0.615
-----------------------------------------------------------------------------------------
| epoch   8 | test  aux    loss 1.030 | bpc    1.486
| epoch   8 | test  main   loss 0.002 | scaled 0.017 | comb loss 1.047
| epoch   8 | test  answer loss 0.123 | bpc    0.177 | F1 0.613
-----------------------------------------------------------------------------------------
```
Language model keeps improving after that:
```
-----------------------------------------------------------------------------------------
| epoch  13 | valid aux    loss 1.015 | bpc    1.464
| epoch  13 | valid main   loss 0.002 | scaled 0.017 | comb loss 1.032
| epoch  13 | valid answer loss 0.127 | bpc    0.183 | F1 0.602
-----------------------------------------------------------------------------------------
| epoch  13 | test  aux    loss 1.013 | bpc    1.462
| epoch  13 | test  main   loss 0.002 | scaled 0.017 | comb loss 1.031
| epoch  13 | test  answer loss 0.127 | bpc    0.184 | F1 0.606
-----------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------
| epoch  14 | valid aux    loss 1.013 | bpc    1.462
| epoch  14 | valid main   loss 0.002 | scaled 0.018 | comb loss 1.031
| epoch  14 | valid answer loss 0.131 | bpc    0.189 | F1 0.597
-----------------------------------------------------------------------------------------
| epoch  14 | test  aux    loss 1.011 | bpc    1.459
| epoch  14 | test  main   loss 0.002 | scaled 0.018 | comb loss 1.029
| epoch  14 | test  answer loss 0.132 | bpc    0.190 | F1 0.594
-----------------------------------------------------------------------------------------
```

## Jan 9, 2018

Try boosting main loss more, but no dice.
```
python char_cnn_test.py --dataset quora-large --levels 5 --ksize 4 --nhid 1000 --optim='Adam' --lr 1e-3 --main 20 --gpu 0 --seq_len 800 --validseqlen 640
```
Peaks fast:
```
-----------------------------------------------------------------------------------------
| epoch   6 | valid aux    loss 1.055 | bpc    1.522
| epoch   6 | valid main   loss 0.002 | scaled 0.034 | comb loss 1.089
| epoch   6 | valid answer loss 0.127 | bpc    0.183 | F1 0.615
-----------------------------------------------------------------------------------------
| epoch   6 | test  aux    loss 1.056 | bpc    1.524
| epoch   6 | test  main   loss 0.002 | scaled 0.034 | comb loss 1.090
| epoch   6 | test  answer loss 0.126 | bpc    0.181 | F1 0.607
-----------------------------------------------------------------------------------------
```
Then declines fast:
```
-----------------------------------------------------------------------------------------
| epoch   9 | valid aux    loss 1.040 | bpc    1.501
| epoch   9 | valid main   loss 0.002 | scaled 0.036 | comb loss 1.076
| epoch   9 | valid answer loss 0.133 | bpc    0.191 | F1 0.600
-----------------------------------------------------------------------------------------
| epoch   9 | test  aux    loss 1.042 | bpc    1.503
| epoch   9 | test  main   loss 0.002 | scaled 0.036 | comb loss 1.078
| epoch   9 | test  answer loss 0.132 | bpc    0.191 | F1 0.593
-----------------------------------------------------------------------------------------
```

Try another architecture, where the label is a constant-value sequence parallel to the input.
```
python train.py --dataset quora-large --levels 5 --ksize 4 --nhid 1000 --optim='Adam' --lr 1e-3 --main 1 --gpu 1 --seq_len 800 --validseqlen 640

-----------------------------------------------------------------------------------------
| epoch   1 | valid aux    loss 1.280 | bpc    1.847
| epoch   1 | valid main   loss 0.201 | scaled 0.201 | comb loss 1.481
| epoch   1 | valid answer loss 0.149 | bpc    0.215 | F1 0.535
-----------------------------------------------------------------------------------------
| epoch   1 | test  aux    loss 1.282 | bpc    1.849
| epoch   1 | test  main   loss 0.200 | scaled 0.200 | comb loss 1.482
| epoch   1 | test  answer loss 0.148 | bpc    0.213 | F1 0.527
-----------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------
| epoch   2 | valid aux    loss 1.189 | bpc    1.716
| epoch   2 | valid main   loss 0.187 | scaled 0.187 | comb loss 1.376
| epoch   2 | valid answer loss 0.132 | bpc    0.191 | F1 0.582
-----------------------------------------------------------------------------------------
| epoch   2 | test  aux    loss 1.191 | bpc    1.718
| epoch   2 | test  main   loss 0.186 | scaled 0.186 | comb loss 1.377
| epoch   2 | test  answer loss 0.131 | bpc    0.189 | F1 0.567
-----------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------
| epoch   3 | valid aux    loss 1.152 | bpc    1.663
| epoch   3 | valid main   loss 0.187 | scaled 0.187 | comb loss 1.339
| epoch   3 | valid answer loss 0.132 | bpc    0.190 | F1 0.597
-----------------------------------------------------------------------------------------
| epoch   3 | test  aux    loss 1.154 | bpc    1.665
| epoch   3 | test  main   loss 0.185 | scaled 0.185 | comb loss 1.339
| epoch   3 | test  answer loss 0.129 | bpc    0.187 | F1 0.586
-----------------------------------------------------------------------------------------
```
Peaks, I think:
```
-----------------------------------------------------------------------------------------
| epoch  12 | valid aux    loss 1.074 | bpc    1.550
| epoch  12 | valid main   loss 0.187 | scaled 0.187 | comb loss 1.261
| epoch  12 | valid answer loss 0.125 | bpc    0.181 | F1 0.625
-----------------------------------------------------------------------------------------
| epoch  12 | test  aux    loss 1.076 | bpc    1.553
| epoch  12 | test  main   loss 0.186 | scaled 0.186 | comb loss 1.262
| epoch  12 | test  answer loss 0.123 | bpc    0.177 | F1 0.623
-----------------------------------------------------------------------------------------
```
Then degrades slowly:
```
-----------------------------------------------------------------------------------------
| epoch  16 | valid aux    loss 1.066 | bpc    1.538
| epoch  16 | valid main   loss 0.185 | scaled 0.185 | comb loss 1.251
| epoch  16 | valid answer loss 0.123 | bpc    0.178 | F1 0.624
-----------------------------------------------------------------------------------------
| epoch  16 | test  aux    loss 1.068 | bpc    1.541
| epoch  16 | test  main   loss 0.184 | scaled 0.184 | comb loss 1.252
| epoch  16 | test  answer loss 0.121 | bpc    0.175 | F1 0.620
-----------------------------------------------------------------------------------------
```
I didn't let it go to long because it became obvious that the priming sequence had to be longer.

Here I try a longer sequence, but still without understanding exactly how to compute the
receptive field length:
```
python train.py --dataset quora-large --levels 6 --ksize 3 --nhid 1000 --optim='Adam' --lr 1e-3 --main 1 --gpu 0 --seq_len 1600 --validseqlen 1200 --batch_size 16

-----------------------------------------------------------------------------------------
| epoch   1 | valid aux    loss 1.268 | bpc    1.829
| epoch   1 | valid main   loss 0.195 | scaled 0.195 | comb loss 1.463
| epoch   1 | valid answer loss 0.141 | bpc    0.203 | F1 0.552
-----------------------------------------------------------------------------------------
| epoch   1 | test  aux    loss 1.270 | bpc    1.832
| epoch   1 | test  main   loss 0.193 | scaled 0.193 | comb loss 1.463
| epoch   1 | test  answer loss 0.139 | bpc    0.201 | F1 0.545
-----------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------
| epoch   2 | valid aux    loss 1.188 | bpc    1.714
| epoch   2 | valid main   loss 0.186 | scaled 0.186 | comb loss 1.374
| epoch   2 | valid answer loss 0.130 | bpc    0.188 | F1 0.591
-----------------------------------------------------------------------------------------
| epoch   2 | test  aux    loss 1.190 | bpc    1.716
| epoch   2 | test  main   loss 0.185 | scaled 0.185 | comb loss 1.375
| epoch   2 | test  answer loss 0.129 | bpc    0.186 | F1 0.578
-----------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------
| epoch   3 | valid aux    loss 1.152 | bpc    1.661
| epoch   3 | valid main   loss 0.182 | scaled 0.182 | comb loss 1.334
| epoch   3 | valid answer loss 0.125 | bpc    0.180 | F1 0.609
-----------------------------------------------------------------------------------------
| epoch   3 | test  aux    loss 1.153 | bpc    1.664
| epoch   3 | test  main   loss 0.181 | scaled 0.181 | comb loss 1.334
| epoch   3 | test  answer loss 0.124 | bpc    0.178 | F1 0.594
-----------------------------------------------------------------------------------------
```
Peaks:
```
-----------------------------------------------------------------------------------------
| epoch  13 | valid aux    loss 1.068 | bpc    1.541
| epoch  13 | valid main   loss 0.176 | scaled 0.176 | comb loss 1.245
| epoch  13 | valid answer loss 0.114 | bpc    0.165 | F1 0.638
-----------------------------------------------------------------------------------------
| epoch  13 | test  aux    loss 1.070 | bpc    1.544
| epoch  13 | test  main   loss 0.176 | scaled 0.176 | comb loss 1.245
| epoch  13 | test  answer loss 0.113 | bpc    0.163 | F1 0.631
-----------------------------------------------------------------------------------------
```
[Still running]

If `n` is the number of layers and `k` is the filter size, then receptive field size is
`1 + 2 * (k - 1) * (2^(n + 1) - 1)`.

After figuring this out, I make sure the priming sequence was larger than the receptive field
(for the first time) and also tried a new dropout rate and main loss boost.
```
python train.py --dataset quora-large --levels 5 --ksize 3 --nhid 1000 --optim='Adam' --lr 1e-3 --main 2 --gpu 1 --seq_len 1600 --validseqlen 1340 --dropout 0.15 --batch_size 16

-----------------------------------------------------------------------------------------
| epoch   1 | valid aux    loss 1.328 | bpc    1.916
| epoch   1 | valid main   loss 0.196 | scaled 0.393 | comb loss 1.721
| epoch   1 | valid answer loss 0.142 | bpc    0.205 | F1 0.544
-----------------------------------------------------------------------------------------
| epoch   1 | test  aux    loss 1.329 | bpc    1.918
| epoch   1 | test  main   loss 0.196 | scaled 0.391 | comb loss 1.720
| epoch   1 | test  answer loss 0.141 | bpc    0.203 | F1 0.536
-----------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------
| epoch   2 | valid aux    loss 1.228 | bpc    1.772
| epoch   2 | valid main   loss 0.188 | scaled 0.376 | comb loss 1.605
| epoch   2 | valid answer loss 0.132 | bpc    0.190 | F1 0.577
-----------------------------------------------------------------------------------------
| epoch   2 | test  aux    loss 1.230 | bpc    1.774
| epoch   2 | test  main   loss 0.187 | scaled 0.373 | comb loss 1.603
| epoch   2 | test  answer loss 0.130 | bpc    0.188 | F1 0.567
-----------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------
| epoch   3 | valid aux    loss 1.187 | bpc    1.712
| epoch   3 | valid main   loss 0.187 | scaled 0.374 | comb loss 1.560
| epoch   3 | valid answer loss 0.128 | bpc    0.185 | F1 0.589
-----------------------------------------------------------------------------------------
| epoch   3 | test  aux    loss 1.188 | bpc    1.714
| epoch   3 | test  main   loss 0.185 | scaled 0.370 | comb loss 1.558
| epoch   3 | test  answer loss 0.127 | bpc    0.183 | F1 0.581
-----------------------------------------------------------------------------------------
```
I let it go as far as:
```
-----------------------------------------------------------------------------------------
| epoch  11 | valid aux    loss 1.110 | bpc    1.601
| epoch  11 | valid main   loss 0.181 | scaled 0.362 | comb loss 1.472
| epoch  11 | valid answer loss 0.120 | bpc    0.173 | F1 0.619
-----------------------------------------------------------------------------------------
| epoch  11 | test  aux    loss 1.112 | bpc    1.604
| epoch  11 | test  main   loss 0.180 | scaled 0.360 | comb loss 1.472
| epoch  11 | test  answer loss 0.119 | bpc    0.171 | F1 0.612
-----------------------------------------------------------------------------------------
```
Then I killed it because it's obvious that the dropout rate was too high.

## Jan 10

I tried the same run with lower dropout:
```
python train.py --dataset quora-large --levels 5 --ksize 3 --nhid 1000 --optim='Adam' --lr 1e-3 --main 2 --gpu 1 --seq_len 1600 --validseqlen 1340 --dropout 0.1 --batch_size 16
```
[Still running]

