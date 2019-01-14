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
receptive field size:
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
I stopped it here:
```
| epoch  20 |   100/ 3900 batches | lr 1.0e-03 | ms/batch 742.28 | aux 1.108 | main 0.163 | loss 1.271
| epoch  20 |   200/ 3900 batches | lr 1.0e-03 | ms/batch 735.91 | aux 1.096 | main 0.160 | loss 1.256
| epoch  20 |   300/ 3900 batches | lr 1.0e-03 | ms/batch 733.75 | aux 1.099 | main 0.153 | loss 1.252
| epoch  20 |   400/ 3900 batches | lr 1.0e-03 | ms/batch 734.90 | aux 1.094 | main 0.163 | loss 1.257
| epoch  20 |   500/ 3900 batches | lr 1.0e-03 | ms/batch 733.30 | aux 1.095 | main 0.158 | loss 1.253
| epoch  20 |   600/ 3900 batches | lr 1.0e-03 | ms/batch 732.09 | aux 1.098 | main 0.172 | loss 1.270
| epoch  20 |   700/ 3900 batches | lr 1.0e-03 | ms/batch 734.82 | aux 1.098 | main 0.162 | loss 1.260
| epoch  20 |   800/ 3900 batches | lr 1.0e-03 | ms/batch 734.60 | aux 1.095 | main 0.163 | loss 1.259
| epoch  20 |   900/ 3900 batches | lr 1.0e-03 | ms/batch 732.97 | aux 1.095 | main 0.166 | loss 1.261
| epoch  20 |  1000/ 3900 batches | lr 1.0e-03 | ms/batch 732.92 | aux 1.097 | main 0.163 | loss 1.260
| epoch  20 |  1100/ 3900 batches | lr 1.0e-03 | ms/batch 736.60 | aux 1.097 | main 0.163 | loss 1.260
| epoch  20 |  1200/ 3900 batches | lr 1.0e-03 | ms/batch 733.08 | aux 1.096 | main 0.161 | loss 1.257
| epoch  20 |  1300/ 3900 batches | lr 1.0e-03 | ms/batch 734.25 | aux 1.096 | main 0.157 | loss 1.253
| epoch  20 |  1400/ 3900 batches | lr 1.0e-03 | ms/batch 734.73 | aux 1.097 | main 0.169 | loss 1.266
| epoch  20 |  1500/ 3900 batches | lr 1.0e-03 | ms/batch 733.09 | aux 1.098 | main 0.163 | loss 1.261
| epoch  20 |  1600/ 3900 batches | lr 1.0e-03 | ms/batch 732.42 | aux 1.096 | main 0.158 | loss 1.254
| epoch  20 |  1700/ 3900 batches | lr 1.0e-03 | ms/batch 734.79 | aux 1.096 | main 0.159 | loss 1.256
| epoch  20 |  1800/ 3900 batches | lr 1.0e-03 | ms/batch 733.05 | aux 1.099 | main 0.162 | loss 1.261
| epoch  20 |  1900/ 3900 batches | lr 1.0e-03 | ms/batch 734.08 | aux 1.093 | main 0.169 | loss 1.262
| epoch  20 |  2000/ 3900 batches | lr 1.0e-03 | ms/batch 734.87 | aux 1.089 | main 0.157 | loss 1.247
| epoch  20 |  2100/ 3900 batches | lr 1.0e-03 | ms/batch 733.08 | aux 1.097 | main 0.158 | loss 1.255
| epoch  20 |  2200/ 3900 batches | lr 1.0e-03 | ms/batch 733.49 | aux 1.098 | main 0.164 | loss 1.262
| epoch  20 |  2300/ 3900 batches | lr 1.0e-03 | ms/batch 733.28 | aux 1.093 | main 0.163 | loss 1.256
| epoch  20 |  2400/ 3900 batches | lr 1.0e-03 | ms/batch 736.52 | aux 1.097 | main 0.155 | loss 1.251
| epoch  20 |  2500/ 3900 batches | lr 1.0e-03 | ms/batch 731.98 | aux 1.096 | main 0.160 | loss 1.256
| epoch  20 |  2600/ 3900 batches | lr 1.0e-03 | ms/batch 733.13 | aux 1.098 | main 0.170 | loss 1.268
| epoch  20 |  2700/ 3900 batches | lr 1.0e-03 | ms/batch 733.13 | aux 1.093 | main 0.159 | loss 1.252
| epoch  20 |  2800/ 3900 batches | lr 1.0e-03 | ms/batch 733.09 | aux 1.099 | main 0.162 | loss 1.261
| epoch  20 |  2900/ 3900 batches | lr 1.0e-03 | ms/batch 731.99 | aux 1.094 | main 0.155 | loss 1.249
| epoch  20 |  3000/ 3900 batches | lr 1.0e-03 | ms/batch 732.62 | aux 1.094 | main 0.164 | loss 1.258
| epoch  20 |  3100/ 3900 batches | lr 1.0e-03 | ms/batch 733.58 | aux 1.096 | main 0.162 | loss 1.258
| epoch  20 |  3200/ 3900 batches | lr 1.0e-03 | ms/batch 735.05 | aux 1.092 | main 0.162 | loss 1.255
| epoch  20 |  3300/ 3900 batches | lr 1.0e-03 | ms/batch 733.87 | aux 1.095 | main 0.165 | loss 1.261
| epoch  20 |  3400/ 3900 batches | lr 1.0e-03 | ms/batch 734.23 | aux 1.094 | main 0.154 | loss 1.248
| epoch  20 |  3500/ 3900 batches | lr 1.0e-03 | ms/batch 734.49 | aux 1.095 | main 0.159 | loss 1.254
| epoch  20 |  3600/ 3900 batches | lr 1.0e-03 | ms/batch 733.88 | aux 1.093 | main 0.161 | loss 1.254
| epoch  20 |  3700/ 3900 batches | lr 1.0e-03 | ms/batch 733.95 | aux 1.094 | main 0.163 | loss 1.257
| epoch  20 |  3800/ 3900 batches | lr 1.0e-03 | ms/batch 735.86 | aux 1.098 | main 0.171 | loss 1.269
-----------------------------------------------------------------------------------------
| epoch  20 | valid aux    loss 1.054 | bpc    1.520
| epoch  20 | valid main   loss 0.180 | scaled 0.180 | comb loss 1.234
| epoch  20 | valid answer loss 0.115 | bpc    0.167 | F1 0.635
-----------------------------------------------------------------------------------------
| epoch  20 | test  aux    loss 1.056 | bpc    1.523
| epoch  20 | test  main   loss 0.180 | scaled 0.180 | comb loss 1.236
| epoch  20 | test  answer loss 0.115 | bpc    0.166 | F1 0.627
-----------------------------------------------------------------------------------------
```

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
Peaked:
```
| epoch  10 |   100/ 3492 batches | lr 1.0e-03 | ms/batch 561.60 | aux 1.167 | main 0.172 | loss 1.512
| epoch  10 |   200/ 3492 batches | lr 1.0e-03 | ms/batch 556.12 | aux 1.156 | main 0.171 | loss 1.498
| epoch  10 |   300/ 3492 batches | lr 1.0e-03 | ms/batch 556.07 | aux 1.158 | main 0.169 | loss 1.495
| epoch  10 |   400/ 3492 batches | lr 1.0e-03 | ms/batch 556.10 | aux 1.155 | main 0.172 | loss 1.500
| epoch  10 |   500/ 3492 batches | lr 1.0e-03 | ms/batch 556.21 | aux 1.154 | main 0.177 | loss 1.508
| epoch  10 |   600/ 3492 batches | lr 1.0e-03 | ms/batch 556.09 | aux 1.158 | main 0.176 | loss 1.510
| epoch  10 |   700/ 3492 batches | lr 1.0e-03 | ms/batch 556.00 | aux 1.157 | main 0.175 | loss 1.507
| epoch  10 |   800/ 3492 batches | lr 1.0e-03 | ms/batch 556.16 | aux 1.155 | main 0.172 | loss 1.499
| epoch  10 |   900/ 3492 batches | lr 1.0e-03 | ms/batch 556.21 | aux 1.153 | main 0.174 | loss 1.501
| epoch  10 |  1000/ 3492 batches | lr 1.0e-03 | ms/batch 556.07 | aux 1.157 | main 0.168 | loss 1.493
| epoch  10 |  1100/ 3492 batches | lr 1.0e-03 | ms/batch 556.10 | aux 1.153 | main 0.172 | loss 1.497
| epoch  10 |  1200/ 3492 batches | lr 1.0e-03 | ms/batch 556.15 | aux 1.156 | main 0.175 | loss 1.507
| epoch  10 |  1300/ 3492 batches | lr 1.0e-03 | ms/batch 556.15 | aux 1.156 | main 0.171 | loss 1.498
| epoch  10 |  1400/ 3492 batches | lr 1.0e-03 | ms/batch 556.37 | aux 1.156 | main 0.172 | loss 1.500
| epoch  10 |  1500/ 3492 batches | lr 1.0e-03 | ms/batch 556.43 | aux 1.155 | main 0.169 | loss 1.493
| epoch  10 |  1600/ 3492 batches | lr 1.0e-03 | ms/batch 556.32 | aux 1.157 | main 0.171 | loss 1.500
| epoch  10 |  1700/ 3492 batches | lr 1.0e-03 | ms/batch 556.27 | aux 1.152 | main 0.179 | loss 1.510
| epoch  10 |  1800/ 3492 batches | lr 1.0e-03 | ms/batch 556.23 | aux 1.148 | main 0.169 | loss 1.486
| epoch  10 |  1900/ 3492 batches | lr 1.0e-03 | ms/batch 556.26 | aux 1.155 | main 0.175 | loss 1.504
| epoch  10 |  2000/ 3492 batches | lr 1.0e-03 | ms/batch 556.21 | aux 1.152 | main 0.176 | loss 1.504
| epoch  10 |  2100/ 3492 batches | lr 1.0e-03 | ms/batch 556.04 | aux 1.154 | main 0.168 | loss 1.489
| epoch  10 |  2200/ 3492 batches | lr 1.0e-03 | ms/batch 556.08 | aux 1.154 | main 0.170 | loss 1.494
| epoch  10 |  2300/ 3492 batches | lr 1.0e-03 | ms/batch 555.99 | aux 1.154 | main 0.182 | loss 1.519
| epoch  10 |  2400/ 3492 batches | lr 1.0e-03 | ms/batch 555.86 | aux 1.154 | main 0.171 | loss 1.496
| epoch  10 |  2500/ 3492 batches | lr 1.0e-03 | ms/batch 556.08 | aux 1.156 | main 0.166 | loss 1.488
| epoch  10 |  2600/ 3492 batches | lr 1.0e-03 | ms/batch 556.10 | aux 1.154 | main 0.167 | loss 1.487
| epoch  10 |  2700/ 3492 batches | lr 1.0e-03 | ms/batch 556.22 | aux 1.153 | main 0.171 | loss 1.495
| epoch  10 |  2800/ 3492 batches | lr 1.0e-03 | ms/batch 556.34 | aux 1.153 | main 0.170 | loss 1.492
| epoch  10 |  2900/ 3492 batches | lr 1.0e-03 | ms/batch 556.37 | aux 1.152 | main 0.179 | loss 1.510
| epoch  10 |  3000/ 3492 batches | lr 1.0e-03 | ms/batch 556.15 | aux 1.152 | main 0.169 | loss 1.490
| epoch  10 |  3100/ 3492 batches | lr 1.0e-03 | ms/batch 556.42 | aux 1.152 | main 0.174 | loss 1.500
| epoch  10 |  3200/ 3492 batches | lr 1.0e-03 | ms/batch 556.33 | aux 1.147 | main 0.169 | loss 1.485
| epoch  10 |  3300/ 3492 batches | lr 1.0e-03 | ms/batch 556.40 | aux 1.151 | main 0.172 | loss 1.495
| epoch  10 |  3400/ 3492 batches | lr 1.0e-03 | ms/batch 556.38 | aux 1.156 | main 0.183 | loss 1.521
-----------------------------------------------------------------------------------------
| epoch  10 | valid aux    loss 1.104 | bpc    1.592
| epoch  10 | valid main   loss 0.179 | scaled 0.358 | comb loss 1.462
| epoch  10 | valid answer loss 0.120 | bpc    0.173 | F1 0.618
-----------------------------------------------------------------------------------------
| epoch  10 | test  aux    loss 1.105 | bpc    1.595
| epoch  10 | test  main   loss 0.179 | scaled 0.357 | comb loss 1.463
| epoch  10 | test  answer loss 0.119 | bpc    0.172 | F1 0.610
-----------------------------------------------------------------------------------------
```
I stopped it here.
```
| epoch  13 |   100/ 3492 batches | lr 1.0e-03 | ms/batch 561.00 | aux 1.157 | main 0.167 | loss 1.491
| epoch  13 |   200/ 3492 batches | lr 1.0e-03 | ms/batch 555.69 | aux 1.145 | main 0.167 | loss 1.480
| epoch  13 |   300/ 3492 batches | lr 1.0e-03 | ms/batch 555.66 | aux 1.147 | main 0.163 | loss 1.472
| epoch  13 |   400/ 3492 batches | lr 1.0e-03 | ms/batch 555.52 | aux 1.142 | main 0.168 | loss 1.478
| epoch  13 |   500/ 3492 batches | lr 1.0e-03 | ms/batch 555.49 | aux 1.141 | main 0.172 | loss 1.484
| epoch  13 |   600/ 3492 batches | lr 1.0e-03 | ms/batch 555.30 | aux 1.146 | main 0.175 | loss 1.496
| epoch  13 |   700/ 3492 batches | lr 1.0e-03 | ms/batch 555.23 | aux 1.144 | main 0.169 | loss 1.482
| epoch  13 |   800/ 3492 batches | lr 1.0e-03 | ms/batch 555.27 | aux 1.143 | main 0.166 | loss 1.475
| epoch  13 |   900/ 3492 batches | lr 1.0e-03 | ms/batch 555.36 | aux 1.144 | main 0.170 | loss 1.483
| epoch  13 |  1000/ 3492 batches | lr 1.0e-03 | ms/batch 555.38 | aux 1.146 | main 0.164 | loss 1.473
| epoch  13 |  1100/ 3492 batches | lr 1.0e-03 | ms/batch 555.45 | aux 1.142 | main 0.168 | loss 1.479
| epoch  13 |  1200/ 3492 batches | lr 1.0e-03 | ms/batch 555.67 | aux 1.146 | main 0.172 | loss 1.489
| epoch  13 |  1300/ 3492 batches | lr 1.0e-03 | ms/batch 555.63 | aux 1.146 | main 0.168 | loss 1.482
| epoch  13 |  1400/ 3492 batches | lr 1.0e-03 | ms/batch 555.72 | aux 1.144 | main 0.168 | loss 1.481
| epoch  13 |  1500/ 3492 batches | lr 1.0e-03 | ms/batch 555.49 | aux 1.144 | main 0.164 | loss 1.471
| epoch  13 |  1600/ 3492 batches | lr 1.0e-03 | ms/batch 555.57 | aux 1.147 | main 0.169 | loss 1.484
| epoch  13 |  1700/ 3492 batches | lr 1.0e-03 | ms/batch 555.84 | aux 1.142 | main 0.173 | loss 1.488
| epoch  13 |  1800/ 3492 batches | lr 1.0e-03 | ms/batch 555.68 | aux 1.138 | main 0.164 | loss 1.465
| epoch  13 |  1900/ 3492 batches | lr 1.0e-03 | ms/batch 555.76 | aux 1.146 | main 0.168 | loss 1.482
| epoch  13 |  2000/ 3492 batches | lr 1.0e-03 | ms/batch 555.49 | aux 1.140 | main 0.172 | loss 1.485
| epoch  13 |  2100/ 3492 batches | lr 1.0e-03 | ms/batch 555.49 | aux 1.144 | main 0.170 | loss 1.483
| epoch  13 |  2200/ 3492 batches | lr 1.0e-03 | ms/batch 555.59 | aux 1.142 | main 0.165 | loss 1.473
| epoch  13 |  2300/ 3492 batches | lr 1.0e-03 | ms/batch 555.43 | aux 1.143 | main 0.175 | loss 1.494
| epoch  13 |  2400/ 3492 batches | lr 1.0e-03 | ms/batch 555.36 | aux 1.142 | main 0.166 | loss 1.473
| epoch  13 |  2500/ 3492 batches | lr 1.0e-03 | ms/batch 555.55 | aux 1.144 | main 0.163 | loss 1.470
| epoch  13 |  2600/ 3492 batches | lr 1.0e-03 | ms/batch 555.33 | aux 1.142 | main 0.162 | loss 1.466
| epoch  13 |  2700/ 3492 batches | lr 1.0e-03 | ms/batch 555.42 | aux 1.141 | main 0.166 | loss 1.473
| epoch  13 |  2800/ 3492 batches | lr 1.0e-03 | ms/batch 555.38 | aux 1.142 | main 0.165 | loss 1.472
| epoch  13 |  2900/ 3492 batches | lr 1.0e-03 | ms/batch 555.26 | aux 1.142 | main 0.174 | loss 1.490
| epoch  13 |  3000/ 3492 batches | lr 1.0e-03 | ms/batch 555.39 | aux 1.142 | main 0.164 | loss 1.470
| epoch  13 |  3100/ 3492 batches | lr 1.0e-03 | ms/batch 555.20 | aux 1.142 | main 0.168 | loss 1.478
| epoch  13 |  3200/ 3492 batches | lr 1.0e-03 | ms/batch 555.33 | aux 1.138 | main 0.166 | loss 1.470
| epoch  13 |  3300/ 3492 batches | lr 1.0e-03 | ms/batch 555.41 | aux 1.140 | main 0.166 | loss 1.473
| epoch  13 |  3400/ 3492 batches | lr 1.0e-03 | ms/batch 555.42 | aux 1.145 | main 0.185 | loss 1.514
-----------------------------------------------------------------------------------------
| epoch  13 | valid aux    loss 1.094 | bpc    1.579
| epoch  13 | valid main   loss 0.183 | scaled 0.366 | comb loss 1.460
| epoch  13 | valid answer loss 0.122 | bpc    0.175 | F1 0.619
-----------------------------------------------------------------------------------------
| epoch  13 | test  aux    loss 1.096 | bpc    1.581
| epoch  13 | test  main   loss 0.183 | scaled 0.365 | comb loss 1.461
| epoch  13 | test  answer loss 0.121 | bpc    0.175 | F1 0.608
-----------------------------------------------------------------------------------------
```
The train loss just doesn't get low enough.

What about less dropout?
```
python train.py --dataset quora-large --levels 5 --ksize 3 --nhid 1000 --optim='Adam' --lr 1e-3 --main 1 --gpu 0 --seq_len 1600 --validseqlen 1340 --dropout 0.05 --batch_size 16
```
Also not so great. Peaked:
```
-----------------------------------------------------------------------------------------
| epoch  10 | valid aux    loss 1.073 | bpc    1.548
| epoch  10 | valid main   loss 0.178 | scaled 0.178 | comb loss 1.251
| epoch  10 | valid answer loss 0.119 | bpc    0.172 | F1 0.614
-----------------------------------------------------------------------------------------
| epoch  10 | test  aux    loss 1.075 | bpc    1.551
| epoch  10 | test  main   loss 0.177 | scaled 0.177 | comb loss 1.252
| epoch  10 | test  answer loss 0.118 | bpc    0.170 | F1 0.607
-----------------------------------------------------------------------------------------
```
Then:
```
-----------------------------------------------------------------------------------------
| epoch  11 | valid aux    loss 1.070 | bpc    1.543
| epoch  11 | valid main   loss 0.183 | scaled 0.183 | comb loss 1.253
| epoch  11 | valid answer loss 0.128 | bpc    0.185 | F1 0.610
-----------------------------------------------------------------------------------------
| epoch  11 | test  aux    loss 1.071 | bpc    1.546
| epoch  11 | test  main   loss 0.183 | scaled 0.183 | comb loss 1.254
| epoch  11 | test  answer loss 0.127 | bpc    0.184 | F1 0.604
-----------------------------------------------------------------------------------------
```

Okay how about more hidden states?
```
python train.py --dataset quora-large --levels 5 --ksize 3 --nhid 1400 --optim='Adam' --lr 1e-3 --main 2 --gpu 1 --seq_len 1600 --validseqlen 1340 --dropout 0.1 --batch_size 16

-----------------------------------------------------------------------------------------
| epoch   1 | valid aux    loss 1.300 | bpc    1.876
| epoch   1 | valid main   loss 0.206 | scaled 0.413 | comb loss 1.713
| epoch   1 | valid answer loss 0.154 | bpc    0.222 | F1 0.545
-----------------------------------------------------------------------------------------
| epoch   1 | test  aux    loss 1.302 | bpc    1.878
| epoch   1 | test  main   loss 0.205 | scaled 0.409 | comb loss 1.711
| epoch   1 | test  answer loss 0.152 | bpc    0.219 | F1 0.537
-----------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------
| epoch   2 | valid aux    loss 1.202 | bpc    1.734
| epoch   2 | valid main   loss 0.191 | scaled 0.383 | comb loss 1.585
| epoch   2 | valid answer loss 0.134 | bpc    0.193 | F1 0.578
-----------------------------------------------------------------------------------------
| epoch   2 | test  aux    loss 1.203 | bpc    1.736
| epoch   2 | test  main   loss 0.190 | scaled 0.379 | comb loss 1.583
| epoch   2 | test  answer loss 0.132 | bpc    0.190 | F1 0.567
-----------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------
| epoch   3 | valid aux    loss 1.162 | bpc    1.677
| epoch   3 | valid main   loss 0.184 | scaled 0.368 | comb loss 1.530
| epoch   3 | valid answer loss 0.126 | bpc    0.182 | F1 0.596
-----------------------------------------------------------------------------------------
| epoch   3 | test  aux    loss 1.164 | bpc    1.679
| epoch   3 | test  main   loss 0.183 | scaled 0.365 | comb loss 1.529
| epoch   3 | test  answer loss 0.125 | bpc    0.180 | F1 0.588
-----------------------------------------------------------------------------------------
```
It runs really slow and is suprisingly lame so far. Peaks:
```
-----------------------------------------------------------------------------------------
| epoch  16 | valid aux    loss 1.063 | bpc    1.534
| epoch  16 | valid main   loss 0.182 | scaled 0.364 | comb loss 1.428
| epoch  16 | valid answer loss 0.122 | bpc    0.176 | F1 0.621
-----------------------------------------------------------------------------------------
| epoch  16 | test  aux    loss 1.065 | bpc    1.537
| epoch  16 | test  main   loss 0.183 | scaled 0.365 | comb loss 1.431
| epoch  16 | test  answer loss 0.122 | bpc    0.176 | F1 0.613
-----------------------------------------------------------------------------------------
```

What about just using main loss?
```
python train.py --dataset quora-large --levels 5 --ksize 3 --nhid 300 --optim='Adam' --lr 1e-3 --aux 0.01 --main 1 --gpu 0 --seq_len 1600 --validseqlen 1340 --dropout 0.05 --batch_size 16
```
Peaking:
```
| epoch  18 |   100/ 3492 batches | lr 1.0e-04 | ms/batch 105.31 | aux 1.765 | main 0.162 | loss 0.180
| epoch  18 |   200/ 3492 batches | lr 1.0e-04 | ms/batch 105.29 | aux 1.747 | main 0.158 | loss 0.175
| epoch  18 |   300/ 3492 batches | lr 1.0e-04 | ms/batch 105.14 | aux 1.748 | main 0.159 | loss 0.176
| epoch  18 |   400/ 3492 batches | lr 1.0e-04 | ms/batch 105.40 | aux 1.742 | main 0.161 | loss 0.178
| epoch  18 |   500/ 3492 batches | lr 1.0e-04 | ms/batch 104.85 | aux 1.740 | main 0.164 | loss 0.181
| epoch  18 |   600/ 3492 batches | lr 1.0e-04 | ms/batch 104.87 | aux 1.744 | main 0.165 | loss 0.182
| epoch  18 |   700/ 3492 batches | lr 1.0e-04 | ms/batch 104.79 | aux 1.747 | main 0.164 | loss 0.181
| epoch  18 |   800/ 3492 batches | lr 1.0e-04 | ms/batch 105.35 | aux 1.744 | main 0.159 | loss 0.176
| epoch  18 |   900/ 3492 batches | lr 1.0e-04 | ms/batch 104.83 | aux 1.744 | main 0.162 | loss 0.180
| epoch  18 |  1000/ 3492 batches | lr 1.0e-04 | ms/batch 105.11 | aux 1.748 | main 0.156 | loss 0.174
| epoch  18 |  1100/ 3492 batches | lr 1.0e-04 | ms/batch 104.65 | aux 1.743 | main 0.161 | loss 0.178
| epoch  18 |  1200/ 3492 batches | lr 1.0e-04 | ms/batch 105.80 | aux 1.745 | main 0.164 | loss 0.182
| epoch  18 |  1300/ 3492 batches | lr 1.0e-04 | ms/batch 105.16 | aux 1.746 | main 0.158 | loss 0.175
| epoch  18 |  1400/ 3492 batches | lr 1.0e-04 | ms/batch 104.52 | aux 1.746 | main 0.160 | loss 0.178
| epoch  18 |  1500/ 3492 batches | lr 1.0e-04 | ms/batch 105.60 | aux 1.745 | main 0.157 | loss 0.175
| epoch  18 |  1600/ 3492 batches | lr 1.0e-04 | ms/batch 105.35 | aux 1.748 | main 0.158 | loss 0.176
| epoch  18 |  1700/ 3492 batches | lr 1.0e-04 | ms/batch 104.60 | aux 1.740 | main 0.164 | loss 0.182
| epoch  18 |  1800/ 3492 batches | lr 1.0e-04 | ms/batch 105.14 | aux 1.736 | main 0.158 | loss 0.175
| epoch  18 |  1900/ 3492 batches | lr 1.0e-04 | ms/batch 104.60 | aux 1.742 | main 0.161 | loss 0.179
| epoch  18 |  2000/ 3492 batches | lr 1.0e-04 | ms/batch 105.51 | aux 1.741 | main 0.163 | loss 0.181
| epoch  18 |  2100/ 3492 batches | lr 1.0e-04 | ms/batch 105.60 | aux 1.745 | main 0.156 | loss 0.174
| epoch  18 |  2200/ 3492 batches | lr 1.0e-04 | ms/batch 104.55 | aux 1.742 | main 0.155 | loss 0.173
| epoch  18 |  2300/ 3492 batches | lr 1.0e-04 | ms/batch 104.70 | aux 1.742 | main 0.165 | loss 0.182
| epoch  18 |  2400/ 3492 batches | lr 1.0e-04 | ms/batch 105.93 | aux 1.742 | main 0.156 | loss 0.174
| epoch  18 |  2500/ 3492 batches | lr 1.0e-04 | ms/batch 104.79 | aux 1.745 | main 0.153 | loss 0.171
| epoch  18 |  2600/ 3492 batches | lr 1.0e-04 | ms/batch 105.19 | aux 1.743 | main 0.156 | loss 0.173
| epoch  18 |  2700/ 3492 batches | lr 1.0e-04 | ms/batch 104.64 | aux 1.742 | main 0.157 | loss 0.175
| epoch  18 |  2800/ 3492 batches | lr 1.0e-04 | ms/batch 105.57 | aux 1.741 | main 0.157 | loss 0.175
| epoch  18 |  2900/ 3492 batches | lr 1.0e-04 | ms/batch 105.74 | aux 1.743 | main 0.163 | loss 0.180
| epoch  18 |  3000/ 3492 batches | lr 1.0e-04 | ms/batch 104.84 | aux 1.745 | main 0.157 | loss 0.175
| epoch  18 |  3100/ 3492 batches | lr 1.0e-04 | ms/batch 104.51 | aux 1.741 | main 0.160 | loss 0.177
| epoch  18 |  3200/ 3492 batches | lr 1.0e-04 | ms/batch 105.17 | aux 1.740 | main 0.153 | loss 0.171
| epoch  18 |  3300/ 3492 batches | lr 1.0e-04 | ms/batch 104.70 | aux 1.744 | main 0.158 | loss 0.175
| epoch  18 |  3400/ 3492 batches | lr 1.0e-04 | ms/batch 105.61 | aux 1.744 | main 0.166 | loss 0.183
-----------------------------------------------------------------------------------------
| epoch  18 | valid aux    loss 1.674 | bpc    2.414
| epoch  18 | valid main   loss 0.180 | scaled 0.180 | comb loss 0.196
| epoch  18 | valid answer loss 0.120 | bpc    0.174 | F1 0.621
-----------------------------------------------------------------------------------------
| epoch  18 | test  aux    loss 1.675 | bpc    2.416
| epoch  18 | test  main   loss 0.180 | scaled 0.180 | comb loss 0.197
| epoch  18 | test  answer loss 0.120 | bpc    0.173 | F1 0.614
-----------------------------------------------------------------------------------------
```
Training seems to have reached a steady state.

What about increasing kernel size?
```
python train.py --dataset quora-large --levels 4 --ksize 5 --nhid 300 --optim='Adam' --lr 1e-3 --aux 0.01 --main 1 --gpu 0 --seq_len 1600 --validseqlen 1340 --dropout 0.05 --batch_size 16
```
Peak?
```
-----------------------------------------------------------------------------------------
| epoch   8 | valid aux    loss 1.750 | bpc    2.524
| epoch   8 | valid main   loss 0.183 | scaled 0.183 | comb loss 0.201
| epoch   8 | valid answer loss 0.127 | bpc    0.183 | F1 0.607
-----------------------------------------------------------------------------------------
| epoch   8 | test  aux    loss 1.750 | bpc    2.525
| epoch   8 | test  main   loss 0.182 | scaled 0.182 | comb loss 0.199
| epoch   8 | test  answer loss 0.124 | bpc    0.179 | F1 0.601
-----------------------------------------------------------------------------------------
```

What about adding another 1-width convolution at the end?
```
python train.py --dataset quora-large --levels 4 --ksize 5 --nhid 300 --optim='Adam' --lr 1e-3 --aux 0.01 --main 1 --gpu 0 --seq_len 1600 --validseqlen 1340 --dropout 0.05 --batch_size 16 --k1levels 1

-----------------------------------------------------------------------------------------
| epoch  11 | valid aux    loss 1.675 | bpc    2.417
| epoch  11 | valid main   loss 0.185 | scaled 0.185 | comb loss 0.201
| epoch  11 | valid answer loss 0.126 | bpc    0.181 | F1 0.612
-----------------------------------------------------------------------------------------
| epoch  11 | test  aux    loss 1.676 | bpc    2.418
| epoch  11 | test  main   loss 0.183 | scaled 0.183 | comb loss 0.200
| epoch  11 | test  answer loss 0.123 | bpc    0.178 | F1 0.603
-----------------------------------------------------------------------------------------
```

More hidden units:
```
python train.py --dataset quora-large --levels 5 --ksize 3 --nhid 500 --optim='Adam' --lr 1e-3 --aux 0.01 --main 1 --gpu 0 --seq_len 1600 --validseqlen 1340 --dropout 0.1 --batch_size 16
```
Peaks:
```
| epoch  28 | valid aux    loss 1.621 | bpc    2.338
| epoch  28 | valid main   loss 0.179 | scaled 0.179 | comb loss 0.195
| epoch  28 | valid answer loss 0.118 | bpc    0.171 | F1 0.624
-----------------------------------------------------------------------------------------
| epoch  28 | test  aux    loss 1.622 | bpc    2.340
| epoch  28 | test  main   loss 0.180 | scaled 0.180 | comb loss 0.196
| epoch  28 | test  answer loss 0.118 | bpc    0.170 | F1 0.615
```

