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
