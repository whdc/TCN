import unidecode
import torch
from torch.autograd import Variable
from collections import Counter
import observations
import os
import pickle
import numpy as np


cuda = torch.cuda.is_available()


def data_generator(args):
    if hasattr(observations, args.dataset):
      trainfile, testfile, validfile = getattr(observations, args.dataset)('data/')
    else:
      trainfile = open('data/%s/train.txt' % args.dataset).read()
      validfile = open('data/%s/valid.txt' % args.dataset).read()
      testfile  = open('data/%s/test.txt'  % args.dataset).read()

    corpus = Corpus(trainfile + " " + validfile + " " + testfile)

    if os.path.exists('data/' + args.dataset + '/data.pkl'):
      data = pickle.load(open('data/' + args.dataset + '/data.pkl', 'rb'))
    else:
      trainstr = char_tensor(corpus, trainfile)
      validstr = char_tensor(corpus, validfile)
      teststr = char_tensor(corpus, testfile)
      n_characters = len(corpus.dict)
      idx_ans = corpus.dict.char2idx['ª']
      idx_one = corpus.dict.char2idx['1']
      data = (trainstr, validstr, teststr, n_characters, idx_ans, idx_one)
      pickle.dump(data, open('data/' + args.dataset + '/data.pkl', 'wb'))

    return data

def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)


class Dictionary(object):
    def __init__(self):
        self.char2idx = {}
        self.idx2char = []
        self.counter = Counter()

    def add_word(self, char):
        self.counter[char] += 1

    def prep_dict(self):
        for char in self.counter:
            if char not in self.char2idx:
                self.idx2char.append(char)
                self.char2idx[char] = len(self.idx2char) - 1

    def __len__(self):
        return len(self.idx2char)


class Corpus(object):
    def __init__(self, string):
        self.dict = Dictionary()
        for c in string:
            self.dict.add_word(c)
        self.dict.prep_dict()


def char_tensor(corpus, string):
    tensor = torch.zeros(len(string)).long()
    for i in range(len(string)):
        tensor[i] = corpus.dict.char2idx[string[i]]
    return Variable(tensor).cuda(args.gpu) if cuda else Variable(tensor)


def batchify(data, batch_size, args):
    """The output should have size [L x batch_size], where L could be a long sequence length"""
    # Work out how cleanly we can divide the dataset into batch_size parts (i.e. continuous seqs).
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the batch_size batches.
    data = data.view(batch_size, -1)
    if args.cuda:
        data = data.cuda(args.gpu)
    return data


def get_batch(source, start_index, args):
    seq_len = min(args.seq_len, source.size(1) - 1 - start_index)
    end_index = start_index + seq_len
    inp = source[:, start_index:end_index].contiguous()
    target = source[:, start_index+1:end_index+1].contiguous()  # The successors of the inp.
    return inp, target


def save(model):
    save_filename = 'model.pt'
    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)

def max_f1(labels, scores):
  # Sort labels by scores.
  y = np.array([l for l, s, in sorted(zip(labels, scores), key=lambda x: x[1])])
  
  tc = np.sum(y)
  fc = len(y) - tc
  fp = np.cumsum(y)
  tp = tc - fp
  tn = np.arange(1, len(y)+1) - fp
  fn = fc - tn
  precis = tp / (tp + fp)
  recall = tp / (tp + fn)
  f1 = 2 * (precis * recall) / (precis + recall)

  # print('tp', tp)
  # print('fp', fp)
  # print('tn', tn)
  # print('fn', fn)
  # print('f1', f1)

  return np.nanmax(f1)
