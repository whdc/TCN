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
    if os.path.exists('data/' + args.dataset + '/data.pkl'):
      data = pickle.load(open('data/' + args.dataset + '/data.pkl', 'rb'))
    else:
      trnqfile = open('data/%s/train-q.txt' % args.dataset).read()
      vldqfile = open('data/%s/valid-q.txt' % args.dataset).read()
      tstqfile = open('data/%s/test-q.txt'  % args.dataset).read()

      if args.model == 'word':
        trnqfile = trnqfile.split(' ')
        vldqfile = vldqfile.split(' ')
        tstqfile = tstqfile.split(' ')

      trnafile = open('data/%s/train-a.txt' % args.dataset).read()
      vldafile = open('data/%s/valid-a.txt' % args.dataset).read()
      tstafile = open('data/%s/test-a.txt'  % args.dataset).read()

      corpus = Corpus(trnqfile + vldqfile + tstqfile, args.thresh)

      trnqstr = token_tensor(corpus, trnqfile)
      vldqstr = token_tensor(corpus, vldqfile)
      tstqstr = token_tensor(corpus, tstqfile)
      trnastr = torch.from_numpy(np.array(list(trnafile), dtype='uint8'))
      vldastr = torch.from_numpy(np.array(list(vldafile), dtype='uint8'))
      tstastr = torch.from_numpy(np.array(list(tstafile), dtype='uint8'))

      n_characters = corpus.num_tokens()
      n_labels = 2
      idx_eol = corpus.lookup('\n')

      data = (trnqstr, vldqstr, tstqstr, trnastr, vldastr, tstastr, n_characters, n_labels, idx_eol)
      pickle.dump(data, open('data/' + args.dataset + '/data.pkl', 'wb'))

    return data

def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)

class SpecialToken:
    UNK = 0
    PAD = 1
    LEN = 2

class Corpus(object):
    # Fine if `tokens` is just a string.
    def __init__(self, tokens, thresh):
        self.char2idx = {}
        self.counter = Counter()
        self.thresh = thresh
        for token in tokens: 
            self.counter[token] += 1
        for token in self.counter:
            if self.counter[token] > self.thresh and token not in self.char2idx:
                self.char2idx[token] = len(self.char2idx)

    def lookup(self, token):
        if self.counter[token] > self.thresh:
            return self.char2idx[token]
        else:
            return len(self.char2idx) + SpecialToken.UNK
    
    def num_tokens(self):
        return len(self.char2idx) + SpecialToken.LEN


def token_tensor(corpus, tokens):
    tensor = torch.zeros(len(tokens)).long()
    for i in range(len(tokens)):
        tensor[i] = corpus.lookup(tokens[i])
    return tensor


def batchify(qdata, adata, batch_size, args):
    assert(len(qdata) == len(adata))

    qdata = Variable(qdata).cuda(args.gpu) if cuda else Variable(qdata)
    """The output should have size [L x batch_size], where L could be a long sequence length"""
    # Work out how cleanly we can divide the dataset into batch_size parts (i.e. continuous seqs).
    nbatch = qdata.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    qdata = qdata.narrow(0, 0, nbatch * batch_size)
    adata = adata.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the batch_size batches.
    qdata = qdata.view(batch_size, -1)
    adata = adata.view(batch_size, -1)
    if args.cuda:
        qdata = qdata.cuda(args.gpu)
        adata = adata.cuda(args.gpu)
    return qdata, adata


def get_batch(qdata, adata, start_index, args):
    seq_len = min(args.seq_len, qdata.size(1) - 1 - start_index)
    end_index = start_index + seq_len
    inp = qdata[:, start_index:end_index].contiguous()
    target = qdata[:, start_index+1:end_index+1].contiguous()  # The successors of the inp.
    labels = adata[:, start_index:end_index].contiguous()  # The successors of the inp.
    return inp, target, labels


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
