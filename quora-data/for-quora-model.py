import os
import pandas as pd

outpath1 = '../TCN/quora/data/quora-large'
outpath2 = '../TCN/quora/data/quora-small'

os.makedirs(outpath1, exist_ok=True)
os.makedirs(outpath2, exist_ok=True)

df = pd.read_csv('train.csv')

def write_strings(df, fname):
  qstr = ''.join(q.replace('\n', '')+'\n' for q in df['question_text'])
  astr = ''.join(str(a)[:1] * (len(q.replace('\n', ''))+1) for q, a in zip(df['question_text'], df['target']))

  assert(len(qstr) == len(astr))

  with open(fname + '-q.txt', 'w') as fo:
    fo.write(qstr)

  with open(fname + '-a.txt', 'w') as fo:
    fo.write(astr)

# Partition and create large dataset.
n3 = len(df)
n2 = 9 * n3 // 10
n1 = 8 * n3 // 10

write_strings(df.iloc[:n1],   outpath1 + '/train')
write_strings(df.iloc[n1:n2], outpath1 + '/valid')
write_strings(df.iloc[n2:n3], outpath1 + '/test')

n3 = len(df) // 10
n2 = 9 * n3 // 10
n1 = 8 * n3 // 10

write_strings(df.iloc[:n1],   outpath2 + '/train')
write_strings(df.iloc[n1:n2], outpath2 + '/valid')
write_strings(df.iloc[n2:n3], outpath2 + '/test')
