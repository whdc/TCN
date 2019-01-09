import os
import pandas as pd

outpath1 = '../TCN/quora/data/quora-large'
outpath2 = '../TCN/quora/data/quora-small'
outpath3 = '../TCN/quora/data/quora-tiny'

os.makedirs(outpath1, exist_ok=True)
os.makedirs(outpath2, exist_ok=True)
os.makedirs(outpath3, exist_ok=True)

df = pd.read_csv('train.csv').sample(frac=1.0, random_state=0)

assert(df['target'].unique().tolist() == [0, 1])

df['question_text'] = df['question_text'].replace('\n', '')

def write_strings(df, fname):
  qstr = ''.join(q+'\n' for q in df['question_text'])
  astr = ''.join(str(a) * (len(q)+1) for q, a in zip(df['question_text'], df['target']))

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

n3 = len(df) // 100
n2 = 9 * n3 // 10
n1 = 8 * n3 // 10

write_strings(df.iloc[:n1],   outpath3 + '/train')
write_strings(df.iloc[n1:n2], outpath3 + '/valid')
write_strings(df.iloc[n2:n3], outpath3 + '/test')
