import os
import pandas as pd

outpath1 = '../TCN/char_cnn/data/quora-large'
outpath2 = '../TCN/char_cnn/data/quora-small'
outpath3 = '../TCN/char_cnn/data/quora-large-rev'
outpath4 = '../TCN/char_cnn/data/quora-small-rev'

os.makedirs(outpath1, exist_ok=True)
os.makedirs(outpath2, exist_ok=True)
os.makedirs(outpath3, exist_ok=True)
os.makedirs(outpath4, exist_ok=True)

df = pd.read_csv('train.csv').sample(frac=1.0, random_state=0)
df['question_text'] = df['question_text'].str.replace('ª', 'a').str.replace('\n', '')
df['qa'] = df['question_text']

def write_strings(df, fname, rev=False):
  def maybe_rev(s):
    return s[::-1] if rev else s

  xstr = ''.join(maybe_rev(row['question_text']) + 'ª' + str(row['target']) + '\n' for _, row in df.iterrows())

  with open(fname + '.txt', 'w') as fo:
    fo.write(xstr)

# Partition and create large dataset.
n3 = len(df)
n2 = 9 * n3 // 10
n1 = 8 * n3 // 10

write_strings(df.iloc[:n1],   outpath1 + '/train')
write_strings(df.iloc[n1:n2], outpath1 + '/valid')
write_strings(df.iloc[n2:n3], outpath1 + '/test')
write_strings(df.iloc[:n1],   outpath3 + '/train', rev=True)
write_strings(df.iloc[n1:n2], outpath3 + '/valid', rev=True)
write_strings(df.iloc[n2:n3], outpath3 + '/test', rev=True)

n3 = len(df) // 10
n2 = 9 * n3 // 10
n1 = 8 * n3 // 10

write_strings(df.iloc[:n1],   outpath2 + '/train')
write_strings(df.iloc[n1:n2], outpath2 + '/valid')
write_strings(df.iloc[n2:n3], outpath2 + '/test')
write_strings(df.iloc[:n1],   outpath4 + '/train', rev=True)
write_strings(df.iloc[n1:n2], outpath4 + '/valid', rev=True)
write_strings(df.iloc[n2:n3], outpath4 + '/test', rev=True)
