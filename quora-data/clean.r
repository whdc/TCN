outpath1 = '../TCN/char_cnn/data/quora'
outpath2 = '../TCN/char_cnn/data/quora-small'

# Create data stream.
a = read.csv('train.csv')
x = paste(a$question_text, a$target, sep='ª')

# Partition and create large dataset.
n3 = length(x)
n2 = 9 * n3 %/% 10
n1 = 8 * n3 %/% 10

trn = x[1:n1]
vld = x[(n1+1):n2]
tst = x[(n2+1):n3]

dir.create(outpath1, showWarnings=F, recursive=T)
write.table(data.frame(x=trn), sprintf('%s/train.txt', outpath1), row.names=F, col.names=F, quote=F)
write.table(data.frame(x=vld), sprintf('%s/valid.txt', outpath1), row.names=F, col.names=F, quote=F)
write.table(data.frame(x=tst), sprintf('%s/test.txt', outpath1), row.names=F, col.names=F, quote=F)

# Thin by 10 to create small dataset.
x = x[1:(length(x) %/% 10)]

n3 = length(x)
n2 = 9 * n3 %/% 10
n1 = 8 * n3 %/% 10

trn = x[1:n1]
vld = x[(n1+1):n2]
tst = x[(n2+1):n3]

dir.create(outpath2, showWarnings=F, recursive=T)
write.table(data.frame(x=trn), sprintf('%s/train.txt', outpath2), row.names=F, col.names=F, quote=F)
write.table(data.frame(x=vld), sprintf('%s/valid.txt', outpath2), row.names=F, col.names=F, quote=F)
write.table(data.frame(x=tst), sprintf('%s/test.txt', outpath2), row.names=F, col.names=F, quote=F)
