import argparse
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append("../../")
from TCN.quora.utils import *
from TCN.quora.model import TCN
import time
import math


import warnings
warnings.filterwarnings("ignore")   # Suppress the RunTimeWarning on unicode


parser = argparse.ArgumentParser(description='Sequence Modeling - Character Level Language Model')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--gpu', type=int, default=0,
                    help='which GPU to use')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (default: 0.1)')
parser.add_argument('--emb_dropout', type=float, default=0.1,
                    help='dropout applied to the embedded layer (0 = no dropout) (default: 0.1)')
parser.add_argument('--clip', type=float, default=0.15,
                    help='gradient clip, -1 means no clip (default: 0.15)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=3,
                    help='kernel size (default: 3)')
parser.add_argument('--levels', type=int, default=3,
                    help='# of levels (default: 3)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=4,
                    help='initial learning rate (default: 4)')
parser.add_argument('--main', type=float, default=0,
                    help='scale main loss before adding to aux loss (default: 0)')
parser.add_argument('--emsize', type=int, default=100,
                    help='dimension of character embeddings (default: 100)')
parser.add_argument('--optim', type=str, default='SGD',
                    help='optimizer to use (default: SGD)')
parser.add_argument('--nhid', type=int, default=450,
                    help='number of hidden units per layer (default: 450)')
parser.add_argument('--validseqlen', type=int, default=320,
                    help='valid sequence length (default: 320)')
parser.add_argument('--seq_len', type=int, default=400,
                    help='total sequence length, including effective history (default: 400)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--dataset', type=str, default='ptb',
                    help='dataset to use (default: ptb)')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


print(args)
trnqstr, vldqstr, tstqstr, trnastr, vldastr, tstastr, n_characters, n_labels, idx_eol = data_generator(args)

trnqdata, trnadata = batchify(trnqstr, trnastr, args.batch_size, args)
vldqdata, vldadata = batchify(vldqstr, vldastr, 1, args)
tstqdata, tstadata = batchify(tstqstr, tstastr, 1, args)
print("Corpus size: ", n_characters)


num_chans = [args.nhid] * (args.levels - 1) + [args.emsize]
k_size = args.ksize
dropout = args.dropout
emb_dropout = args.emb_dropout
model = TCN(args.emsize, n_characters, n_labels, num_chans, kernel_size=k_size, dropout=dropout, emb_dropout=emb_dropout)


if args.cuda:
    model.cuda(args.gpu)


criterion = nn.CrossEntropyLoss(reduction='none')
lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def evaluate(qdata, adata):
    # F1 only makes sense if two labels.
    assert(n_labels == 2)

    model.eval()
    char_count = 0
    aux_loss_sum = 0
    main_loss_sum = 0
    answer_count = 0
    answer_loss_sum = 0
    # count1 = 0
    # count2 = 0

    label_values = []
    scores = []

    source_len = qdata.size(1)
    for batch, i in enumerate(range(0, source_len - 1, args.validseqlen)):
        if i + args.seq_len - args.validseqlen >= source_len:
            continue
        inp, target, labels = get_batch(qdata, adata, i, args)
        output, labels_out = model(inp)
        eff_history = args.seq_len - args.validseqlen

        final_output = output[:, eff_history:].contiguous().view(-1, n_characters)
        final_target = target[:, eff_history:].contiguous().view(-1)
        aux_element_loss = criterion(final_output, final_target).detach().cpu().numpy()

        final_labels_out = labels_out[:, eff_history:].contiguous().view(-1, n_labels)
        final_labels = labels[:, eff_history:].contiguous().view(-1)
        main_element_loss = criterion(final_labels_out, final_labels.long()).detach().cpu().numpy()

        char_count += args.validseqlen
        aux_loss_sum += aux_element_loss.sum()
        main_loss_sum += main_element_loss.sum()

        answer = np.equal(inp[:, eff_history:].contiguous().view(-1).cpu().numpy(), idx_eol)
        answer_count += answer.sum()
        answer_loss_sum += main_element_loss[answer].sum()

        answer_val = np.equal(final_labels.cpu().numpy()[answer], 1, dtype='int')
        label_values.extend(answer_val)
        labels_probs = torch.nn.functional.softmax(final_labels_out, dim=1)
        scores.extend(labels_probs[:, 1].detach().cpu().numpy()[answer])

        # count1 += len(answer_val)
        # count2 += answer_val.sum()

    aux_loss = aux_loss_sum / char_count
    main_loss = main_loss_sum / char_count
    answer_loss = answer_loss_sum / answer_count

    # print(char_count, answer_count, count1, count2)
    # print(len(label_values), np.quantile(label_values, np.linspace(0.9, 1, 11)))
    # print(len(scores), ['%.2f' % x for x in np.quantile(scores, np.linspace(0.9, 1, 11))])

    return aux_loss, main_loss, answer_loss, max_f1(label_values, scores)


def train(epoch):
    model.train()
    total_loss = 0
    total_aux_loss = 0
    total_main_loss = 0
    start_time = time.time()
    source_len = trnqdata.size(1)
    for batch_idx, i in enumerate(range(0, source_len - 1, args.validseqlen)):
        if i + args.seq_len - args.validseqlen >= source_len:
            continue
        inp, target, labels = get_batch(trnqdata, trnadata, i, args)
        optimizer.zero_grad()
        output, labels_out = model(inp)
        eff_history = args.seq_len - args.validseqlen

        final_output = output[:, eff_history:].contiguous().view(-1, n_characters)
        final_target = target[:, eff_history:].contiguous().view(-1)
        aux_element_loss = criterion(final_output, final_target)

        final_labels_out = labels_out[:, eff_history:].contiguous().view(-1, n_labels)
        final_labels = labels[:, eff_history:].contiguous().view(-1)
        main_element_loss = criterion(final_labels_out, final_labels.long())

        aux_loss = aux_element_loss.mean()
        main_loss = main_element_loss.mean()
        loss = aux_loss + args.main * main_loss
        loss.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.data[0]
        total_aux_loss += aux_loss.data[0]
        total_main_loss += main_loss.data[0]

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            cur_loss = total_loss / args.log_interval
            cur_aux_loss = total_aux_loss / args.log_interval
            cur_main_loss = total_main_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:5.1e} | ms/batch {:5.2f} | '
                  'aux {:5.3f} | main {:5.3f} | loss {:5.3f}'.format(
                epoch, batch_idx, int((source_len-0.5) / args.validseqlen), lr,
                              elapsed * 1000 / args.log_interval,
                              cur_aux_loss, cur_main_loss, cur_loss))
            total_loss = 0
            total_aux_loss = 0
            total_main_loss = 0

            start_time = time.time()

def main():
    global lr
    try:
        print("Training for %d epochs..." % args.epochs)
        all_losses = []
        best_valid_loss = 1e7
        for epoch in range(1, args.epochs + 1):
            train(epoch)

            print('-' * 89)
            valid_aux_loss, valid_main_loss, valid_answer_loss, f1 = evaluate(vldqdata, vldadata)
            valid_loss = valid_aux_loss + args.main * valid_main_loss
            print('| epoch {:3d} | valid aux    loss {:5.3f} | bpc {:8.3f}'.format(
                epoch, valid_aux_loss, valid_aux_loss / math.log(2)))
            print('| epoch {:3d} | valid main   loss {:5.3f} | scaled {:5.3f} | comb loss {:5.3f}'.format(
                epoch, valid_main_loss, valid_main_loss * args.main, valid_loss))
            print('| epoch {:3d} | valid answer loss {:5.3f} | bpc {:8.3f} | F1 {:5.3f}'.format(
                epoch, valid_answer_loss, valid_answer_loss / math.log(2), f1))

            print('-' * 89)
            test_aux_loss, test_main_loss, test_answer_loss, f1 = evaluate(tstqdata, tstadata)
            test_loss = test_aux_loss + args.main * test_main_loss
            print('| epoch {:3d} | test  aux    loss {:5.3f} | bpc {:8.3f}'.format(
                epoch, test_aux_loss, test_aux_loss / math.log(2)))
            print('| epoch {:3d} | test  main   loss {:5.3f} | scaled {:5.3f} | comb loss {:5.3f}'.format(
                epoch, test_main_loss, test_main_loss * args.main, test_loss))
            print('| epoch {:3d} | test  answer loss {:5.3f} | bpc {:8.3f} | F1 {:5.3f}'.format(
                epoch, test_answer_loss, test_answer_loss / math.log(2), f1))
            print('-' * 89)

            if epoch > 5 and valid_loss > max(all_losses[-3:]):
                lr = lr / 10.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            all_losses.append(valid_loss)

            if valid_loss < best_valid_loss:
                print("Saving...")
                save(model)
                best_valid_loss = valid_loss

    except KeyboardInterrupt:
        print('-' * 89)
        print("Saving before quit...")
        save(model)

    # Run on test data.
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.3f} | test bpc {:8.3f}'.format(
        test_loss, test_loss / math.log(2)))
    print('=' * 89)

# train_by_random_chunk()
if __name__ == "__main__":
    main()
