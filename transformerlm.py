# coding: utf-8
from clearml import Task

task = Task.init(project_name='test-hpsearch', task_name='transformers-lm')
logger = task.get_logger()

import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import data
import model as transformerlm

from torchinfo import summary
import sys

device = torch.device("cuda")

###############################################################################
# Load data
###############################################################################
def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

###############################################################################
# Training code
###############################################################################
def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(args, source, i):
    seq_len = min(args.get('bptt'), len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(args, model, data_source, corpus, criterion):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.get('model') != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.get('bptt')):
            data, targets = get_batch(args, data_source, i)
            if args.get('model') == 'Transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train(args, model, train_data, corpus, epoch, criterion, lr):
    # Turn on training mode which enables dropout.
    model.train()
    optimizer= torch.optim.SGD(model.parameters(),lr=lr)
    #summary(model, input_size=(60, 35), dtypes=[torch.long])
    # Loop over epochs.

    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.get('model') != 'Transformer':
        hidden = model.init_hidden(args.get('batch_size'))

    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.get('bptt'))):

        data, targets = get_batch(args, train_data, i)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.

        #optimizer.zero_grad()
        model.zero_grad()
        if args.get('model') == 'Transformer':
            output = model(data)
            output = output.view(-1, ntokens)
        
        loss = criterion(output, targets)
        loss.backward()
        
        #optimizer.step()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.get('clip'))
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % args.get('log-interval') == 0 and batch > 0:
            cur_loss = total_loss / args.get('log-interval')
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.get('bptt'), lr,
                elapsed * 1000 / args.get('log-interval'), cur_loss, math.exp(cur_loss)))
            logger.report_scalar(title='ppl', series='training', value=math.exp(cur_loss), iteration=epoch)
            total_loss = 0
            start_time = time.time()

        if args.get('dry-run'):
            break

def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


def main(args):
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.get('seed', 1111))
    if torch.cuda.is_available():
        if not args.get('cuda'):
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    corpus = data.Corpus(args.get('data'))
    eval_batch_size = 10
    train_data = batchify(corpus.train, args.get('batch_size'))
    val_data = batchify(corpus.valid, eval_batch_size)
    test_data = batchify(corpus.test, eval_batch_size)

    ###############################################################################
    # Build the model
    ###############################################################################
    ntokens = len(corpus.dictionary)
    model = transformerlm.TransformerModel(ntokens, args.get('emsize'), args.get('nhead'), args.get('nhid'), args.get('nlayers'), args.get('dropout')).to(device)

    best_val_loss = None
    lr = args.get('lr')
    criterion = nn.NLLLoss()

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.get('epochs')+1):
            epoch_start_time = time.time()
            train(args, model, train_data, corpus, epoch, criterion, lr)
            
            val_loss = evaluate(args, model, val_data, corpus, criterion)

            logger.report_scalar(title='ppl',series='validation', value=math.exp(val_loss), iteration=epoch)

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            print('-' * 89)

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.get('save'), 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 2.0

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.get('save'), 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_loss = evaluate(args, model, test_data, corpus, criterion)
    logger.report_scalar(title='ppl',series='test', value=math.exp(test_loss), iteration=epoch)

    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

    if len(args.get('onnx-export')) > 0:
        # Export the model in ONNX format.
        export_onnx(args.get('onnx-export'), batch_size=1, seq_len=args.get('bptt'))


if __name__ == '__main__':

    configuration_dict = {
    'epochs': 5, 
    'batch_size': 60, 
    'dropout': 0.2, 
    'lr': 20,
    'cuda': True,
    'log-interval': 200,
    'save': 'model.pt',
    'onnx-export': '',
    'dry-run': False,
    'nhead': 8,
    'seed': 1111,
    'model': 'Transformer',
    'data': './data/wikitext-2',
    'emsize': 512,
    'nhid': 1024,
    'nlayers': 6,
    'clip': 0.25,
    'bptt': 35,
    'tied': True,
    'onnx-export': ''
    }


    configuration_dict = task.connect(configuration_dict)
    print('Task ID number is: {}'.format(task.id)) 
    main(configuration_dict)









