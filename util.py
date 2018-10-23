from text import torchtext
import time
import os
import sys
import torch
import random
import numpy as np


def get_context_question(ex, context, question, field):
    return ex.context_special + ex.context + ex.question_special + ex.question


def preprocess_examples(args, tasks, splits, field, logger=None, train=True):
    min_length = 1
    max_context_length = args.max_train_context_length if train else args.max_val_context_length
    is_too_long = lambda ex: (len(ex.answer)>args.max_answer_length or
        len(ex.context)>max_context_length)
    is_too_short = lambda ex: (len(ex.answer)<min_length or 
        len(ex.context)<min_length)

    for task, s in zip(tasks, splits):
        if logger is not None:
            logger.info(f'{task} has {len(s.examples)} examples')
        if 'cnn' in task or 'dailymail' in task or 'imdb' in task:
            for x in s.examples:
                x.context = x.context[:max_context_length]

        if train:
            l = len(s.examples)
            s.examples = [ex for ex in s.examples if not is_too_long(ex)]
            if len(s.examples) < l:
                if logger is not None:
                    logger.info(f'Filtering out long {task} examples: {l} -> {len(s.examples)}')
    
            l = len(s.examples)
            s.examples = [ex for ex in s.examples if not is_too_short(ex)]
            if len(s.examples) < l:
                if logger is not None:
                   logger.info(f'Filtering out short {task} examples: {l} -> {len(s.examples)}')
    
            l = len(s.examples)
            s.examples = [ex for ex in s.examples if 'This page includes the show' not in ex.answer]
            if len(s.examples) < l:
                if logger is not None:
                   logger.info(f'Filtering {task} examples with a dummy summary: {l} -> {len(s.examples)} ')
       
        if logger is not None:
            context_lengths = [len(ex.context) for ex in s.examples] 
            question_lengths = [len(ex.question) for ex in s.examples] 
            answer_lengths = [len(ex.answer) for ex in s.examples] 

            logger.info(f'{task} context lengths (min, mean, max): {np.min(context_lengths)}, {int(np.mean(context_lengths))}, {np.max(context_lengths)}') 
            logger.info(f'{task} question lengths (min, mean, max): {np.min(question_lengths)}, {int(np.mean(question_lengths))}, {np.max(question_lengths)}')
            logger.info(f'{task} answer lengths (min, mean, max): {np.min(answer_lengths)}, {int(np.mean(answer_lengths))}, {np.max(answer_lengths)}')

        for x in s.examples:
            x.context_question = get_context_question(x, x.context, x.question, field)

        if logger is not None:
            logger.info('Tokenized examples:')
            for ex in s.examples[:10]:
                logger.info('Context: ' + ' '.join(ex.context))
                logger.info('Question: ' + ' '.join(ex.question))
                logger.info(' '.join(ex.context_question))
                logger.info('Answer: ' + ' '.join(ex.answer))



def set_seed(args, rank=None):
    if rank is None and len(args.devices) > 0:
        ordinal = args.devices[0]
    else:
        ordinal = args.devices[rank] 
    device = torch.device(f'cuda:{ordinal}' if ordinal > -1 else 'cpu')
    print(f'device: {device}')
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    with torch.cuda.device(ordinal):
        torch.cuda.manual_seed(args.seed)
    return device


def count_params(params):
    def mult(ps):
        r = 0
        for p in ps:
            this_r = 1
            for s in p.size():
                this_r *= s
            r += this_r
        return r
    return mult(params)


def get_trainable_params(model):
    return list(filter(lambda p: p.requires_grad, model.parameters()))


def elapsed_time(log):
    t = time.time() - log.start
    day = int(t // (24 * 3600))
    t = t % (24 * 3600)
    hour = int(t // 3600)
    t %= 3600
    minutes = int(t // 60)
    t %= 60
    seconds = int(t)
    return f'{day:02}:{hour:02}:{minutes:02}:{seconds:02}'


def get_splits(args, task, FIELD, **kwargs):
    if 'multi30k' in task:
        src, trg = ['.'+x for x in task.split('.')[1:]]
        split = torchtext.datasets.generic.Multi30k.splits(exts=(src, trg), 
            fields=FIELD, root=args.data, **kwargs)
    elif 'iwslt' in task:
        src, trg = ['.'+x for x in task.split('.')[1:]]
        split = torchtext.datasets.generic.IWSLT.splits(exts=(src, trg), 
            fields=FIELD, root=args.data, **kwargs)
    elif 'squad' in task:
        split = torchtext.datasets.generic.SQuAD.splits(
            fields=FIELD, root=args.data, description=task, **kwargs)
    elif 'wikisql' in task:
        split = torchtext.datasets.generic.WikiSQL.splits(
            fields=FIELD, root=args.data, query_as_question='query_as_question' in task, **kwargs)
    elif 'ontonotes.ner' in task:
        split_task = task.split('.')
        _, _, subtask, nones, counting = split_task
        split = torchtext.datasets.generic.OntoNotesNER.splits(
            subtask=subtask, nones=True if nones == 'nones' else False,
            fields=FIELD, root=args.data, **kwargs)
    elif 'woz' in task:
        split = torchtext.datasets.generic.WOZ.splits(description=task,
            fields=FIELD, root=args.data, **kwargs)
    elif 'multinli' in task:
        split = torchtext.datasets.generic.MultiNLI.splits(description=task,
            fields=FIELD, root=args.data, **kwargs)
    elif 'srl' in task:
        split = torchtext.datasets.generic.SRL.splits(
            fields=FIELD, root=args.data, **kwargs)
    elif 'snli' in task:
        split = torchtext.datasets.generic.SNLI.splits(
            fields=FIELD, root=args.data, **kwargs)
    elif 'schema' in task:
        split = torchtext.datasets.generic.WinogradSchema.splits(
            fields=FIELD, root=args.data, **kwargs)
    elif task == 'cnn':
        split = torchtext.datasets.generic.CNN.splits(
            fields=FIELD, root=args.data, **kwargs)
    elif task == 'dailymail':
        split = torchtext.datasets.generic.DailyMail.splits(
            fields=FIELD, root=args.data, **kwargs)
    elif task == 'cnn_dailymail':
        split_cnn = torchtext.datasets.generic.CNN.splits(
            fields=FIELD, root=args.data, **kwargs)
        split_dm = torchtext.datasets.generic.DailyMail.splits(
            fields=FIELD, root=args.data, **kwargs)
        for scnn, sdm in zip(split_cnn, split_dm):
            scnn.examples.extend(sdm)
        split = split_cnn
    elif 'sst' in task:
        split = torchtext.datasets.generic.SST.splits(
            fields=FIELD, root=args.data, **kwargs)
    elif 'imdb' in task:
        kwargs['validation'] = None
        split = torchtext.datasets.generic.IMDb.splits(
            fields=FIELD, root=args.data, **kwargs)
    elif 'zre' in task:
        split = torchtext.datasets.generic.ZeroShotRE.splits(
            fields=FIELD, root=args.data, **kwargs)
    elif os.path.exists(os.path.join(args.data, task)):
        split = torchtext.datasets.generic.JSON.splits(
            fields=FIELD, root=args.data, name=task, **kwargs)
    return split


def batch_fn(new, i, sofar):
    prev_max_len = sofar / (i - 1) if i > 1 else 0
    return max(len(new.context), 5*len(new.answer), prev_max_len) * i


def pad(x, new_channel, dim, val=None):
    if x.size(dim) > new_channel:
        x = x.narrow(dim, 0, new_channel)
    channels = x.size()
    assert (new_channel >= channels[dim])
    if new_channel == channels[dim]:
        return x
    size = list(channels)
    size[dim] = new_channel - size[dim]
    padding = x.new(*size).fill_(val)
    return torch.cat([x, padding], dim)
