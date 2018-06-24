#!/usr/bin/env python3

import os
import math
import time
import random
import collections
from copy import deepcopy

import logging
from pprint import pformat
from logging import handlers
import ujson as json

import torch
import numpy as np

from text import torchtext

from tensorboardX import SummaryWriter
import string

import arguments
import models
from validate import validate
from multiprocess import Multiprocess, DistributedDataParallel
from metrics import compute_metrics
from util import elapsed_time, get_splits, batch_fn, set_seed, preprocess_examples, get_trainable_params, count_params


def initialize_logger(args, rank='main'):
    # set up file logger
    logger = logging.getLogger(f'process_{rank}')
    logger.setLevel(logging.DEBUG)
    handler = handlers.RotatingFileHandler(os.path.join(args.log_dir, f'process_{rank}.log'), maxBytes=1024*1024*10, backupCount=1)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log(rank='main'):
    return logging.getLogger(f'process_{rank}')


def prepare_data(args, field, logger):

    if field is None: 
        logger.info(f'Constructing field')
        FIELD = torchtext.data.ReversibleField(batch_first=True, init_token='<init>', eos_token='<eos>', lower=args.lower, include_lengths=True)
    else:
        FIELD = field

    train_sets, val_sets, vocab_sets = [], [], []
    for task in args.train_tasks:
        logger.info(f'Loading {task}')
        kwargs = {'test': None}
        kwargs['subsample'] = args.subsample
        kwargs['validation'] = None
        logger.info(f'Adding {task} to training datasets')
        split = get_splits(args, task, FIELD, **kwargs)[0]
        logger.info(f'{task} has {len(split)} training examples')
        train_sets.append(split)
        if args.vocab_tasks is not None and task in args.vocab_tasks:
            vocab_sets.extend(split)

    for task in args.val_tasks:
        logger.info(f'Loading {task}')
        kwargs = {'test': None}
        kwargs['subsample'] = args.subsample
        kwargs['train'] = None
        logger.info(f'Adding {task} to validation datasets')
        split = get_splits(args, task, FIELD, **kwargs)[0]
        logger.info(f'{task} has {len(split)} validation examples')
        val_sets.append(split)
        if args.vocab_tasks is not None and task in args.vocab_tasks:
            vocab_sets.extend(split) 

    if args.load is None:
        logger.info(f'Building vocabulary')
        char_vectors = torchtext.vocab.CharNGram(cache=args.embeddings)
        glove_vectors = torchtext.vocab.GloVe(cache=args.embeddings)
        vectors = [char_vectors, glove_vectors]
        vocab_sets = (train_sets + val_sets) if len(vocab_sets) == 0 else vocab_sets
        FIELD.build_vocab(*vocab_sets, max_size=args.max_effective_vocab, vectors=vectors)

    FIELD.decoder_itos = FIELD.vocab.itos[:args.max_generative_vocab]
    FIELD.decoder_stoi = {word: idx for idx, word in enumerate(FIELD.decoder_itos)} 
    FIELD.decoder_to_vocab = {idx: FIELD.vocab.stoi[word] for idx, word in enumerate(FIELD.decoder_itos)}
    FIELD.vocab_to_decoder = {idx: FIELD.decoder_stoi[word] for idx, word in enumerate(FIELD.vocab.itos) if word in FIELD.decoder_stoi}

    logger.info(f'Vocabulary has {len(FIELD.vocab)} tokens')
    logger.info(f'The first 500 tokens:')
    print(FIELD.vocab.itos[:500])

    logger.info('Preprocessing training data')
    preprocess_examples(args, args.train_tasks, train_sets, FIELD, logger, train=True) 
    logger.info('Preprocessing validation data')
    preprocess_examples(args, args.val_tasks, val_sets, FIELD, logger, train=args.val_filter)

    return FIELD, train_sets, val_sets


def to_iter(args, world_size, val_batch_size, data, train=True, token_testing=False, sort=None):
    sort = sort if not token_testing else True
    shuffle = None if not token_testing else False
    reverse = args.reverse
    Iterator = torchtext.data.BucketIterator if train else torchtext.data.Iterator
    it = Iterator(data, batch_size=val_batch_size, 
       device=0 if world_size > 0 else -1, batch_size_fn=batch_fn if train else None, 
       distributed=world_size>1, train=train, repeat=train, sort=sort, 
       shuffle=shuffle, reverse=args.reverse)
    return it


def get_learning_rate(i):
    return 0.1 * 10 / math.sqrt(args.dimension) * min(
        1 / math.sqrt(i), i / (args.warmup * math.sqrt(args.warmup)))


def step(model, batch, opt, iteration, field, task, lr=None, grad_clip=None, writer=None, it=None):
    model.train()
    opt.zero_grad()
    loss, predictions = model(batch)
    loss.backward()
    if lr is not None:
        opt.param_groups[0]['lr'] = lr
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm(model.params, grad_clip)
    opt.step()
    return loss.data[0], {}


def train(args, model, opt, train_iters, train_iterations, field, rank=0, world_size=1, 
    log_every=10, val_every=100, save_every=1000, rounds=False, val_iters=[], writer=None, start_iteration=1, rnd=1):
    """main training function"""

    logger = log(rank) 
    local_loss, num_examples, len_contexts, len_answers, iteration = 0, 0, 0, 0, start_iteration

    train_iter_deep = deepcopy(train_iterations)
    local_train_metric_dict = {}

    train_iters = [(task, iter(train_iter)) for task, train_iter in train_iters]
    while True:

        # For some number of rounds, we 'jump start' some subset of the tasks
        # by training them and not others
        # once the specified number of rounds is completed, 
        # switch to normal round robin training
        if rnd<args.jump_start:
            train_iterations = [0]*len(train_iterations)
            for _ in range(args.n_jump_start): train_iterations[_] = 1
        else:
            train_iterations = train_iter_deep

        for task_idx, (task, train_iter) in enumerate(train_iters):
            task_best_metrics = {}
            task_iterations = train_iterations[task_idx] if train_iterations is not None else None
            if task_iterations == 0:
                continue
            task_iteration = 1
            for batch in train_iter:
                if not args.resume or iteration > start_iteration:
                    task_progress = f'{task_iteration}/{task_iterations}:' if task_iterations is not None else ''
                    round_progress = f'round_{rnd}:' if rounds else ''
    
                    # validate
                    if (val_every is not None and 
                        ((iteration % args.val_every == 0 % args.val_every) or 
                            (args.load and iteration == start_iteration + 1))):
                        train_task_val_metric = None
                        for val_task_idx, (val_task, val_iter) in enumerate(val_iters):
                            val_loss, metric_dict = validate(val_task, val_iter, model, logger, field, world_size, rank, num_print=args.num_print, args=args)
                            if val_loss is not None:
                                log_entry = f'{args.timestamp}:{elapsed_time(logger)}:iteration_{iteration}:{round_progress}train_{task}:{task_progress}val_{val_task}:val_loss{val_loss.data[0]:.4f}:'
                                writer.add_scalars(f'loss/val', {val_task: val_loss.data[0]}, iteration)
                            else:
                                log_entry = f'{args.timestamp}:{elapsed_time(logger)}:iteration_{iteration}:{round_progress}train_{task}:{task_progress}val_{val_task}:'
                               
                            metric_entry = ''
                            for metric_key, metric_value in metric_dict.items():
                                metric_entry += f'{metric_key}_{metric_value:.2f}:'
                            metric_entry = metric_entry[:-1]
                           
                            # val log
                            logger.info(log_entry + metric_entry)
                            if writer is not None:
                                for metric_key, metric_value in metric_dict.items():
                                    writer.add_scalars(f'val/{metric_key}', {val_task: metric_value}, iteration)
                                    writer.add_scalars(f'{metric_key}/val', {val_task: metric_value}, iteration)
                                    writer.add_scalars(f'{metric_key}/{val_task}', {'val': metric_value}, iteration)
                                    writer.add_scalars(f'{val_task}/{metric_key}', {'val': metric_value}, iteration)
                                    writer.add_scalars(f'{val_task}/val', {f'{metric_key}': metric_value}, iteration)

                        # saving
                        if save_every is not None and (iteration % args.save_every == 0 % args.save_every):
                            if world_size > 1:
                                torch.distributed.barrier() 
                            if rank is not None and rank == 0:
                                torch.save({'model_state_dict': model.state_dict(), 'field': field}, os.path.join(args.log_dir, f'iteration_{iteration}.pth'))
                            if world_size > 1:
                                torch.distributed.barrier() 
                            torch.save(opt.state_dict(), os.path.join(args.log_dir, f'iteration_{iteration}_rank_{rank}_optim.pth'))
                            if world_size > 1:
                                torch.distributed.barrier() 

                    # lr update
                    lr = opt.param_groups[0]['lr'] 
                    if args.warmup > 0 and args.transformer_lr:
                        lr = get_learning_rate(iteration) 

                    # param update
                    loss, train_metric_dict = step(model, batch, opt, iteration, field, task, lr=lr, grad_clip=args.grad_clip, writer=writer, it=train_iter)

                    # train metrics
                    local_loss += loss
                    for metric_name, metric_val in train_metric_dict.items():
                        if metric_name in local_train_metric_dict:
                            local_train_metric_dict[metric_name] += metric_val / args.log_every
                        else:
                            local_train_metric_dict[metric_name] = metric_val / args.log_every

                    # train logs
                    num_examples += batch.context.size(0)
                    len_contexts += batch.context.size(1)
                    len_answers += batch.answer.size(1)

                    if log_every is not None and (iteration % log_every == 0 % log_every):
                        local_loss /= args.log_every
                        num_examples /= args.log_every
                        len_contexts /= args.log_every
                        len_answers /= args.log_every
                        avg_batch_size = f'avbatch_{num_examples:.0f}_{len_contexts:.0f}_{len_answers:.0f}:'
                        metric_entry = ''
                        for metric_key, metric_value in local_train_metric_dict.items():
                            metric_entry += f'{metric_key}_{metric_value:.2f}:'
                        metric_entry = f'{metric_entry[:-1]}'
                        logger.info(f'{args.timestamp}:{elapsed_time(logger)}:iteration_{iteration}:{round_progress}train_{task}:{task_progress}{avg_batch_size}loss_{local_loss:.4f}{metric_entry}') 
                        num_examples = 0 
                        len_contexts = 0 
                        len_answers = 0  
    
                        if writer is not None:
                            writer.add_scalars(f'loss/train', {f'{task}': local_loss}, iteration)
                            for metric_key, metric_value in local_train_metric_dict.items():
                                writer.add_scalars(f'train/{metric_key}', {task: metric_value}, iteration)
                                writer.add_scalars(f'{metric_key}/train', {task: metric_value}, iteration)
                                writer.add_scalars(f'{metric_key}/{task}', {'train': metric_value}, iteration)
                                writer.add_scalars(f'{task}/{metric_key}', {'train': metric_value}, iteration)
                                writer.add_scalars(f'{task}/train', {f'{metric_key}': metric_value}, iteration)


                        local_loss = 0
                        local_train_metric_dict = {}
                        num_examples = 0
                    
                # book keeping
                task_iteration += 1
                iteration += 1
                if task_iterations is not None and task_iteration > task_iterations:
                    break

        # book keeping
        rnd += 1
        if not rounds:
            break


def run(args, run_args, rank=0, world_size=1):
    set_seed(args, rank=rank)
    logger = initialize_logger(args, rank)
    field, train_sets, val_sets, save_dict = run_args

    logger.start = time.time()

    logger.info(f'Preparing iterators')
    train_iters = [(name, to_iter(args, world_size, tok, x, token_testing=args.token_testing)) 
                      for name, x, tok in zip(args.train_tasks, train_sets, args.train_batch_tokens)]
    val_iters = [(name, to_iter(args, world_size, tok, x, train=False, token_testing=args.token_testing, sort=False if 'sql' in name else None))
                    for name, x, tok in zip(args.val_tasks, val_sets, args.val_batch_size)]

    logger.info(f'Initializing Writer')
    writer = SummaryWriter(log_dir=args.log_dir)

    model = init_model(args, field, logger, world_size)
    opt = init_opt(args, model) 
    start_iteration = 1

    if save_dict is not None:
        logger.info(f'Loading model from {os.path.join(args.save, args.load)}')
        save_dict = torch.load(os.path.join(args.save, args.load))
        model.load_state_dict(save_dict['model_state_dict'])
        if args.resume:
            logger.info(f'Resuming Training from {os.path.splitext(args.load)[0]}_rank_{rank}_optim.pth')
            opt.load_state_dict(torch.load(os.path.join(args.save, f'{os.path.splitext(args.load)[0]}_rank_{rank}_optim.pth')))
            start_iteration = int(os.path.splitext(os.path.basename(args.load))[0].split('_')[1])

    logger.info(f'Begin Training')
    train(args, model, opt, train_iters, args.train_iterations, field, val_iters=val_iters, 
        rank=rank, world_size=world_size, 
        log_every=args.log_every, val_every=args.val_every, rounds=len(train_iters)>1,
        writer=writer if rank==0 else None, save_every=args.save_every, start_iteration=start_iteration)


def init_model(args, field, logger, world_size):
    logger.info(f'Initializing {args.model}')
    Model = getattr(models, args.model) 
    model = Model(field, args)
    params = get_trainable_params(model) 
    num_param = count_params(params)
    logger.info(f'{args.model} has {num_param:,} parameters')

    if args.gpus[0] > -1:
        model.cuda()
    if world_size > 1: 
        logger.info(f'Wrapping model for distributed')
        model = DistributedDataParallel(model)

    model.params = params
    return model


def init_opt(args, model):
    opt = None
    if args.transformer_lr:
        opt = torch.optim.Adam(model.params, betas=(0.9, 0.98), eps=1e-9)
    else:
        opt = torch.optim.Adam(model.params, betas=(args.beta0, 0.999))
    return opt


if __name__ == '__main__':
    args = arguments.parse()
    set_seed(args)
    logger = initialize_logger(args)
    logger.info(f'Arguments:\n{pformat(vars(args))}')

    field, save_dict = None, None
    if args.load is not None:
        logger.info(f'Loading field from {os.path.join(args.save, args.load)}')
        save_dict = torch.load(os.path.join(args.save, args.load))
        field = save_dict['field']
    field, train_sets, val_sets = prepare_data(args, field, logger)

    run_args = (field, train_sets, val_sets, save_dict)
    if len(args.gpus) > 1:
        logger.info(f'Multiprocessing')
        mp = Multiprocess(run, args)
        mp.run(run_args)
    else:
        logger.info(f'Processing')
        run(args, run_args, world_size=args.world_size)
