import os
from copy import deepcopy
import types
import sys
from argparse import ArgumentParser
import subprocess
import json
import datetime
from dateutil import tz


def get_commit():
    directory = os.path.dirname(sys.argv[0])
    return subprocess.Popen("cd {} && git log | head -n 1".format(directory), shell=True, stdout=subprocess.PIPE).stdout.read().split()[1].decode()


def save_args(args):
    os.makedirs(args.log_dir, exist_ok=args.exist_ok)
    with open(os.path.join(args.log_dir, 'config.json'), 'wt') as f:
        json.dump(vars(args), f, indent=2)


def parse():
    """
    Returns the arguments from the command line.
    """
    parser = ArgumentParser()
    parser.add_argument('--root', default='/decaNLP', type=str, help='root directory for data, results, embeddings, code, etc.')
    parser.add_argument('--data', default='.data/', type=str, help='where to load data from.')
    parser.add_argument('--save', default='results', type=str, help='where to save results.')
    parser.add_argument('--embeddings', default='.embeddings', type=str, help='where to save embeddings.')
    parser.add_argument('--name', default='', type=str, help='name of the experiment; if blank, a name is automatically generated from the arguments')

    parser.add_argument('--train_tasks', nargs='+', type=str, help='tasks to use for training', required=True)
    parser.add_argument('--train_iterations', nargs='+', type=int, help='number of iterations to focus on each task')
    parser.add_argument('--train_batch_tokens', nargs='+', default=[9000], type=int, help='Number of tokens to use for dynamic batching, corresponging to tasks in train tasks')
    parser.add_argument('--jump_start', default=0, type=int, help='number of iterations to give jump started tasks')
    parser.add_argument('--n_jump_start', default=0, type=int, help='how many tasks to jump start (presented in order)')    
    parser.add_argument('--num_print', default=15, type=int, help='how many validation examples with greedy output to print to std out')

    parser.add_argument('--no_tensorboard', action='store_false', dest='tensorboard', help='Turn of tensorboard logging') 
    parser.add_argument('--log_every', default=int(1e2), type=int, help='how often to log results in # of iterations')
    parser.add_argument('--save_every', default=int(1e3), type=int, help='how often to save a checkpoint in # of iterations')

    parser.add_argument('--val_tasks', nargs='+', type=str, help='tasks to collect evaluation metrics for')
    parser.add_argument('--val_every', default=int(1e3), type=int, help='how often to run validation in # of iterations')
    parser.add_argument('--val_no_filter', action='store_false', dest='val_filter', help='whether to allow filtering on the validation sets')
    parser.add_argument('--val_batch_size', nargs='+', default=[256], type=int, help='Batch size for validation corresponding to tasks in val tasks')

    parser.add_argument('--vocab_tasks', nargs='+', type=str, help='tasks to use in the construction of the vocabulary')
    parser.add_argument('--max_output_length', default=100, type=int, help='maximum output length for generation')
    parser.add_argument('--max_effective_vocab', default=int(1e6), type=int, help='max effective vocabulary size for pretrained embeddings')
    parser.add_argument('--max_generative_vocab', default=50000, type=int, help='max vocabulary for the generative softmax')
    parser.add_argument('--max_train_context_length', default=400, type=int, help='maximum length of the contexts during training')
    parser.add_argument('--max_val_context_length', default=400, type=int, help='maximum length of the contexts during validation')
    parser.add_argument('--max_answer_length', default=50, type=int, help='maximum length of answers during training and validation')
    parser.add_argument('--subsample', default=20000000, type=int, help='subsample the datasets')
    parser.add_argument('--preserve_case', action='store_false', dest='lower', help='whether to preserve casing for all text')

    parser.add_argument('--model', type=str, default='MultitaskQuestionAnsweringNetwork', help='which model to import')
    parser.add_argument('--dimension', default=200, type=int, help='output dimensions for all layers')
    parser.add_argument('--rnn_layers', default=1, type=int, help='number of layers for RNN modules')
    parser.add_argument('--transformer_layers', default=2, type=int, help='number of layers for transformer modules')
    parser.add_argument('--transformer_hidden', default=150, type=int, help='hidden size of the transformer modules')
    parser.add_argument('--transformer_heads', default=3, type=int, help='number of heads for transformer modules')
    parser.add_argument('--dropout_ratio', default=0.2, type=float, help='dropout for the model')
    parser.add_argument('--no_transformer_lr', action='store_false', dest='transformer_lr', help='turns off the transformer learning rate strategy') 
    parser.add_argument('--cove', action='store_true', help='whether to use contextualized word vectors (McCann et al. 2017)')
    parser.add_argument('--intermediate_cove', action='store_true', help='whether to use the intermediate layers of contextualized word vectors (McCann et al. 2017)')
    parser.add_argument('--elmo', default=[-1], nargs='+', type=int,  help='which layer(s) (0, 1, or 2) of ELMo (Peters et al. 2018) to use; -1 for none ')
    parser.add_argument('--no_glove_and_char', action='store_false', dest='glove_and_char', help='turn off GloVe and CharNGram embeddings')

    parser.add_argument('--warmup', default=800, type=int, help='warmup for learning rate')
    parser.add_argument('--grad_clip', default=1.0, type=float, help='gradient clipping')
    parser.add_argument('--beta0', default=0.9, type=float, help='alternative momentum for Adam (only when not using transformer_lr)')

    parser.add_argument('--load', default=None, type=str, help='path to checkpoint to load model from inside args.save')
    parser.add_argument('--resume', action='store_true', help='whether to resume training with past optimizers')

    parser.add_argument('--seed', default=123, type=int, help='Random seed.')
    parser.add_argument('--devices', default=[0], nargs='+', type=int, help='a list of devices that can be used for training (multi-gpu currently WIP)')
    parser.add_argument('--backend', default='gloo', type=str, help='backend for distributed training')

    parser.add_argument('--no_commit', action='store_false', dest='commit', help='do not track the git commit associated with this training run') 
    parser.add_argument('--exist_ok', action='store_true', help='Ok if the save directory already exists, i.e. overwrite is ok') 
    parser.add_argument('--token_testing', action='store_true', help='if true, sorts all iterators') 
    parser.add_argument('--reverse', action='store_true', help='if token_testing and true, sorts all iterators in reverse') 

    args = parser.parse_args()
    if args.model is None:
        args.model = 'mcqa'
    if args.val_tasks is None:
        args.val_tasks = []
        for t in args.train_tasks:
            if t not in args.val_tasks:
                args.val_tasks.append(t)

    if 'imdb' in args.val_tasks:
        args.val_tasks.remove('imdb')
    args.world_size = len(args.devices) if args.devices[0] > -1 else -1
    if args.world_size > 1:
        print('multi-gpu training is currently a work in progress')
        return
    args.timestamp = '-'.join(datetime.datetime.now(tz=tz.tzoffset(None, -8*60*60)).strftime("%y/%m/%d/%H/%M/%S.%f").split())

    if len(args.train_tasks) > 1:
        if args.train_iterations is  None:
            args.train_iterations = [1]
        if len(args.train_iterations) < len(args.train_tasks):
            args.train_iterations = len(args.train_tasks) * args.train_iterations
        if len(args.train_batch_tokens) < len(args.train_tasks):
            args.train_batch_tokens = len(args.train_tasks) * args.train_batch_tokens
    if len(args.val_batch_size) < len(args.val_tasks):
        args.val_batch_size = len(args.val_tasks) * args.val_batch_size
        
    # postprocess arguments
    if args.commit:
        args.commit = get_commit()
    else:
        args.commit = ''
    train_out = f'{",".join(args.train_tasks)}'
    if len(args.train_tasks) > 1:
        train_out += f'{"-".join([str(x) for x in args.train_iterations])}'
    args.log_dir = os.path.join(args.save, args.timestamp,
        f'{train_out}{(",val=" + ",".join(args.val_tasks)) if args.val_tasks != args.train_tasks else ""},{args.model},' \
        f'{args.world_size}g',
        args.commit[:7])
    if len(args.name) > 0:
        args.log_dir = os.path.join(args.save, args.name)
    args.dist_sync_file = os.path.join(args.log_dir, 'distributed_sync_file')
    
    save_args(args)
    for x in ['data', 'save', 'embeddings']:
        setattr(args, x, os.path.join(args.root, getattr(args, x)))

    return args
