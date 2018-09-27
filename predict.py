#!/usr/bin/env python3
import os
from text.torchtext.datasets.generic import Query
from text import torchtext
from argparse import ArgumentParser
import ujson as json
import torch
import numpy as np
import random
from pprint import pformat

from util import get_splits, set_seed, preprocess_examples
from metrics import compute_metrics
import models


def get_all_splits(args, new_vocab):
    splits = []
    for task in args.tasks:
        print(f'Loading {task}')
        kwargs = {}
        if not 'train' in args.evaluate:
            kwargs['train'] =  None
        if not 'valid' in  args.evaluate:
            kwargs['validation'] =  None
        if not 'test' in args.evaluate:
            kwargs['test'] =  None
        s = get_splits(args, task, new_vocab, **kwargs)[0]
        preprocess_examples(args, [task], [s], new_vocab, train=False)
        splits.append(s)
    return splits


def prepare_data(args, FIELD):
    new_vocab = torchtext.data.ReversibleField(batch_first=True, init_token='<init>', eos_token='<eos>', lower=args.lower, include_lengths=True)
    splits = get_all_splits(args, new_vocab)
    new_vocab.build_vocab(*splits)
    print(f'Vocabulary has {len(FIELD.vocab)} tokens from training')
    args.max_generative_vocab = min(len(FIELD.vocab), args.max_generative_vocab)
    FIELD.append_vocab(new_vocab)
    print(f'Vocabulary has expanded to {len(FIELD.vocab)} tokens')

    char_vectors = torchtext.vocab.CharNGram(cache=args.embeddings)
    glove_vectors = torchtext.vocab.GloVe(cache=args.embeddings)
    vectors = [char_vectors, glove_vectors]
    FIELD.vocab.load_vectors(vectors, True)
    FIELD.decoder_to_vocab = {idx: FIELD.vocab.stoi[word] for idx, word in enumerate(FIELD.decoder_itos)}
    FIELD.vocab_to_decoder = {idx: FIELD.decoder_stoi[word] for idx, word in enumerate(FIELD.vocab.itos) if word in FIELD.decoder_stoi}
    splits = get_all_splits(args, FIELD)

    return FIELD, splits


def to_iter(data, bs):
    Iterator = torchtext.data.Iterator
    it = Iterator(data, batch_size=bs, 
       device=0, batch_size_fn=None, 
       train=False, repeat=False, sort=None, 
       shuffle=None, reverse=False)

    return it


def run(args, field, val_sets, model):
    set_seed(args)
    print(f'Preparing iterators')
    iters = [(name, to_iter(x, bs)) for name, x, bs in zip(args.tasks, val_sets, args.val_batch_size)]
 
    def mult(ps):
        r = 0
        for p in ps:
            this_r = 1
            for s in p.size():
                this_r *= s
            r += this_r
        return r
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    num_param = mult(params)
    print(f'{args.model} has {num_param:,} parameters')
    if args.gpus > -1:
        model.cuda()

    model.eval()
    for task, it in iters:
        prediction_file_name = os.path.join(os.path.splitext(args.best_checkpoint)[0], args.evaluate, task + '.txt')
        answer_file_name = os.path.join(os.path.splitext(args.best_checkpoint)[0], args.evaluate, task + '.gold.txt')
        results_file_name = answer_file_name.replace('gold', 'results')
        if 'sql' in task:
            ids_file_name = answer_file_name.replace('gold', 'ids')
        if os.path.exists(prediction_file_name):
            print('** ', prediction_file_name, ' already exists -- this is where predictions are stored **')
        if os.path.exists(answer_file_name):
            print('** ', answer_file_name, ' already exists -- this is where ground truth answers are stored **')
        if os.path.exists(results_file_name):
            print('** ', results_file_name, ' already exists -- this is where metrics are stored **')
            with open(results_file_name) as results_file:
              for l in results_file:
                  print(l)
            if not 'schema' in task and not args.overwrite_predictions and args.silent:
                continue
        for x in [prediction_file_name, answer_file_name, results_file_name]:
            os.makedirs(os.path.dirname(x), exist_ok=True)

        if not os.path.exists(prediction_file_name) or args.overwrite_predictions:
            with open(prediction_file_name, 'w') as prediction_file:
                predictions = []
                wikisql_ids = []
                for batch_idx, batch in enumerate(it):
                    _, p = model(batch)
                    p = field.reverse(p)
                    for i, pp in enumerate(p):
                        if 'sql' in task:
                            wikisql_id = int(batch.wikisql_id[i])
                            wikisql_ids.append(wikisql_id)
                        prediction_file.write(pp + '\n')
                        predictions.append(pp) 
        else:
            with open(prediction_file_name) as prediction_file:
                predictions = [x.strip() for x in prediction_file.readlines()] 

        if 'sql' in task:
            with open(ids_file_name, 'w') as id_file:
                for i in wikisql_ids:
                    id_file.write(json.dumps(i) + '\n')

        def from_all_answers(an):
            return [it.dataset.all_answers[sid] for sid in an.tolist()] 

        if not os.path.exists(answer_file_name):
            with open(answer_file_name, 'w') as answer_file:
                answers = []
                for batch_idx, batch in enumerate(it):
                    if hasattr(batch, 'wikisql_id'):
                        a = from_all_answers(batch.wikisql_id.data.cpu())
                    elif hasattr(batch, 'squad_id'):
                        a = from_all_answers(batch.squad_id.data.cpu())
                    elif hasattr(batch, 'woz_id'):
                        a = from_all_answers(batch.woz_id.data.cpu())
                    else:
                        a = field.reverse(batch.answer.data)
                    for aa in a:
                        answers.append(aa) 
                        answer_file.write(json.dumps(aa) + '\n')
        else:
            with open(answer_file_name) as answer_file:
                answers = [json.loads(x.strip()) for x in answer_file.readlines()] 

        if len(answers) > 0:
            if not os.path.exists(results_file_name):
                metrics, answers = compute_metrics(predictions, answers, bleu='iwslt' in task or 'multi30k' in task or args.bleu, dialogue='woz' in task,
                    rouge='cnn' in task or 'dailymail' in task or args.rouge, logical_form='sql' in task, corpus_f1='zre' in task, args=args)
                with open(results_file_name, 'w') as results_file:
                    results_file.write(json.dumps(metrics) + '\n')
            else:
                with open(results_file_name) as results_file:
                    metrics = json.loads(results_file.readlines()[0])

            print(metrics)
            if not args.silent:
                for i, (p, a) in enumerate(zip(predictions, answers)):
                    print(f'Prediction {i+1}: {p}\nAnswer {i+1}: {a}\n')


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--evaluate', type=str, required=True)
    parser.add_argument('--tasks', default=['wikisql', 'woz.en', 'cnn_dailymail', 'iwslt.en.de', 'zre', 'srl', 'squad', 'sst', 'multinli.in.out', 'schema'], nargs='+')
    parser.add_argument('--gpus', type=int, help='gpus to use', required=True)
    parser.add_argument('--seed', default=123, type=int, help='Random seed.')
    parser.add_argument('--data', default='/decaNLP/.data/', type=str, help='where to load data from.')
    parser.add_argument('--embeddings', default='/decaNLP/.embeddings', type=str, help='where to save embeddings.')
    parser.add_argument('--checkpoint_name')
    parser.add_argument('--bleu', action='store_true', help='whether to use the bleu metric (always on for iwslt)')
    parser.add_argument('--rouge', action='store_true', help='whether to use the bleu metric (always on for cnn, dailymail, and cnn_dailymail)')
    parser.add_argument('--overwrite_predictions', action='store_true', help='whether to overwrite previously written predictions')
    parser.add_argument('--silent', action='store_true', help='whether to print predictions to stdout')

    args = parser.parse_args()

    with open(os.path.join(args.path, 'config.json')) as config_file:
        config = json.load(config_file)
        retrieve = ['model', 'val_batch_size',
                    'transformer_layers', 'rnn_layers', 'transformer_hidden', 
                    'dimension', 'load', 'max_val_context_length', 'val_batch_size', 
                    'transformer_heads', 'max_output_length', 'max_generative_vocab', 
                    'lower', 'cove', 'intermediate_cove']
        for r in retrieve:
            if r in config:
                setattr(args, r,  config[r])
            else:
                setattr(args, r, None)
        args.dropout_ratio = 0.0

    args.task_to_metric = {'cnn_dailymail': 'avg_rouge',
        'iwslt.en.de': 'bleu',
        'multinli.in.out': 'em',
        'squad': 'nf1',
        'srl': 'nf1',
        'sst': 'em',
        'wikisql': 'lfem',
        'woz.en': 'joint_goal_em',
        'zre': 'corpus_f1',
        'schema': 'em'}

    if os.path.exists(os.path.join(args.path, 'process_0.log')):
        args.best_checkpoint = get_best(args)
    else:
        args.best_checkpoint = os.path.join(args.path, args.checkpoint_name)
           
    return args


def get_best(args):
    with open(os.path.join(args.path, 'config.json')) as f:
        save_every = json.load(f)['save_every']
    
    with open(os.path.join(args.path, 'process_0.log')) as f:
        lines = f.readlines()

    best_score = 0
    best_it = 0
    deca_scores = {}
    for l in lines:
        if 'val' in l:
            try:
                task = l.split('val_')[1].split(':')[0]
            except Exception as e:
                print(e)
                continue
            it = int(l.split('iteration_')[1].split(':')[0])
            metric = args.task_to_metric[task]
            score = float(l.split(metric+'_')[1].split(':')[0])
            if it in deca_scores:
                deca_scores[it]['deca'] += score
                deca_scores[it][metric] = score
            else:
                deca_scores[it] = {'deca': score, metric: score}
            if deca_scores[it]['deca'] > best_score:
                best_score = deca_scores[it]['deca']
                best_it = it
    print(best_it)
    print(best_score)
    return os.path.join(args.path, f'iteration_{int(best_it)}.pth')


if __name__ == '__main__':
    args = get_args()
    print(f'Arguments:\n{pformat(vars(args))}')
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpus}'

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print(f'Loading from {args.best_checkpoint}')
    save_dict = torch.load(args.best_checkpoint)
    field = save_dict['field']
    print(f'Initializing Model')
    Model = getattr(models, args.model) 
    model = Model(field, args)
    model.load_state_dict(save_dict['model_state_dict'])
    field, splits = prepare_data(args, field)
    model.set_embeddings(field.vocab.vectors)

    run(args, field, splits, model)
