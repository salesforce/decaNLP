#!/usr/bin/env python3
from text.torchtext.datasets.generic import Query
from argparse import ArgumentParser
import os
import re
import ujson as json
from metrics import to_lf


def correct_format(x):
    if len(x.keys()) == 0:
        x = {'query': None, 'error': 'Invalid'}
    else: 
        c = x['conds']
        proper = True
        for cc in c:
           if len(cc) < 3:
               proper = False
        if proper: 
            x = {'query': x, 'error': ''} 
        else:
            x = {'query': None, 'error': 'Invalid'}
    return x


def write_logical_forms(greedy, args):
    data_dir = os.path.join(args.data, 'wikisql', 'data')
    path = os.path.join(data_dir, 'dev.jsonl') if 'valid' in args.evaluate else os.path.join(data_dir, 'test.jsonl')
    table_path = os.path.join(data_dir, 'dev.tables.jsonl') if 'valid' in args.evaluate else os.path.join(data_dir, 'test.tables.jsonl')
    with open(table_path) as tables_file:
        tables = [json.loads(line) for line in tables_file]
        id_to_tables = {x['id']: x for x in tables}

    examples = []
    with open(path) as example_file:
        for line in example_file:
            entry = json.loads(line)
            table = id_to_tables[entry['table_id']]
            sql = entry['sql']
            header = table['header']
            a = repr(Query.from_dict(entry['sql'], table['header']))
            ex = {'sql': sql, 'header': header, 'answer': a, 'table': table}
            examples.append(ex)

    with open(args.output, 'a') as f:
        count = 0
        correct = 0
        text_answers = []
        for idx, (g, ex) in enumerate(zip(greedy, examples)):
            count += 1
            text_answers.append([ex['answer'].lower()])
            try:
                lf = to_lf(g, ex['table'])
                f.write(json.dumps(correct_format(lf)) + '\n')
                gt = ex['sql']
                conds = gt['conds']
                lower_conds = []
                for c in conds:
                    lc = c
                    lc[2] = str(lc[2]).lower()
                    lower_conds.append(lc)
                gt['conds'] = lower_conds
                correct += lf == gt
            except Exception as e:
                f.write(json.dumps(correct_format({})) + '\n')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data', help='path to the directory containing data for WikiSQL')
    parser.add_argument('predictions', help='path to prediction file, containing one prediction per line')
    parser.add_argument('ids', help='path to file for indices, a list of integers indicating the index into the dev/test set of the predictions on the corresponding line in \'predicitons\'')
    parser.add_argument('output', help='path for logical forms output line by line')
    parser.add_argument('evaluate', help='running on the \'validation\' or \'test\' set')
    args = parser.parse_args()
    with open(args.predictions) as f:
        greedy = [l for l in f]
    if args.ids is not None:
        with open(args.ids) as f:
            ids = [int(l.strip()) for l in f]
        greedy = [x[1] for x in sorted([(i, g) for i, g in zip(ids, greedy)])]
    write_logical_forms(greedy, args)
