import os
import re
import zipfile
import revtok
import torch
import io
import csv
import json
import glob
import hashlib
import unicodedata


from . import sst
from . import imdb
from . import snli
from . import translation
from . import mood

from .. import data


CONTEXT_SPECIAL = 'Context:'
QUESTION_SPECIAL = 'Question:'


def get_context_question(context, question):
    return CONTEXT_SPECIAL +  ' ' + context + ' ' + QUESTION_SPECIAL + ' ' + question


class CQA(data.Dataset):

    fields = ['context', 'question', 'answer', 'context_special', 'question_special', 'context_question']

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.context), len(ex.answer))
 

class IMDb(CQA, imdb.IMDb):

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.context), len(ex.answer))

    def __init__(self, path, field, subsample=None, **kwargs):
        fields = [(x, field) for x in self.fields]
        examples = []
        labels = {'neg': 'negative', 'pos': 'positive'}
        question = 'Is this review negative or positive?'

        cache_name = os.path.join(os.path.dirname(path), '.cache', os.path.basename(path), str(subsample))
        if os.path.exists(cache_name):
            print(f'Loading cached data from {cache_name}')
            examples = torch.load(cache_name)
        else:
            for label in ['pos', 'neg']:
                for fname in glob.iglob(os.path.join(path, label, '*.txt')):
                    with open(fname, 'r') as f:
                        context = f.readline()
                    answer = labels[label]
                    context_question = get_context_question(context, question) 
                    examples.append(data.Example.fromlist([context, question, answer, CONTEXT_SPECIAL, QUESTION_SPECIAL, context_question], fields))
                    if subsample is not None and len(examples) > subsample:
                        break
            os.makedirs(os.path.dirname(cache_name), exist_ok=True)
            print(f'Caching data to {cache_name}')
            torch.save(examples, cache_name)
        super(imdb.IMDb, self).__init__(examples, fields, **kwargs)


    @classmethod
    def splits(cls, fields, root='.data',
               train='train', validation=None, test='test', **kwargs):
        assert validation is None
        path = cls.download(root)
        train_data = None if train is None else cls(
            os.path.join(path, f'{train}'), fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, f'{test}'), fields, **kwargs)
        return tuple(d for d in (train_data, test_data)
                     if d is not None)


class SST(CQA):

    urls = ['https://raw.githubusercontent.com/openai/generating-reviews-discovering-sentiment/master/data/train_binary_sent.csv',
        'https://raw.githubusercontent.com/openai/generating-reviews-discovering-sentiment/master/data/dev_binary_sent.csv',
        'https://raw.githubusercontent.com/openai/generating-reviews-discovering-sentiment/master/data/test_binary_sent.csv']
    name = 'sst'
    dirname = ''

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.context), len(ex.answer))

    def __init__(self, path, field, subsample=None, **kwargs):
        fields = [(x, field) for x in self.fields]
        cache_name = os.path.join(os.path.dirname(path), '.cache', os.path.basename(path), str(subsample))

        examples = []
        if os.path.exists(cache_name):
            print(f'Loading cached data from {cache_name}')
            examples = torch.load(cache_name)
        else:
            labels = ['negative', 'positive']
            question = 'Is this review ' + labels[0] + ' or ' + labels[1] + '?'

            with io.open(os.path.expanduser(path), encoding='utf8') as f:
                next(f)
                for line in f:
                    parsed = list(csv.reader([line.rstrip('\n')]))[0]
                    context = parsed[-1]
                    answer = labels[int(parsed[0])] 
                    context_question = get_context_question(context, question) 
                    examples.append(data.Example.fromlist([context, question, answer, CONTEXT_SPECIAL, QUESTION_SPECIAL, context_question], fields))

                    if subsample is not None and len(examples) > subsample:
                        break
       
            os.makedirs(os.path.dirname(cache_name), exist_ok=True)
            print(f'Caching data to {cache_name}')
            torch.save(examples, cache_name)

        self.examples = examples
        super().__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, fields, root='.data',
               train='train', validation='dev', test='test', **kwargs):
        path = cls.download(root)
        postfix = f'_binary_sent.csv'
        train_data = None if train is None else cls(
            os.path.join(path, f'{train}{postfix}'), fields, **kwargs)
        validation_data = None if validation is None else cls(
            os.path.join(path, f'{validation}{postfix}'), fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, f'{test}{postfix}'), fields, **kwargs)
        return tuple(d for d in (train_data, validation_data, test_data)
                     if d is not None)


class Mood(CQA):
    
    urls = [] #TODO
    name = 'mood'
    dirname = ''

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.context), len(ex.answer))

    def __init__(self, path, field, subsample=None, **kwargs):
        fields = [(x, field) for x in self.fields]
        cache_name = os.path.join(os.path.dirname(path), '.cache', os.path.basename(path), str(subsample))

        examples = []
        if os.path.exists(cache_name):
            print(f'Loading cached data from {cache_name}')
            examples = torch.load(cache_name)
        else:
            labels = ['extremely angry', 'extremely joyful', 'extremely sad', 'extremely fearful',
                      'fairly angry', 'fairly joyful', 'fairly sad', 'fairly fearful',
                      'slightly angry', 'slightly joyful', 'slightly sad', 'slightly fearful']
            question =  "What’s the tweet’s emotion, angry or fearful or joyful or sad, " \
                        "and what’s the intensity level, slightly or fairly or extremely?"

            with io.open(os.path.expanduser(path), encoding='utf8') as f:
                next(f)
                for line in f:
                    parsed = list(csv.reader([line.rstrip('\n')]))[0]
                    context = parsed[-1]
                    answer = labels[int(parsed[0])]
                    context_question = get_context_question(context, question)
                    examples.append(data.Example.fromlist(
                        [context, question, answer, CONTEXT_SPECIAL, QUESTION_SPECIAL, context_question], fields))

                    if subsample is not None and len(examples) > subsample:
                        break

            os.makedirs(os.path.dirname(cache_name), exist_ok=True)
            print(f'Caching data to {cache_name}')
            torch.save(examples, cache_name)

        self.examples = examples
        super().__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, fields, root='.data',
               train='train', validation='dev', test='test', **kwargs):
        path = cls.download(root)
        postfix = f'_binary_sent.csv'
        train_data = None if train is None else cls(
            os.path.join(path, f'{train}{postfix}'), fields, **kwargs)
        validation_data = None if validation is None else cls(
            os.path.join(path, f'{validation}{postfix}'), fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, f'{test}{postfix}'), fields, **kwargs)
        return tuple(d for d in (train_data, validation_data, test_data)
                     if d is not None)



class TranslationDataset(translation.TranslationDataset):

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.context), len(ex.answer))

    def __init__(self, path, exts, field, subsample=None, **kwargs):
        """Create a TranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields$: fields for handling all columns
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [(x, field) for x in self.fields]
        cache_name = os.path.join(os.path.dirname(path), '.cache', os.path.basename(path), str(subsample))

        if os.path.exists(cache_name):
            print(f'Loading cached data from {cache_name}')
            examples = torch.load(cache_name)
        else:
            langs = {'.de': 'German', '.en': 'English', '.fr': 'French', '.ar': 'Arabic', '.cs': 'Czech'}
            source, target = langs[exts[0]], langs[exts[1]]
            src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)
            question = f'Translate from {source} to {target}'

            examples = []
            with open(src_path) as src_file, open(trg_path) as trg_file:
                for src_line, trg_line in zip(src_file, trg_file):
                    src_line, trg_line = src_line.strip(), trg_line.strip()
                    if src_line != '' and trg_line != '':
                        context = src_line
                        answer = trg_line
                        context_question = get_context_question(context, question) 
                        examples.append(data.Example.fromlist([context, question, answer, CONTEXT_SPECIAL, QUESTION_SPECIAL, context_question], fields))
                        if subsample is not None and len(examples) >= subsample:
                            break


            os.makedirs(os.path.dirname(cache_name), exist_ok=True)
            print(f'Caching data to {cache_name}')
            torch.save(examples, cache_name)
        super(translation.TranslationDataset, self).__init__(examples, fields, **kwargs)


class Multi30k(TranslationDataset, CQA, translation.Multi30k):
    pass


class IWSLT(TranslationDataset, CQA, translation.IWSLT):
    pass


class SQuAD(CQA, data.Dataset):

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.context), len(ex.answer))

    urls = ['https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json',
            'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json',
            'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json',
            'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json',]
    name = 'squad'
    dirname = ''

    def __init__(self, path, field, subsample=None, **kwargs):
        fields = [(x, field) for x in self.fields]
        cache_name = os.path.join(os.path.dirname(path), '.cache', os.path.basename(path), str(subsample))

        examples, all_answers, q_ids = [], [], []
        if os.path.exists(cache_name):
            print(f'Loading cached data from {cache_name}')
            examples, all_answers, q_ids = torch.load(cache_name)
        else:
            with open(os.path.expanduser(path)) as f:
                squad = json.load(f)['data']
                for document in squad:
                    title = document['title']
                    paragraphs = document['paragraphs']
                    for paragraph in paragraphs:
                        context = paragraph['context']
                        qas = paragraph['qas']
                        for qa in qas:
                            question = ' '.join(qa['question'].split())
                            q_ids.append(qa['id'])
                            squad_id = len(all_answers)
                            context_question = get_context_question(context, question) 
                            if len(qa['answers']) == 0:
                                answer = 'unanswerable'
                                all_answers.append(['unanswerable'])
                                context = ' '.join(context.split())
                                ex = data.Example.fromlist([context, question, answer, CONTEXT_SPECIAL, QUESTION_SPECIAL, context_question], fields)
                                ex.context_spans = [-1, -1]
                                ex.answer_start = -1
                                ex.answer_end = -1
                            else:
                                answer = qa['answers'][0]['text']
                                all_answers.append([a['text'] for a in qa['answers']])
                                #print('original: ', answer)
                                answer_start = qa['answers'][0]['answer_start']
                                answer_end = answer_start + len(answer) 
                                context_before_answer = context[:answer_start]
                                context_after_answer = context[answer_end:]
                                BEGIN = 'beginanswer ' 
                                END = ' endanswer'

                                tagged_context = context_before_answer + BEGIN + answer + END + context_after_answer
                                ex = data.Example.fromlist([tagged_context, question, answer, CONTEXT_SPECIAL, QUESTION_SPECIAL, context_question], fields)

                                tokenized_answer = ex.answer
                                #print('tokenized: ', tokenized_answer)
                                for xi, x in enumerate(ex.context):
                                    if BEGIN in x: 
                                        answer_start = xi + 1
                                        ex.context[xi] = x.replace(BEGIN, '')
                                    if END in x: 
                                        answer_end = xi
                                        ex.context[xi] = x.replace(END, '')
                                new_context = []
                                original_answer_start = answer_start
                                original_answer_end = answer_end
                                indexed_with_spaces = ex.context[answer_start:answer_end]
                                if len(indexed_with_spaces) != len(tokenized_answer):
                                    import pdb; pdb.set_trace()

                                # remove spaces
                                for xi, x in enumerate(ex.context):
                                    if len(x.strip()) == 0:
                                        if xi <= original_answer_start:
                                            answer_start -= 1
                                        if xi < original_answer_end:
                                            answer_end -= 1
                                    else:
                                        new_context.append(x)
                                ex.context = new_context
                                ex.answer = [x for x in ex.answer if len(x.strip()) > 0] 
                                if len(ex.context[answer_start:answer_end]) != len(ex.answer):
                                    import pdb; pdb.set_trace()
                                ex.context_spans = list(range(answer_start, answer_end)) 
                                indexed_answer = ex.context[ex.context_spans[0]:ex.context_spans[-1]+1]
                                if len(indexed_answer) != len(ex.answer):
                                    import pdb; pdb.set_trace()
                                if field.eos_token is not None:
                                    ex.context_spans += [len(ex.context)]
                                for context_idx, answer_word in zip(ex.context_spans, ex.answer):
                                    if context_idx == len(ex.context):
                                        continue
                                    if ex.context[context_idx] != answer_word:
                                        import pdb; pdb.set_trace()
                                ex.answer_start = ex.context_spans[0]
                                ex.answer_end = ex.context_spans[-1]
                            ex.squad_id = squad_id
                            examples.append(ex)
                            if subsample is not None and len(examples) > subsample:
                                break
                        if subsample is not None and len(examples) > subsample:
                            break
                    if subsample is not None and len(examples) > subsample:
                        break

            os.makedirs(os.path.dirname(cache_name), exist_ok=True)
            print(f'Caching data to {cache_name}')
            torch.save((examples, all_answers, q_ids), cache_name)


        FIELD = data.Field(batch_first=True, use_vocab=False, sequential=False, 
            lower=False, numerical=True, eos_token=field.eos_token, init_token=field.init_token)
        fields.append(('context_spans', FIELD))
        fields.append(('answer_start', FIELD))
        fields.append(('answer_end', FIELD))
        fields.append(('squad_id', FIELD))

        super(SQuAD, self).__init__(examples, fields, **kwargs)
        self.all_answers = all_answers
        self.q_ids = q_ids


    @classmethod
    def splits(cls, fields, root='.data', description='squad1.1',
               train='train', validation='dev', test=None, **kwargs):
        """Create dataset objects for splits of the SQuAD dataset.
        Arguments:
            root: directory containing SQuAD data
            field: field for handling all columns
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        assert test is None
        path = cls.download(root)

        extension = 'v2.0.json' if '2.0' in description else 'v1.1.json'
        train = '-'.join([train, extension]) if train is not None else None
        validation = '-'.join([validation, extension]) if validation is not None else None

        train_data = None if train is None else cls(
            os.path.join(path, train), fields, **kwargs)
        validation_data = None if validation is None else cls(
            os.path.join(path, validation), fields, **kwargs)
        return tuple(d for d in (train_data, validation_data)
                     if d is not None)


# https://github.com/abisee/cnn-dailymail/blob/8eace60f306dcbab30d1f1d715e379f07a3782db/make_datafiles.py
dm_single_close_quote = u'\u2019'
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@highlight" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  return line + "."


class Summarization(CQA, data.Dataset):

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.context), len(ex.answer))

    def __init__(self, path, field, one_answer=True, subsample=None, **kwargs):
        fields = [(x, field) for x in self.fields]
        cache_name = os.path.join(os.path.dirname(path), '.cache', os.path.basename(path), str(subsample))

        examples = []
        if os.path.exists(cache_name):
            print(f'Loading cached data from {cache_name}')
            examples = torch.load(cache_name)
        else:
            with open(os.path.expanduser(path)) as f:
                lines = f.readlines()
                for line in lines:
                    ex = json.loads(line)
                    context, question, answer = ex['context'], ex['question'], ex['answer']
                    context_question = get_context_question(context, question) 
                    ex = data.Example.fromlist([context, question, answer, CONTEXT_SPECIAL, QUESTION_SPECIAL, context_question], fields)
                    examples.append(ex)
                    if subsample is not None and len(examples) >= subsample: 
                        break
            os.makedirs(os.path.dirname(cache_name), exist_ok=True)
            print(f'Caching data to {cache_name}')
            torch.save(examples, cache_name)

        super(Summarization, self).__init__(examples, fields, **kwargs)

    @classmethod
    def cache_splits(cls, path):

        for split in ['training', 'validation', 'test']:
            missing_stories, collected_stories = 0, 0
            split_file_name = os.path.join(path, f'{split}.jsonl')
            if os.path.exists(split_file_name):
                continue
            with open(split_file_name, 'w') as split_file:
                url_file_name = os.path.join(path, f'{cls.name}_wayback_{split}_urls.txt')
                with open(url_file_name) as url_file:
                    for url in url_file:
                        story_file_name = os.path.join(path, 'stories', 
                            f"{hashlib.sha1(url.strip().encode('utf-8')).hexdigest()}.story")
                        try:
                            story_file = open(story_file_name)
                        except EnvironmentError as e:
                            missing_stories += 1
                            print(e)
                            if os.path.exists(split_file_name):
                                os.remove(split_file_name)
                        else:
                            with story_file:
                                article, highlight = [], []
                                is_highlight = False
                                for line in story_file: 
                                    line = line.strip()
                                    if line == "":
                                        continue
                                    line = fix_missing_period(line)
                                    if line.startswith("@highlight"):
                                        is_highlight = True
                                    elif "@highlight" in line:
                                        raise
                                    elif is_highlight:
                                        highlight.append(line)
                                    else:
                                        article.append(line)
                                example = {'context': unicodedata.normalize('NFKC', ' '.join(article)), 
                                           'answer': unicodedata.normalize('NFKC', ' '.join(highlight)), 
                                           'question': 'What is the summary?'}
                                split_file.write(json.dumps(example)+'\n')
                                collected_stories += 1
                                if collected_stories % 1000 == 0:
                                    print(example) 
            print(f'Missing {missing_stories} stories')
            print(f'Collected {collected_stories} stories')


    @classmethod
    def splits(cls, fields, root='.data',
               train='training', validation='validation', test='test', **kwargs):
        path = cls.download(root)
        cls.cache_splits(path)

        train_data = None if train is None else cls(
            os.path.join(path, 'training.jsonl'), fields, **kwargs)
        validation_data = None if validation is None else cls(
            os.path.join(path, 'validation.jsonl'), fields, one_answer=False, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, 'test.jsonl'), fields, one_answer=False, **kwargs)
        return tuple(d for d in (train_data, validation_data, test_data)
                     if d is not None)


class DailyMail(Summarization):
    name = 'dailymail'
    dirname = 'dailymail'
    urls = [('https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfM1BxdkxVaTY2bWs', 'dailymail_stories.tgz'),
            ('https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/dailymail_wayback_training_urls.txt', 'dailymail/dailymail_wayback_training_urls.txt'),
            ('https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/dailymail_wayback_validation_urls.txt', 'dailymail/dailymail_wayback_validation_urls.txt'),
            ('https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/dailymail_wayback_test_urls.txt', 'dailymail/dailymail_wayback_test_urls.txt')]


class CNN(Summarization):
    name = 'cnn'
    dirname = 'cnn'
    urls = [('https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ', 'cnn_stories.tgz'),
            ('https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/cnn_wayback_training_urls.txt', 'cnn/cnn_wayback_training_urls.txt'),
            ('https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/cnn_wayback_validation_urls.txt', 'cnn/cnn_wayback_validation_urls.txt'),
            ('https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/cnn_wayback_test_urls.txt', 'cnn/cnn_wayback_test_urls.txt')]



class Query:
    #https://github.com/salesforce/WikiSQL/blob/c2ed4f9b22db1cc2721805d53e6e76e07e2ccbdc/lib/query.py#L10

    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']
    syms = ['SELECT', 'WHERE', 'AND', 'COL', 'TABLE', 'CAPTION', 'PAGE', 'SECTION', 'OP', 'COND', 'QUESTION', 'AGG', 'AGGOPS', 'CONDOPS']

    def __init__(self, sel_index, agg_index, columns, conditions=tuple()):
        self.sel_index = sel_index
        self.agg_index = agg_index
        self.columns = columns
        self.conditions = list(conditions)

    def __repr__(self):
        rep = 'SELECT {agg} {sel} FROM table'.format(
            agg=self.agg_ops[self.agg_index],
            sel= self.columns[self.sel_index] if self.columns is not None else 'col{}'.format(self.sel_index),
        )
        if self.conditions:
            rep +=  ' WHERE ' + ' AND '.join(['{} {} {}'.format(self.columns[i], self.cond_ops[o], v) for i, o, v in self.conditions])
        return ' '.join(rep.split())

    @classmethod
    def from_dict(cls, d, t):
        return cls(sel_index=d['sel'], agg_index=d['agg'], columns=t, conditions=d['conds'])


class WikiSQL(CQA, data.Dataset):

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.context), len(ex.answer))

    urls = ['https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2']
    name = 'wikisql'
    dirname = 'data'

    def __init__(self, path, field, query_as_question=False, subsample=None, **kwargs):
        fields = [(x, field) for x in self.fields]
        FIELD = data.Field(batch_first=True, use_vocab=False, sequential=False, 
            lower=False, numerical=True, eos_token=field.eos_token, init_token=field.init_token)
        fields.append(('wikisql_id', FIELD))


        cache_name = os.path.join(os.path.dirname(path), '.cache', 'query_as_question' if query_as_question else 'query_as_context', os.path.basename(path), str(subsample))
        if os.path.exists(cache_name):
            print(f'Loading cached data from {cache_name}')
            examples, all_answers = torch.load(cache_name)
        else:

            expanded_path = os.path.expanduser(path)
            table_path = os.path.splitext(expanded_path)
            table_path = table_path[0] + '.tables' + table_path[1]
           
            with open(table_path) as tables_file:
                tables = [json.loads(line) for line in tables_file]
                id_to_tables = {x['id']: x for x in tables}

            all_answers = []
            examples = []
            with open(expanded_path) as example_file:
                for idx, line in enumerate(example_file):
                    entry = json.loads(line)
                    human_query = entry['question']
                    table = id_to_tables[entry['table_id']]
                    sql = entry['sql']
                    header = table['header']
                    answer = repr(Query.from_dict(sql, header))
                    context = (f'The table has columns {", ".join(table["header"])} ' +
                               f'and key words {", ".join(Query.agg_ops[1:] + Query.cond_ops + Query.syms)}')
                    if query_as_question:
                        question = human_query
                    else:
                        question = 'What is the translation from English to SQL?'
                        context += f'-- {human_query}'  
                    context_question = get_context_question(context, question) 
                    ex = data.Example.fromlist([context, question, answer, CONTEXT_SPECIAL, QUESTION_SPECIAL, context_question, idx], fields)
                    examples.append(ex)
                    all_answers.append({'sql': sql, 'header': header, 'answer': answer, 'table': table})
                    if subsample is not None and len(examples) > subsample:
                        break

            os.makedirs(os.path.dirname(cache_name), exist_ok=True)
            print(f'Caching data to {cache_name}')
            torch.save((examples, all_answers), cache_name)

        super(WikiSQL, self).__init__(examples, fields, **kwargs)
        self.all_answers = all_answers


    @classmethod
    def splits(cls, fields, root='.data',
               train='train.jsonl', validation='dev.jsonl', test='test.jsonl', **kwargs):
        """Create dataset objects for splits of the SQuAD dataset.
        Arguments:
            root: directory containing SQuAD data
            field: field for handling all columns
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        path = cls.download(root)

        train_data = None if train is None else cls(
            os.path.join(path, train), fields, **kwargs)
        validation_data = None if validation is None else cls(
            os.path.join(path, validation), fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), fields, **kwargs)
        return tuple(d for d in (train_data, validation_data, test_data)
                     if d is not None)


class SRL(CQA, data.Dataset):

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.context), len(ex.answer))

    urls = ['https://dada.cs.washington.edu/qasrl/data/wiki1.train.qa',
            'https://dada.cs.washington.edu/qasrl/data/wiki1.dev.qa',
            'https://dada.cs.washington.edu/qasrl/data/wiki1.test.qa']

    name = 'srl'
    dirname = ''

    @classmethod
    def clean(cls, s):
        closing_punctuation = set([ ' .', ' ,', ' ;', ' !', ' ?', ' :', ' )', " 'll", " n't ", " %", " 't", " 's", " 'm", " 'd", " 're"])
        opening_punctuation = set(['( ', '$ '])
        both_sides = set([' - '])
        s = ' '.join(s.split()).strip()
        s = s.replace('-LRB-', '(')
        s = s.replace('-RRB-', ')')
        s = s.replace('-LAB-', '<')
        s = s.replace('-RAB-', '>')
        s = s.replace('-AMP-', '&')
        s = s.replace('%pw', ' ')

        for p in closing_punctuation:
            s = s.replace(p, p.lstrip())
        for p in opening_punctuation:
            s = s.replace(p, p.rstrip())
        for p in both_sides:
            s = s.replace(p, p.strip())
        s = s.replace('``', '')
        s = s.replace('`', '')
        s = s.replace("''", '')
        s = s.replace('“', '')
        s = s.replace('”', '')
        s = s.replace(" '", '')
        return ' '.join(s.split()).strip()

    def __init__(self, path, field, one_answer=True, subsample=None, **kwargs):
        fields = [(x, field) for x in self.fields]
        cache_name = os.path.join(os.path.dirname(path), '.cache', os.path.basename(path), str(subsample))

        examples, all_answers = [], []
        if os.path.exists(cache_name):
            print(f'Loading cached data from {cache_name}')
            examples, all_answers = torch.load(cache_name)
        else:
            with open(os.path.expanduser(path)) as f:
                for line in f:
                    ex = json.loads(line)
                    t = ex['type']
                    aa = ex['all_answers']
                    context, question, answer = ex['context'], ex['question'], ex['answer']
                    context_question = get_context_question(context, question) 
                    ex = data.Example.fromlist([context, question, answer, CONTEXT_SPECIAL, QUESTION_SPECIAL, context_question], fields)
                    examples.append(ex)
                    ex.squad_id = len(all_answers)
                    all_answers.append(aa)
                    if subsample is not None and len(examples) >= subsample: 
                        break
            os.makedirs(os.path.dirname(cache_name), exist_ok=True)
            print(f'Caching data to {cache_name}')
            torch.save((examples, all_answers), cache_name)

        FIELD = data.Field(batch_first=True, use_vocab=False, sequential=False, 
            lower=False, numerical=True, eos_token=field.eos_token, init_token=field.init_token)
        fields.append(('squad_id', FIELD))

        super(SRL, self).__init__(examples, fields, **kwargs)
        self.all_answers = all_answers


    @classmethod
    def cache_splits(cls, path, path_to_files, train='train', validation='dev', test='test'):

        for split in [train, validation, test]:
            split_file_name = os.path.join(path, f'{split}.jsonl')
            if os.path.exists(split_file_name):
                continue
            wiki_file = os.path.join(path, f'wiki1.{split}.qa')

            with open(split_file_name, 'w') as split_file:
                with open(os.path.expanduser(wiki_file)) as f:
                    def is_int(x):
                        try:
                            int(x)
                            return True
                        except:
                            return False

                    lines = []
                    for line in f.readlines():
                        line = ' '.join(line.split()).strip()
                        if len(line) == 0:
                            lines.append(line)
                            continue
                        if not 'WIKI1' in line.split('_')[0]:
                            if not is_int(line.split()[0]) or len(line.split()) > 3:
                                lines.append(line)

                    new_example = True
                    for line in lines:
                        line = line.strip() 
                        if new_example:
                            context = cls.clean(line)
                            new_example = False
                            continue
                        if len(line) == 0:
                            new_example = True
                            continue
                        question, answers = line.split('?')
                        question = cls.clean(line.split('?')[0].replace(' _', '') +'?') 
                        answer = cls.clean(answers.split('###')[0])
                        all_answers = [cls.clean(x) for x in answers.split('###')]
                        if answer not in context:
                            low_answer = answer[0].lower() + answer[1:]
                            up_answer = answer[0].upper() + answer[1:]
                            if low_answer in context or up_answer in context:
                                answer = low_answer if low_answer in context else up_answer
                            else:
                                if 'Darcy Burner' in answer:
                                    answer = 'Darcy Burner and other 2008 Democratic congressional candidates, in cooperation with some retired national security officials'
                                elif 'E Street Band' in answer:
                                    answer = 'plan to work with the E Street Band again in the future'
                                elif 'an electric sender' in answer:
                                    answer = 'an electronic sender'
                                elif 'the US army' in answer:
                                    answer = 'the US Army'
                                elif 'Rather than name the' in answer:
                                    answer = 'rather die than name the cause of his disease to his father'
                                elif answer.lower() in context:
                                    answer = answer.lower()
                                else:
                                    import pdb; pdb.set_trace()
                        assert answer in context
                        modified_all_answers = []
                        for a in all_answers:
                            if a not in context:
                                low_answer = a[0].lower() + a[1:]
                                up_answer = a[0].upper() + a[1:]
                                if low_answer in context or up_answer in context:
                                    a = low_answer if low_answer in context else up_answer
                                else:
                                    if 'Darcy Burner' in a:
                                        a = 'Darcy Burner and other 2008 Democratic congressional candidates, in cooperation with some retired national security officials'
                                    elif 'E Street Band' in a:
                                        a = 'plan to work with the E Street Band again in the future'
                                    elif 'an electric sender' in a:
                                        a = 'an electronic sender'
                                    elif 'the US army' in a:
                                        a = 'the US Army'
                                    elif 'Rather than name the' in a:
                                        a = 'rather die than name the cause of his disease to his father'
                                    elif a.lower() in context:
                                        a = a.lower()
                                    else:
                                        import pdb; pdb.set_trace()
                            assert a in context
                            modified_all_answers.append(a)
                        split_file.write(json.dumps({'context': context, 'question': question, 'answer': answer, 'type': 'wiki', 'all_answers': modified_all_answers})+'\n')
            

            

    @classmethod
    def splits(cls, fields, root='.data',
               train='train', validation='dev', test='test', **kwargs):
        path = cls.download(root)
        cls.cache_splits(path, None)

        train_data = None if train is None else cls(
            os.path.join(path, f'{train}.jsonl'), fields, **kwargs)
        validation_data = None if validation is None else cls(
            os.path.join(path, f'{validation}.jsonl'), fields, one_answer=False, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, f'{test}.jsonl'), fields, one_answer=False, **kwargs)
        return tuple(d for d in (train_data, validation_data, test_data)
                     if d is not None)


class WinogradSchema(CQA, data.Dataset):

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.context), len(ex.answer))

    urls = ['https://s3.amazonaws.com/research.metamind.io/decaNLP/data/schema.txt']

    name = 'schema'
    dirname = ''

    def __init__(self, path, field, subsample=None, **kwargs):
        fields = [(x, field) for x in self.fields]

        cache_name = os.path.join(os.path.dirname(path), '.cache', os.path.basename(path), str(subsample))
        if os.path.exists(cache_name):
            print(f'Loading cached data from {cache_name}')
            examples = torch.load(cache_name)
        else:
            examples = []
            with open(os.path.expanduser(path)) as f:
                for line in f:
                    ex = json.loads(line)
                    context, question, answer = ex['context'], ex['question'], ex['answer']
                    context_question = get_context_question(context, question) 
                    ex = data.Example.fromlist([context, question, answer, CONTEXT_SPECIAL, QUESTION_SPECIAL, context_question], fields)
                    examples.append(ex)
                    if subsample is not None and len(examples) >= subsample: 
                        break
            os.makedirs(os.path.dirname(cache_name), exist_ok=True)
            print(f'Caching data to {cache_name}')
            torch.save(examples, cache_name)

        super(WinogradSchema, self).__init__(examples, fields, **kwargs)

    @classmethod
    def cache_splits(cls, path):
        pattern = '\[.*\]'
        train_jsonl = os.path.expanduser(os.path.join(path, 'train.jsonl'))
        if os.path.exists(train_jsonl):
            return

        def get_both_schema(context):
             variations = [x[1:-1].split('/') for x in re.findall(pattern, context)]
             splits = re.split(pattern, context)
             results = []
             for which_schema in range(2):
                 vs = [v[which_schema] for v in variations] 
                 context = ''
                 for idx in range(len(splits)):
                     context += splits[idx]
                     if idx < len(vs):
                         context += vs[idx]
                 results.append(context) 
             return results


        schemas = []
        with open(os.path.expanduser(os.path.join(path, 'schema.txt'))) as schema_file:
            schema = []
            for line in schema_file:
                if len(line.split()) == 0:
                    schemas.append(schema)
                    schema = []
                    continue 
                else:
                    schema.append(line.strip())

        examples = []
        for schema in schemas:
            context, question, answer = schema
            contexts = get_both_schema(context)
            questions = get_both_schema(question)
            answers = answer.split('/')
            for idx in range(2):
                answer = answers[idx]
                question = questions[idx] + f' {answers[0]} or {answers[1]}?'
                examples.append({'context': contexts[idx], 'question': question, 'answer': answer})

        traindev = examples[:-100]
        test = examples[-100:]
        train = traindev[:80]
        dev = traindev[80:]

        splits = ['train', 'validation', 'test']
        for split, examples in zip(splits, [train, dev, test]):
            with open(os.path.expanduser(os.path.join(path, f'{split}.jsonl')), 'a') as split_file:
                for ex in examples:
                    split_file.write(json.dumps(ex)+'\n')


    @classmethod
    def splits(cls, fields, root='.data',
               train='train', validation='validation', test='test', **kwargs):
        path = cls.download(root)
        cls.cache_splits(path)

        train_data = None if train is None else cls(
            os.path.join(path, f'{train}.jsonl'), fields, **kwargs)
        validation_data = None if validation is None else cls(
            os.path.join(path, f'{validation}.jsonl'), fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, f'{test}.jsonl'), fields, **kwargs)
        return tuple(d for d in (train_data, validation_data, test_data)
                     if d is not None)


class WOZ(CQA, data.Dataset):

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.context), len(ex.answer))

    urls = ['https://raw.githubusercontent.com/nmrksic/neural-belief-tracker/master/data/woz/woz_train_en.json',
            'https://raw.githubusercontent.com/nmrksic/neural-belief-tracker/master/data/woz/woz_test_de.json',
            'https://raw.githubusercontent.com/nmrksic/neural-belief-tracker/master/data/woz/woz_test_en.json',
            'https://raw.githubusercontent.com/nmrksic/neural-belief-tracker/master/data/woz/woz_train_de.json',
            'https://raw.githubusercontent.com/nmrksic/neural-belief-tracker/master/data/woz/woz_train_en.json',
            'https://raw.githubusercontent.com/nmrksic/neural-belief-tracker/master/data/woz/woz_validate_de.json',
            'https://raw.githubusercontent.com/nmrksic/neural-belief-tracker/master/data/woz/woz_validate_en.json']

    name = 'woz'
    dirname = ''

    def __init__(self, path, field, subsample=None, description='woz.en', **kwargs):
        fields = [(x, field) for x in self.fields]
        FIELD = data.Field(batch_first=True, use_vocab=False, sequential=False, 
            lower=False, numerical=True, eos_token=field.eos_token, init_token=field.init_token)
        fields.append(('woz_id', FIELD))

        examples, all_answers = [], []
        cache_name = os.path.join(os.path.dirname(path), '.cache', os.path.basename(path), str(subsample), description)
        if os.path.exists(cache_name):
            print(f'Loading cached data from {cache_name}')
            examples, all_answers = torch.load(cache_name)
        else:
            with open(os.path.expanduser(path)) as f:
                for woz_id, line in enumerate(f):
                    ex = example_dict = json.loads(line)
                    if example_dict['lang'] in description:
                        context, question, answer = ex['context'], ex['question'], ex['answer']
                        context_question = get_context_question(context, question) 
                        all_answers.append((ex['lang_dialogue_turn'], answer))
                        ex = data.Example.fromlist([context, question, answer, CONTEXT_SPECIAL, QUESTION_SPECIAL, context_question, woz_id], fields)
                        examples.append(ex)

                    if subsample is not None and len(examples) >= subsample: 
                        break
            os.makedirs(os.path.dirname(cache_name), exist_ok=True)
            print(f'Caching data to {cache_name}')
            torch.save((examples, all_answers), cache_name)

        super(WOZ, self).__init__(examples, fields, **kwargs)
        self.all_answers = all_answers

    @classmethod
    def cache_splits(cls, path, train='train', validation='validate', test='test'):
        train_jsonl = os.path.expanduser(os.path.join(path, 'train.jsonl'))
        if os.path.exists(train_jsonl):
            return

        file_name_base = 'woz_{}_{}.json'
        question_base = "What is the change in state"
        for split in [train, validation, test]:
            with open (os.path.expanduser(os.path.join(path, f'{split}.jsonl')), 'a') as split_file:
                for lang in ['en', 'de']:
                    file_path = file_name_base.format(split, lang)
                    with open(os.path.expanduser(os.path.join(path, file_path))) as src_file:
                        dialogues = json.loads(src_file.read())
                        for di, d in enumerate(dialogues):
                            previous_state = {'inform': [], 'request': []}
                            turns = d['dialogue']
                            for ti, t in enumerate(turns):
                                question = 'What is the change in state?'
                                actions = []
                                for act in t['system_acts']:
                                    if isinstance(act, list):
                                        act = ': '.join(act)
                                    actions.append(act)
                                actions = ', '.join(actions)
                                if len(actions) > 0:
                                    actions += ' -- '
                                context = actions + t['transcript']
                                belief_state = t['belief_state']
                                delta_state = {'inform': [], 'request': []}
                                current_state = {'inform': [], 'request': []}
                                for item in belief_state:
                                    if 'slots' in item:
                                        slots = item['slots']
                                        for slot in slots:
                                            act = item['act']
                                            if act == 'inform':
                                                current_state['inform'].append(slot)
                                                if not slot in previous_state['inform']:
                                                    delta_state['inform'].append(slot)
                                                else:
                                                    prev_slot = previous_state['inform'][previous_state['inform'].index(slot)]
                                                    if prev_slot[1] != slot[1]:
                                                        delta_state['inform'].append(slot)
                                            else: 
                                                delta_state['request'].append(slot[1])
                                                current_state['request'].append(slot[1])
                                previous_state = current_state
                                answer = ''
                                if len(delta_state['inform']) > 0:
                                    answer = ', '.join([f'{x[0]}: {x[1]}' for x in delta_state['inform']])
                                answer += ';'
                                if len(delta_state['request']) > 0:
                                    answer += ' '
                                    answer += ', '.join(delta_state['request'])
                                ex = {'context': ' '.join(context.split()), 
                                     'question': ' '.join(question.split()), 'lang': lang,
                                     'answer': answer if len(answer) > 1 else 'None',
                                     'lang_dialogue_turn': f'{lang}_{di}_{ti}'}
                                split_file.write(json.dumps(ex)+'\n')


    @classmethod
    def splits(cls, fields, root='.data', train='train', validation='validate', test='test', **kwargs):
        path = cls.download(root)
        cls.cache_splits(path)

        train_data = None if train is None else cls(
            os.path.join(path, f'{train}.jsonl'), fields, **kwargs)
        validation_data = None if validation is None else cls(
            os.path.join(path, f'{validation}.jsonl'), fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, f'{test}.jsonl'), fields, **kwargs)
        return tuple(d for d in (train_data, validation_data, test_data)
                     if d is not None)


class MultiNLI(CQA, data.Dataset):

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.context), len(ex.answer))

    urls = ['http://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip']

    name = 'multinli'
    dirname = 'multinli_1.0'

    def __init__(self, path, field, subsample=None, description='multinli.in.out', **kwargs):
        fields = [(x, field) for x in self.fields]

        cache_name = os.path.join(os.path.dirname(path), '.cache', os.path.basename(path), str(subsample), description)
        if os.path.exists(cache_name):
            print(f'Loading cached data from {cache_name}')
            examples = torch.load(cache_name)
        else:
            examples = []
            with open(os.path.expanduser(path)) as f:
                for line in f:
                    ex = example_dict = json.loads(line)
                    if example_dict['subtask'] in description:
                        context, question, answer = ex['context'], ex['question'], ex['answer']
                        context_question = get_context_question(context, question) 
                        ex = data.Example.fromlist([context, question, answer, CONTEXT_SPECIAL, QUESTION_SPECIAL, context_question], fields)
                        examples.append(ex)
                    if subsample is not None and len(examples) >= subsample: 
                        break
            os.makedirs(os.path.dirname(cache_name), exist_ok=True)
            print(f'Caching data to {cache_name}')
            torch.save(examples, cache_name)

        super(MultiNLI, self).__init__(examples, fields, **kwargs)

    @classmethod
    def cache_splits(cls, path, train='multinli_1.0_train', validation='mulinli_1.0_dev_{}', test='test'):
        train_jsonl = os.path.expanduser(os.path.join(path, 'train.jsonl'))
        if os.path.exists(train_jsonl):
            return

        with open(os.path.expanduser(os.path.join(path, f'train.jsonl')), 'a') as split_file:
            with open(os.path.expanduser(os.path.join(path, f'multinli_1.0_train.jsonl'))) as src_file:
                for line in src_file:
                   ex = json.loads(line)
                   ex = {'context': f'Premise: "{ex["sentence1"]}"', 
                         'question': f'Hypothesis: "{ex["sentence2"]}" -- entailment, neutral, or contradiction?', 
                         'answer': ex['gold_label'], 
                         'subtask': 'multinli'}
                   split_file.write(json.dumps(ex)+'\n')

        with open(os.path.expanduser(os.path.join(path, f'validation.jsonl')), 'a') as split_file:
            for subtask in ['matched', 'mismatched']:
                with open(os.path.expanduser(os.path.join(path, 'multinli_1.0_dev_{}.jsonl'.format(subtask)))) as src_file:
                    for line in src_file:
                       ex = json.loads(line)
                       ex = {'context': f'Premise: "{ex["sentence1"]}"', 
                             'question': f'Hypothesis: "{ex["sentence2"]}" -- entailment, neutral, or contradiction?', 
                             'answer': ex['gold_label'], 
                             'subtask': 'in' if subtask == 'matched' else 'out'}
                       split_file.write(json.dumps(ex)+'\n')


    @classmethod
    def splits(cls, fields, root='.data', train='train', validation='validation', test='test', **kwargs):
        path = cls.download(root)
        cls.cache_splits(path)

        train_data = None if train is None else cls(
            os.path.join(path, f'{train}.jsonl'), fields, **kwargs)
        validation_data = None if validation is None else cls(
            os.path.join(path, f'{validation}.jsonl'), fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, f'{test}.jsonl'), fields, **kwargs)
        return tuple(d for d in (train_data, validation_data, test_data)
                     if d is not None)


class ZeroShotRE(CQA, data.Dataset):

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.context), len(ex.answer))

    urls = ['http://nlp.cs.washington.edu/zeroshot/relation_splits.tar.bz2']
    dirname = 'relation_splits'
    name = 'zre'


    def __init__(self, path, field, subsample=None, **kwargs):
        fields = [(x, field) for x in self.fields]

        cache_name = os.path.join(os.path.dirname(path), '.cache', os.path.basename(path), str(subsample))
        if os.path.exists(cache_name):
            print(f'Loading cached data from {cache_name}')
            examples = torch.load(cache_name)
        else:
            examples = []
            with open(os.path.expanduser(path)) as f:
                for line in f:
                    ex = example_dict = json.loads(line)
                    context, question, answer = ex['context'], ex['question'], ex['answer']
                    context_question = get_context_question(context, question) 
                    ex = data.Example.fromlist([context, question, answer, CONTEXT_SPECIAL, QUESTION_SPECIAL, context_question], fields)
                    examples.append(ex)

                    if subsample is not None and len(examples) >= subsample: 
                        break
            os.makedirs(os.path.dirname(cache_name), exist_ok=True)
            print(f'Caching data to {cache_name}')
            torch.save(examples, cache_name)

        super().__init__(examples, fields, **kwargs)

    @classmethod
    def cache_splits(cls, path, train='train', validation='dev', test='test'):
        train_jsonl = os.path.expanduser(os.path.join(path, f'{train}.jsonl'))
        if os.path.exists(train_jsonl):
            return

        base_file_name = '{}.0'
        for split in [train, validation, test]:
            src_file_name = base_file_name.format(split)
            with open(os.path.expanduser(os.path.join(path, f'{split}.jsonl')), 'a') as split_file:
                with open(os.path.expanduser(os.path.join(path, src_file_name))) as src_file:
                    for line in src_file:
                       split_line = line.split('\t')
                       if len(split_line) == 4:
                           answer = ''
                           relation, question, subject, context = split_line
                       else:
                           relation, question, subject, context = split_line[:4]
                           answer = ', '.join(split_line[4:])
                       question = question.replace('XXX', subject)
                       ex = {'context': context, 
                             'question': question, 
                             'answer': answer if len(answer) > 0 else 'unanswerable'}
                       split_file.write(json.dumps(ex)+'\n')


    @classmethod
    def splits(cls, fields, root='.data', train='train', validation='dev', test='test', **kwargs):
        path = cls.download(root)
        cls.cache_splits(path)

        train_data = None if train is None else cls(
            os.path.join(path, f'{train}.jsonl'), fields, **kwargs)
        validation_data = None if validation is None else cls(
            os.path.join(path, f'{validation}.jsonl'), fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, f'{test}.jsonl'), fields, **kwargs)
        return tuple(d for d in (train_data, validation_data, test_data)
                     if d is not None)


class OntoNotesNER(CQA, data.Dataset):

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.context), len(ex.answer))

    urls = ['http://conll.cemantix.org/2012/download/ids/english/all/train.id',
            'http://conll.cemantix.org/2012/download/ids/english/all/development.id',
            'http://conll.cemantix.org/2012/download/ids/english/all/test.id']

    name = 'ontonotes.ner'
    dirname = ''

    @classmethod
    def clean(cls, s):
        closing_punctuation = set([ ' .', ' ,', ' ;', ' !', ' ?', ' :', ' )', " '", " n't ", " %"])
        opening_punctuation = set(['( ', '$ '])
        both_sides = set([' - '])
        s = ' '.join(s.split()).strip()
        s = s.replace(' /.', ' .')
        s = s.replace(' /?', ' ?')
        s = s.replace('-LRB-', '(')
        s = s.replace('-RRB-', ')')
        s = s.replace('-LAB-', '<')
        s = s.replace('-RAB-', '>')
        s = s.replace('-AMP-', '&')
        s = s.replace('%pw', ' ')

        for p in closing_punctuation:
            s = s.replace(p, p.lstrip())
        for p in opening_punctuation:
            s = s.replace(p, p.rstrip())
        for p in both_sides:
            s = s.replace(p, p.strip())
        s = s.replace('``', '"')
        s = s.replace("''", '"')
        quote_is_open = True
        quote_idx = s.find('"')
        raw = ''
        while quote_idx >= 0:
            start_enamex_open_idx = s.find('<ENAMEX')
            if start_enamex_open_idx > -1:
                end_enamex_open_idx = s.find('">') + 2
                if start_enamex_open_idx <= quote_idx  <= end_enamex_open_idx:
                    raw += s[:end_enamex_open_idx]
                    s = s[end_enamex_open_idx:]
                    quote_idx = s.find('"')
                    continue
            if quote_is_open:
                raw += s[:quote_idx+1]
                s = s[quote_idx+1:].strip()
                quote_is_open = False
            else:
                raw += s[:quote_idx].strip() + '"'
                s =  s[quote_idx+1:]
                quote_is_open = True
            quote_idx = s.find('"')
        raw += s

        return ' '.join(raw.split()).strip()

    def __init__(self, path, field, one_answer=True, subsample=None, path_to_files='.data/ontonotes-release-5.0/data/files', subtask='all', nones=True, **kwargs):
        fields = [(x, field) for x in self.fields]

        cache_name = os.path.join(os.path.dirname(path), '.cache', os.path.basename(path), str(subsample), subtask, str(nones))
        if os.path.exists(cache_name):
            print(f'Loading cached data from {cache_name}')
            examples = torch.load(cache_name)
        else:
            examples = []
            with open(os.path.expanduser(path)) as f:
                for line in f:
                    example_dict = json.loads(line)  
                    t = example_dict['type']
                    a = example_dict['answer']
                    if (subtask == 'both' or t == subtask):
                        if a != 'None' or nones:
                            ex = example_dict
                            context, question, answer = ex['context'], ex['question'], ex['answer']
                            context_question = get_context_question(context, question) 
                            ex = data.Example.fromlist([context, question, answer, CONTEXT_SPECIAL, QUESTION_SPECIAL, context_question], fields)
                            examples.append(ex)

                    if subsample is not None and len(examples) >= subsample: 
                        break
            os.makedirs(os.path.dirname(cache_name), exist_ok=True)
            print(f'Caching data to {cache_name}')
            torch.save(examples, cache_name)

        super(OntoNotesNER, self).__init__(examples, fields, **kwargs)


    @classmethod
    def cache_splits(cls, path, path_to_files, train='train', validation='development', test='test'):

        label_to_answer = {'PERSON': 'person',
                           'NORP': 'political',
                           'FAC': 'facility',
                           'ORG': 'organization',
                           'GPE': 'geopolitical',
                           'LOC': 'location',
                           'PRODUCT': 'product',
                           'EVENT': 'event',
                           'WORK_OF_ART': 'artwork',
                           'LAW': 'legal',
                           'LANGUAGE': 'language',
                           'DATE': 'date',
                           'TIME': 'time',
                           'PERCENT': 'percentage',
                           'MONEY': 'monetary',
                           'QUANTITY': 'quantitative',
                           'ORDINAL': 'ordinal',
                           'CARDINAL': 'cardinal'}
        
        pluralize = {'person': 'persons', 'political': 'political', 'facility': 'facilities', 'organization': 'organizations', 
                     'geopolitical': 'geopolitical', 'location': 'locations', 'product': 'products', 'event': 'events',
                     'artwork': 'artworks', 'legal': 'legal', 'language': 'languages', 'date': 'dates', 'time': 'times', 
                     'percentage': 'percentages', 'monetary': 'monetary', 'quantitative': 'quantitative', 'ordinal': 'ordinal',
                     'cardinal': 'cardinal'}

 
        for split in [train, validation, test]:
            split_file_name = os.path.join(path, f'{split}.jsonl')
            if os.path.exists(split_file_name):
                continue
            id_file = os.path.join(path, f'{split}.id')

            num_file_ids = 0
            examples = []
            with open(split_file_name, 'w') as split_file:
                with open(os.path.expanduser(id_file)) as f:
                    for file_id in f:
                        example_file_name = os.path.join(os.path.expanduser(path_to_files), file_id.strip()) + '.name'
                        if not os.path.exists(example_file_name) or 'annotations/tc/ch' in example_file_name:
                            continue
                        num_file_ids += 1
                        with open(example_file_name) as example_file:
                            lines = [x.strip() for x in example_file.readlines() if 'DOC' not in x]
                            for line in lines:
                                original = line
                                line = cls.clean(line)
                                entities = []  
                                while True:
                                    start_enamex_open_idx = line.find('<ENAMEX')
                                    if start_enamex_open_idx == -1:
                                        break
                                    end_enamex_open_idx = line.find('">') + 2
                                    start_enamex_close_idx = line.find('</ENAMEX>')
                                    end_enamex_close_idx = start_enamex_close_idx + len('</ENAMEX>')
    
                                    enamex_open_tag = line[start_enamex_open_idx:end_enamex_open_idx]
                                    enamex_close_tag = line[start_enamex_close_idx:end_enamex_close_idx]
                                    before_entity = line[:start_enamex_open_idx]
                                    entity = line[end_enamex_open_idx:start_enamex_close_idx]
                                    after_entity = line[end_enamex_close_idx:]
    
                                    if 'S_OFF' in enamex_open_tag:
                                        s_off_start = enamex_open_tag.find('S_OFF="')
                                        s_off_end = enamex_open_tag.find('">') if 'E_OFF' not in enamex_open_tag else enamex_open_tag.find('" E_OFF')
                                        s_off = int(enamex_open_tag[s_off_start+len('S_OFF="'):s_off_end])
                                        enamex_open_tag = enamex_open_tag[:s_off_start-2] + '">'
                                        before_entity += entity[:s_off]
                                        entity = entity[s_off:]
    
                                    if 'E_OFF' in enamex_open_tag:
                                        s_off_start = enamex_open_tag.find('E_OFF="')
                                        s_off_end = enamex_open_tag.find('">')
                                        s_off = int(enamex_open_tag[s_off_start+len('E_OFF="'):s_off_end])
                                        enamex_open_tag = enamex_open_tag[:s_off_start-2] + '">'
                                        after_entity = entity[-s_off:] + after_entity
                                        entity = entity[:-s_off]
    
    
                                    label_start = enamex_open_tag.find('TYPE="') + len('TYPE="')
                                    label_end = enamex_open_tag.find('">')
                                    label = enamex_open_tag[label_start:label_end]
                                    assert label in label_to_answer

                                    offsets = (len(before_entity), len(before_entity) + len(entity))
                                    entities.append({'entity': entity, 'char_offsets': offsets, 'label': label})
                                    line = before_entity + entity + after_entity
                                
                                context = line.strip()
                                is_no_good = False
                                for entity_tuple in entities:
                                    entity = entity_tuple['entity']
                                    start, end = entity_tuple['char_offsets']
                                    if not context[start:end] == entity:
                                        is_no_good = True
                                        break
                                if is_no_good:
                                    print('Throwing out example that looks poorly labeled: ', original.strip(), ' (', file_id.strip(), ')')
                                    continue
                                question = 'What are the tags for all entities?'
                                answer = '; '.join([f'{x["entity"]} -- {label_to_answer[x["label"]]}' for x in entities]) 
                                if len(answer) == 0:
                                    answer = 'None'
                                split_file.write(json.dumps({'context': context, 'question': question, 'answer': answer, 'file_id': file_id.strip(), 
                                                             'original': original.strip(), 'entity_list': entities, 'type': 'all'})+'\n')
                                partial_question = 'Which entities are {}?'
 
                                for lab, ans in label_to_answer.items():
                                    question = partial_question.format(pluralize[ans])
                                    entity_of_type_lab = [x['entity'] for x in entities if x['label'] == lab]
                                    answer = ', '.join(entity_of_type_lab)
                                    if len(answer) == 0:
                                        answer = 'None'
                                    split_file.write(json.dumps({'context': context, 
                                                                 'question': question, 
                                                                 'answer': answer, 
                                                                 'file_id': file_id.strip(), 
                                                                 'original': original.strip(), 
                                                                 'entity_list': entities, 
                                                                 'type': 'one', 
                                                                 })+'\n')



    @classmethod
    def splits(cls, fields, root='.data',
               train='train', validation='development', test='test', **kwargs):
        path_to_files = os.path.join(root, 'ontonotes-release-5.0', 'data', 'files')
        assert os.path.exists(path_to_files)
        path = cls.download(root)
        cls.cache_splits(path, path_to_files)

        train_data = None if train is None else cls(
            os.path.join(path, f'{train}.jsonl'), fields, **kwargs)
        validation_data = None if validation is None else cls(
            os.path.join(path, f'{validation}.jsonl'), fields, one_answer=False, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, f'{test}.jsonl'), fields, one_answer=False, **kwargs)
        return tuple(d for d in (train_data, validation_data, test_data)
                     if d is not None)

class SNLI(CQA, data.Dataset):

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.context), len(ex.answer))

    urls = ['http://nlp.stanford.edu/projects/snli/snli_1.0.zip']
    dirname = 'snli_1.0'
    name = 'snli'


    def __init__(self, path, field, subsample=None, **kwargs):
        fields = [(x, field) for x in self.fields]

        cache_name = os.path.join(os.path.dirname(path), '.cache', os.path.basename(path), str(subsample))
        if os.path.exists(cache_name):
            print(f'Loading cached data from {cache_name}')
            examples = torch.load(cache_name)
        else:
            examples = []
            with open(os.path.expanduser(path)) as f:
                for line in f:
                    example_dict = json.loads(line)
                    ex = example_dict
                    context, question, answer = ex['context'], ex['question'], ex['answer']
                    context_question = get_context_question(context, question) 
                    ex = data.Example.fromlist([context, question, answer, CONTEXT_SPECIAL, QUESTION_SPECIAL, context_question], fields)
                    examples.append(ex)

                    if subsample is not None and len(examples) >= subsample: 
                        break
            os.makedirs(os.path.dirname(cache_name), exist_ok=True)
            print(f'Caching data to {cache_name}')
            torch.save(examples, cache_name)

        super().__init__(examples, fields, **kwargs)

    @classmethod
    def cache_splits(cls, path, train='train', validation='dev', test='test'):
        train_jsonl = os.path.expanduser(os.path.join(path, f'{train}.jsonl'))
        if os.path.exists(train_jsonl):
            return

        base_file_name = 'snli_1.0_{}.jsonl'
        for split in [train, validation, test]:
            src_file_name = base_file_name.format(split)
            with open(os.path.expanduser(os.path.join(path, f'{split}.jsonl')), 'a') as split_file:
                with open(os.path.expanduser(os.path.join(path, src_file_name))) as src_file:
                    for line in src_file:
                       ex = json.loads(line)
                       ex = {'context': f'Premise: "{ex["sentence1"]}"', 
                             'question': f'Hypothesis: "{ex["sentence2"]}" -- entailment, neutral, or contradiction?', 
                             'answer': ex['gold_label']}
                       split_file.write(json.dumps(ex)+'\n')


    @classmethod
    def splits(cls, fields, root='.data', train='train', validation='dev', test='test', **kwargs):
        path = cls.download(root)
        cls.cache_splits(path)

        train_data = None if train is None else cls(
            os.path.join(path, f'{train}.jsonl'), fields, **kwargs)
        validation_data = None if validation is None else cls(
            os.path.join(path, f'{validation}.jsonl'), fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, f'{test}.jsonl'), fields, **kwargs)
        return tuple(d for d in (train_data, validation_data, test_data)
                     if d is not None)


class JSON(CQA, data.Dataset):

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.context), len(ex.answer))

    def __init__(self, path, field, subsample=None, **kwargs):
        fields = [(x, field) for x in self.fields]
        cache_name = os.path.join(os.path.dirname(path), '.cache', os.path.basename(path), str(subsample))

        examples = []
        if os.path.exists(cache_name):
            print(f'Loading cached data from {cache_name}')
            examples = torch.load(cache_name)
        else:
            with open(os.path.expanduser(path)) as f:
                lines = f.readlines()
                for line in lines:
                    ex = json.loads(line)
                    context, question, answer = ex['context'], ex['question'], ex['answer']
                    context_question = get_context_question(context, question) 
                    ex = data.Example.fromlist([context, question, answer, CONTEXT_SPECIAL, QUESTION_SPECIAL, context_question], fields)
                    examples.append(ex)
                    if subsample is not None and len(examples) >= subsample: 
                        break
            os.makedirs(os.path.dirname(cache_name), exist_ok=True)
            print(f'Caching data to {cache_name}')
            torch.save(examples, cache_name)

        super(JSON, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, fields, name, root='.data',
               train='train', validation='val', test='test', **kwargs):
        path = os.path.join(root, name) 

        train_data = None if train is None else cls(
            os.path.join(path, 'train.jsonl'), fields, **kwargs)
        validation_data = None if validation is None else cls(
            os.path.join(path, 'val.jsonl'), fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, 'test.jsonl'), fields, **kwargs)
        return tuple(d for d in (train_data, validation_data, test_data)
                     if d is not None)
