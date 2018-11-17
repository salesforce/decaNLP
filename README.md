![decaNLP Logo](decaNLP_logo.png)
--------------------------------------------------------------------------------
[![Build Status](https://travis-ci.org/salesforce/decaNLP.svg?branch=master)](https://travis-ci.org/salesforce/decaNLP)

The Natural Language Decathlon is a multitask challenge that spans ten tasks:
question answering ([SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)), machine translation ([IWSLT](https://wit3.fbk.eu/mt.php?release=2016-01)), summarization ([CNN/DM](https://cs.nyu.edu/~kcho/DMQA/)), natural language inference ([MNLI](https://www.nyu.edu/projects/bowman/multinli/)), sentiment analysis ([SST](https://nlp.stanford.edu/sentiment/treebank.html)), semantic role labeling([QA&#8209;SRL](https://dada.cs.washington.edu/qasrl/)), zero-shot relation extraction ([QA&#8209;ZRE](http://nlp.cs.washington.edu/zeroshot/)), goal-oriented dialogue ([WOZ](https://github.com/nmrksic/neural-belief-tracker/tree/master/data/woz), semantic parsing ([WikiSQL](https://github.com/salesforce/WikiSQL)), and commonsense reasoning ([MWSC](https://github.com/salesforce/decaNLP/blob/d594b2bf127e13d0e61151b6a2af3bf63612f380/local_data/schema.txt)).
Each task is cast as question answering, which makes it possible to use our new Multitask Question Answering Network ([MQAN](https://github.com/salesforce/decaNLP/blob/d594b2bf127e13d0e61151b6a2af3bf63612f380/models/multitask_question_answering_network.py)).
This model jointly learns all tasks in decaNLP without any task-specific modules or parameters in the multitask setting. For a more thorough introduction to decaNLP and the tasks, see the main [website](http://decanlp.com/), our [blog post](https://einstein.ai/research/the-natural-language-decathlon), or the [paper](https://arxiv.org/abs/1806.08730).

While the research direction associated with this repository focused on multitask learning, the framework itself is designed in a way that should make single-task training, transfer learning, and zero-shot evaluation simple. Similarly, the [paper](https://arxiv.org/abs/1806.08730) focused on multitask learning as a form of question answering, but this framework can be easily adapted for different approached to single-task or multitask learning.

## Leaderboard

| Model | decaNLP | [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) | [IWSLT](https://wit3.fbk.eu/mt.php?release=2016-01) | [CNN/DM](https://cs.nyu.edu/~kcho/DMQA/) | [MNLI](https://www.nyu.edu/projects/bowman/multinli/) | [SST](https://nlp.stanford.edu/sentiment/treebank.html) | [QA&#8209;SRL](https://dada.cs.washington.edu/qasrl/) | [QA&#8209;ZRE](http://nlp.cs.washington.edu/zeroshot/) | [WOZ](https://github.com/nmrksic/neural-belief-tracker/tree/master/data/woz) | [WikiSQL](https://github.com/salesforce/WikiSQL) | [MWSC](https://github.com/salesforce/decaNLP/blob/d594b2bf127e13d0e61151b6a2af3bf63612f380/local_data/schema.txt) |
| --- | --- | --- | --- | --- | --- | --- | ---- | ---- | --- | --- |--- | 
| [MQAN](https://arxiv.org/abs/1806.08730)(QA&#8209;first+[CoVe](http://papers.nips.cc/paper/7209-learned-in-translation-contextualized-word-vectors)) | 599.9 | 75.5 | 18.9 | 24.4 | 73.6 | 86.4 | 80.8 | 37.4 | 85.8 | 68.5 | 48.8 |
| [MQAN](https://arxiv.org/abs/1806.08730)(QA&#8209;first) | 590.5 | 74.4 | 18.6 | 24.3 | 71.5 | 87.4 | 78.4 | 37.6 | 84.8 | 64.8 | 48.7 |
| [S2S](https://arxiv.org/abs/1806.08730) | 513.6 | 47.5 | 14.2 | 25.7 | 60.9 | 85.9 | 68.7 | 28.5 | 84.0 | 45.8 | 52.4 |

## Getting Started

First, you will need to make a `.data` directory and move the examples for the Winograd Schemas into it:
```bash
mkdir -p .data/schema
cp local_data/schema.txt .data/schema/
```

### GPU vs. CPU

The `devices` argument can be used to specify the devices for training. For CPU training, specify `--devices -1`; for GPU training, specify `--devices DEVICEID`. Note that Multi-GPU training is currently a WIP, so `--device` is sufficient for commands below. The default will be to train on GPU 0 as training on CPU will be quite time-consuming to train on all ten tasks in decaNLP.

If you want to use CPU, then remove the `nvidia-` and the `cuda9_` prefixes from the default commands listed in sections below. This will allow you to use Docker without CUDA.

For example, if you have CUDA and all the necessary drivers and GPUs, you you can run a command inside the CUDA Docker image using:
```bash
nvidia-docker run -it --rm -v `pwd`:/decaNLP/ -u $(id -u):$(id -g) bmccann/decanlp:cuda9_torch041 bash -c "COMMAND --device 0"
```

If you want to run the same command without CUDA:
```bash
docker run -it --rm -v `pwd`:/decaNLP/ -u $(id -u):$(id -g) bmccann/decanlp:cuda9_torch041 bash -c "COMMAND --device -1"
```

For those in the Docker know, you can look at the Dockerfiles used to build these two images in `dockerfiles/`.


### PyTorch Version
The research associated with the original paper was done using Pytorch 0.3, but we have since migrated to 0.4. If you want to replicate results from the paper, then to be safe, you should use the code at a commit on or before 3c4f94b88768f4c3efc2fd4f015fed2f5453ebce. You should also replace `toch041` with `torch03` in the commands below to access a Docker image with the older version of PyTorch.

## Training

For example, to train a Multitask Question Answering Network (MQAN) on the Stanford Question Answering Dataset (SQuAD) on GPU 0:
```bash
nvidia-docker run -it --rm -v `pwd`:/decaNLP/ -u $(id -u):$(id -g) bmccann/decanlp:cuda9_torch041 bash -c "python /decaNLP/train.py --train_tasks squad --device 0"
```

To multitask with the fully joint, round-robin training described in the paper, you can add multiple tasks:
```bash
nvidia-docker run -it --rm -v `pwd`:/decaNLP/ -u $(id -u):$(id -g) bmccann/decanlp:cuda9_torch041 bash -c "python /decaNLP/train.py --train_tasks squad iwslt.en.de --train_iterations 1 --device 0"
```

To train on the entire Natural Language Decathlon:
```bash
nvidia-docker run -it --rm -v `pwd`:/decaNLP/ -u $(id -u):$(id -g) bmccann/decanlp:cuda9_torch041 bash -c "python /decaNLP/train.py --train_tasks squad iwslt.en.de cnn_dailymail multinli.in.out sst srl zre woz.en wikisql schema --train_iterations 1 --device 0"
```

To pretrain on `n_jump_start=1` tasks for `jump_start=75000` iterations before switching to round-robin sampling of all tasks in the Natural Language Decathlon:
```bash
nvidia-docker run -it --rm -v `pwd`:/decaNLP/ -u $(id -u):$(id -g) bmccann/decanlp:cuda9_torch041 bash -c "python /decaNLP/train.py --n_jump_start 1 --jump_start 75000 --train_tasks squad iwslt.en.de cnn_dailymail multinli.in.out sst srl zre woz.en wikisql schema --train_iterations 1 --device 0"
```
This jump starting (or pretraining) on a subset of tasks can be done for any set of tasks, not only the entirety of decaNLP.

### Tensorboard

If you would like to make use of tensorboard, you can add the `--tensorboard` flag to your training runs. This will log things in the format that Tensorboard expects.

To read those files and run the Tensorboard server, run (typically in a `tmux` pane or equivalent so that the process is not killed when you shut your laptop) the following command:

```bash
docker run -it --rm -p 0.0.0.0:6006:6006 -v `pwd`:/decaNLP/ -u $(id -u):$(id -g) bmccann/decanlp:cuda9_torch041 bash -c "tensorboard --logdir /decaNLP/results"
```

If you are running the server on a remote machine, you can run the following on your local machine to forward to http://localhost:6006/:

```bash
ssh -4 -N -f -L 6006:127.0.0.1:6006 YOUR_REMOTE_IP
```

If you are having trouble with the specified port on either machine, run `lsof -if:6006` and kill the process if it is unnecessary. Otherwise, try changing the port numbers in the commands above. The first port number is the port the local machine tries to bind to, and and the second port is the one exposed by the remote machine (or docker container).

### Notes on Training

- On a single NVIDIA Volta GPU, the code should take about 3 days to complete 500k iterations. These should be sufficient to approximately reproduce the experiments in the paper. Training for about 7 days should be enough to fully replicate those scores, which should be only a few points higher than what is achieved by 500k iterations.
- The model can be resumed using stored checkpoints using `--load <PATH_TO_CHECKPOINT>` and `--resume`. By default, models are stored every `--save_every` iterations in the `results/` folder tree. 
- During training, validation can be slow! Especially when computing ROUGE scores. Use the `--val_every` flag to change the frequency of validation. 
- If you run out of GPU memory, reduce `--train_batch_tokens` and `--val_batch_size`.
- If you run out of CPU memory, make sure that you are running the most recent version of the code that interns strings; if you are still running out of CPU memory, post an issue with the command you ran and your peak memory usage. 
- The first time you run, the code will download and cache all considered datasets. Please be advised that this might take a while, especially for some of the larger datasets. 

### Notes on Cached Data
- In order to make data loading much quicker for repeated experiments, datasets are cached using code in `text/torchtext/datasets/generic.py`. 
- If there is an update to this repository that touches any files in `text/`, then it might have changed the way a dataset is cached. If this is the case, then you'll need to delete all relevant cached files or you will not see the changes.
- Paths to cached files should be printed out when a dataset is loaded, either in training or in prediction. Search the text logged to stdout for `Loading cached data from` or `Caching data to` in order to locate the relevant path names for data caches.

## Evaluation

You can evaluate a model for a specific task with `EVALUATION_TYPE` as `validation` or `test`:

```bash
nvidia-docker run -it --rm -v `pwd`:/decaNLP/ -u $(id -u):$(id -g) bmccann/decanlp:cuda9_torch041 bash -c "python /decaNLP/predict.py --evaluate EVALUATION_TYPE --path PATH_TO_CHECKPOINT_DIRECTORY --device 0 --tasks squad"
```

or evaluate on the entire decathlon by removing any task specification:
```bash
nvidia-docker run -it --rm -v `pwd`:/decaNLP/ -u $(id -u):$(id -g) bmccann/decanlp:cuda9_torch041 bash -c "python /decaNLP/predict.py --evaluate EVALUATION_TYPE --path PATH_TO_CHECKPOINT_DIRECTORY --device 0"
```

For test performance, please use the original [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/), [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/), and [WikiSQL](https://github.com/salesforce/WikiSQL) evaluation systems. For WikiSQL, there is a detailed walk-through of how to get test numbers in the section of this document concerning [pretrained models](https://github.com/salesforce/decaNLP#pretrained-models).

## Pretrained Models

This model is the best MQAN trained on decaNLP so far. It was trained first on SQuAD and then on all of decaNLP. You can obtain this model and run it on the validation sets with the following.

```bash
wget https://s3.amazonaws.com/research.metamind.io/decaNLP/pretrained/mqan_decanlp_qa_first_cpu.tar.gz
tar -xvzf mqan_decanlp_qa_first_cpu.tar.gz
nvidia-docker run -it --rm -v `pwd`:/decaNLP/ -u $(id -u):$(id -g) bmccann/decanlp:cuda9_torch041 bash -c "python /decaNLP/predict.py --evaluate validation --path /decaNLP/mqan_decanlp_qa_first_cpu --checkpoint_name iteration_1140000.pth --device 0"
```

This model is the best MQAN trained on WikiSQL alone, which established [a new state-of-the-art performance by several points on that task](https://github.com/salesforce/WikiSQL): 73.2 / 75.4 / 81.4 (ordered test logical form accuracy, unordered test logical form accuracy, test execution accuracy).

```bash
wget https://s3.amazonaws.com/research.metamind.io/decaNLP/pretrained/mqan_wikisql_cpu.tar.gz
tar -xvzf mqan_wikisql_cpu.tar.gz
nvidia-docker run -it --rm -v `pwd`:/decaNLP/  bmccann/decanlp:cuda9_torch041 -c "python /decaNLP/predict.py --evaluate validation --path /decaNLP/mqan_wikisql_cpu --checkpoint_name iteration_57000.pth --device 0 --tasks wikisql"
nvidia-docker run -it --rm -v `pwd`:/decaNLP/ -u $(id -u):$(id -g) bmccann/decanlp:cuda9_torch041 bash -c "python /decaNLP/predict.py --evaluate test --path /decaNLP/mqan_wikisql_cpu --checkpoint_name iteration_57000.pth --device 0 --tasks wikisql"
docker run -it --rm -v `pwd`:/decaNLP/ -u $(id -u):$(id -g) bmccann/decanlp:cuda9_torch041 bash -c "python /decaNLP/convert_to_logical_forms.py /decaNLP/.data/ /decaNLP/mqan_wikisql_cpu/iteration_57000/validation/wikisql.txt /decaNLP/mqan_wikisql_cpu/iteration_57000/validation/wikisql.ids.txt /decaNLP/mqan_wikisql_cpu/iteration_57000/validation/wikisql_logical_forms.jsonl valid"
docker run -it --rm -v `pwd`:/decaNLP/ -u $(id -u):$(id -g) bmccann/decanlp:cuda9_torch041 bash -c "python /decaNLP/convert_to_logical_forms.py /decaNLP/.data/ /decaNLP/mqan_wikisql_cpu/iteration_57000/test/wikisql.txt /decaNLP/mqan_wikisql_cpu/iteration_57000/test/wikisql.ids.txt /decaNLP/mqan_wikisql_cpu/iteration_57000/test/wikisql_logical_forms.jsonl test"
git clone https://github.com/salesforce/WikiSQL.git #git@github.com:salesforce/WikiSQL.git for ssh
docker run -it --rm -v `pwd`:/decaNLP/ -u $(id -u):$(id -g) bmccann/decanlp:cuda9_torch041 bash -c "python /decaNLP/WikiSQL/evaluate.py /decaNLP/.data/wikisql/data/dev.jsonl /decaNLP/.data/wikisql/data/dev.db /decaNLP/mqan_wikisql_cpu/iteration_57000/validation/wikisql_logical_forms.jsonl" # assumes that you have data stored in .data
docker run -it --rm -v `pwd`:/decaNLP/ -u $(id -u):$(id -g) bmccann/decanlp:cuda9_torch041 bash -c "python /decaNLP/WikiSQL/evaluate.py /decaNLP/.data/wikisql/data/test.jsonl /decaNLP/.data/wikisql/data/test.db /decaNLP/mqan_wikisql_cpu/iteration_57000/test/wikisql_logical_forms.jsonl" # assumes that you have data stored in .data
```

## Inference on a Custom Dataset

Using a pretrained model or a model you have trained yourself, you can run on new, custom datasets easily by following the instructions below. In this example, we use the checkpoint for the best MQAN trained on the entirety of decaNLP (see the section on Pretrained Models to see how to get this checkpoint) to run on `my_custom_dataset`.

```bash
mkdir -p .data/my_custom_dataset/
touch .data/my_custom_dataset/val.jsonl
echo '{"context": "The answer is answer.", "question": "What is the answer?", "answer": "answer"}' >> .data/my_custom_dataset/val.jsonl 
# TODO add your own examples line by line to val.jsonl in the form of a JSON dictionary, as demonstrated above.
# Make sure to delete the first line if you don't want the demonstrated example.
nvidia-docker run -it --rm -v `pwd`:/decaNLP/ -u $(id -u):$(id -g) bmccann/decanlp:cuda9_torch041 bash -c "python /decaNLP/predict.py --evaluate valid --path /decaNLP/mqan_decanlp_qa_first_cpu --checkpoint_name iteration_1140000.pth --tasks my_custom_dataset"
```
You should get output that ends with something like this:
```
**  /decaNLP/mqan_decanlp_qa_first_cpu/iteration_1140000/valid/my_custom_dataset.txt  already exists -- this is where predictions are stored **
**  /decaNLP/mqan_decanlp_qa_first_cpu/modeltion_1140000/valid/my_custom_dataset.gold.txt  already exists -- this is where ground truth answers are stored **
**  /decaNLP/mqan_decanlp_qa_first_cpu/modeltion_1140000/valid/my_custom_dataset.results.txt  already exists -- this is where metrics are stored **
{"em":0.0,"nf1":100.0,"nem":100.0}

{'em': 0.0, 'nf1': 100.0, 'nem': 100.0}
Prediction: the answer
Answer: answer
```
From this output, you can see where predictions are stored along with ground truth outputs and metrics. If you want to rerun using this model checkpoint on this particular dataset, you'll need to pass the `--overwrite_predictions` argument to `predict.py`. If you do not want predictions and answers printed to stdout, then pass the `--silent` argument to `predict.py`.

The metrics dictionary should have printed something like `{'em': 0.0, 'nf1': 100.0, 'nem': 100.0}`. Here `em` stands for exact match. This is the percentage of predictions that had every token match the ground truth answer exactly. The normalized version, `nem`, lowercases and strips punctuation -- all of our models are trained on lowercased data, so `nem` is a more accurate representation of performance than `em` for our models. For tasks that are typically treated as classification problems, these exact match scores should correspond to accuracy. `nf1` is a normalized (lowercased; punctuation stripped) [F1 score](https://en.wikipedia.org/wiki/F1_score) over the predicted and ground truth sequences. If you would like to add additional metrics that are already implemented you can try adding `--bleu` (the typical metric for machine translation) and `--rouge` (the typical metric for summarization). Other metrics can be implemented following the patterns in `metrics.py`.

## Citation

If you use this in your work, please cite [*The Natural Language Decathlon: Multitask Learning as Question Answering*](https://arxiv.org/abs/1806.08730).

```
@article{McCann2018decaNLP,
  title={The Natural Language Decathlon: Multitask Learning as Question Answering},
  author={Bryan McCann and Nitish Shirish Keskar and Caiming Xiong and Richard Socher},
  journal={arXiv preprint arXiv:1806.08730},
  year={2018}
}
```

## Contact

Contact: [bmccann@salesforce.com](mailto:bmccann@salesforce.com) and [nkeskar@salesforce.com](mailto:nkeskar@salesforce.com)
