# Leaderboard 
<hr>

<table class='performanceTable'>

<tr><th>Rank</th><th>Model</th><th>decaScore</th><th>Breakdown by Task</th></tr>

<tr>
<td><p>1</p><span class="date label label-default">May 18, 2018</span></td>
<td>MQAN <p class="institution">Salesforce Research</p> </td>
<td>571.7</td>
<td>
<table>
<tr><th>SQuAD </th><td>74.3</td><th>QA-SRL </th><td>77.6</td></tr>
<tr><th>IWSLT </th><td>13.7</td><th>QA-ZRE </th><td>34.7</td></tr>
<tr><th>CNN/DM</th><td>24.6</td><th>WOZ    </th><td>84.1</td></tr>
<tr><th>MNLI  </th><td>69.2</td><th>WikiSQL</th><td>58.7</td></tr>
<tr><th>SST   </th><td>86.4</td><th>MWSC   </th><td>48.4</td></tr>
</table>
</td>
</tr>

<tr>
<td><p>2</p><span class="date label label-default">May 18, 2018</span></td>
<td style="word-break:break-word;">Sequence-to-sequence baseline <p class="institution">Salesforce Research</p> </td>
<td>513.6</td>
<td>
<table>
<tr><th>SQuAD </th><td>47.5</td><th>QA-SRL </th><td>68.7</td></tr>
<tr><th>IWSLT </th><td>14.2</td><th>QA-ZRE </th><td>28.5</td></tr>
<tr><th>CNN/DM</th><td>25.7</td><th>WOZ    </th><td>84.0</td></tr>
<tr><th>MNLI  </th><td>60.9</td><th>WikiSQL</th><td>45.8</td></tr>
<tr><th>SST   </th><td>85.9</td><th>MWSC   </th><td>52.4</td></tr>
</table>
</td>
</tr>

</table>


# Tasks
<hr>
The Natural Language Decathlon is a multitask challenge that spans ten tasks:

### Question Answering
Question answering (QA) models receive a question and a context that contains information necessary to output the desired answer. decaNLP uses the <a href="https://rajpurkar.github.io/SQuAD-explorer/">Stanford Question Answering Dataset (SQuAD 1.1)</a> as the dataset for this task.
### Machine Translation 
Machine translation models receive an input document in a source language that must be translated into a target language. decaNLP uses the 2016 English to German training data prepared for the <a href="https://wit3.fbk.eu/mt.php?release=2016-01">International Workshop on Spoken Language Translation (IWSLT)</a>, and we use the 2013 and 2014 test sets for our own validation and test sets, respectively. 

### Summarization
Summarization models take in a document and output a summary of that document. decaNLP uses the non-anonymized version of <a href="https://cs.nyu.edu/~kcho/DMQA/">CNN/DailyMail (CNN/DM) corpus</a>. We include the non-anonymized version of this dataset in decaNLP.

### Natural Language Inference
Natural Language Inference (NLI) models receive two input sentences: a premise and a hypothesis. Models must then output whether the premise entails contradicts or is neutral with respect to the hypothesis. decaNLP uses the <a href="https://www.nyu.edu/projects/bowman/multinli/">Multi-Genre Natural Language Inference Corpus (MNLI)</a>.

### Sentiment Analysis
Sentiment analysis models are trained to classify the sentiment expressed by input text. decaNLP includes the unparsed, binary version of the <a href="https://nlp.stanford.edu/sentiment/index.html">Stanford Sentiment Treebank (SST)</a>, which consists of movie reviews with corresponding sentiment (positive, neutral, negative).

### Semantic Role Labeling
Semantic role labeling (SRL) models are given a sentence and predicate (typically a verb) and must determine ‘who did what to whom,’ ‘when,’ and ‘where’. decaNLP includes the Wikipedia domain of <a href="https://dada.cs.washington.edu/qasrl/#dataset">QA-SRL 1.0</a>.

### Relation Extraction
Relation extraction systems take in a text document and the kind of relation that is to be extracted from that text. decaNLP includes <a href="http://nlp.cs.washington.edu/zeroshot/">QA-ZRE</a>, a dataset that maps relations to a set of questions so that relation extraction can be treated as question answering. At validation and test time, the relations are different than those seen in training, so that this is a zero-shot relation extraction task.

### Goal-Oriented Dialogue
Dialogue state tracking is a key component of goal-oriented dialogue systems. Based on user utterances and system actions, dialogue state trackers keep track of which predefined goals the user has for the dialogue system and which kinds of requests the user makes as the system and user interact turn-by-turn. decaNLP includes the English <a href="https://github.com/nmrksic/neural-belief-tracker/tree/master/data/woz">Wizard of Oz (WOZ)</a> restaurant reservation task, which comes with a predefined ontology of foods, dates, times, addresses, and other information that would help an agent make a reservation for a customer.

### Semantic Parsing
Semantic parsing requires models to translate unstructured information into structured formats so that users can interact with structured information (e.g. a database) in natural language . decaNLP includes the <a href="https://github.com/salesforce/WikiSQL">WikiSQL</a> dataset, which maps natural language questions into structured SQL queries.

### Commonsense Pronoun Resolution
As an example of commonsense pronoun resolution, models are required to answer questions like "Joan made sure to thank Susan for the help she had [given/received]. Who had [given/received] help? Susan or Joan?". We started with examples taken from the Winograd Schema Challenge and modified them (resulting in the <a href="https://github.com/salesforce/decaNLP/blob/master/local_data/schema.txt">Modified Winograd Schema Challenge, MWSC</a>) to ensure that answers were a single word from the context and scores are neither inflated nor deflated by oddities in phrasing or inconsistencies between context, question, and answer.


# FAQ
<hr>

### Do I have to use your code?
We have released all the code necessary to reproduce the main experiments from the paper.
We hope that this is useful to the community of researchers that choose to work on decaNLP, 
but it is not necessary to use our code base.
Feel free to use it as little or as much as you would like.

### What do I need to know about licensing?
Our code for running decaNLP has been open sourced under BSD-3-Clause.
We chose to restrict decaNLP to datasets that were free and publicly accessible for research,
but you should check their individual terms if you deviate from this use case.

### Do I have to treat everything as question answering?
We chose to treat everything as question anwering in our approach to decaNLP.
This choice comes with a set of tradeoffs,
but it can be considered independent of decaNLP itself.
We encourage exploration of alternative paradigms that allow training a single system on decaNLP,
e.g. language modeling or dialogue.
Our thought process suggested that dialogue could be considered an iterative question answering process,
and in early experiments language modeling did not seem as promising as question asnwering.

### What exactly do you mean by a single system?
In our work, we use what we consider to be a fully unified model with all parameters trained on all tasks jointly.
There is no task-specific modularization or parameters in our model, 
which allows the model to seamlessly attempt new tasks and learn relationships between tasks through natural language descriptions.
Single systems do not have to be unified to that degree, 
but they should not contain any explicit code that changes the behavior of the model based on the task.
Any kind of switching or conditional behavior should be learned.
For example, one could imagine a system of eleven submodels, ten of them are each a state-of-the-art model for a corresponding task in decaNLP,
and the eleventh is a classification network that learns to predict the task and routes examples to one of the other ten submodels for processing.
With the classificaiton network, this qualifies as a single system, but manually routing tasks based on an unlearned task id is not.

### Are there any restrictions on data?
Use as much data from outside sources as you would like as long as that data does not explicitly include supervision on the test sets of each decaNLP task.
For example, it is fair game to augment decaNLP with extra machine translation data from WMT.

### Why did you not include task/dataset X?
NLP has been decomposed into many different kinds of tasks, and each one often has many different representative datasets.
For decaNLP, we chose ten tasks and corresponding datasets that capture a broad variety of desired behavior for general NLP models,
but if you have suggestions for additional tasks, please feel free to reach out..
We want to get as much feedback from the community as possible to inform the inevitable decaNLP 2.0, 
and we strongly encourage your involvement in our future work on multitask learning for NLP.

### Where should I ask questions about the code?
Check out <a href="github.com/salesforce/decaNLP">github.com/salesforce/decaNLP</a> and post an issue if you're question is not answered in the README or previous issues.

### How do I evaluate on decaNLP?
Training and validation can all be completed using our code at github.com/salesforce/decaNLP,
but testing is more involved because decaNLP contains SQuAD 1.0 and MultiNLI as tasks, 
both of which have private, held-out test sets.
For those two tasks, inference on the test sets must be obtained through their systems, which will provide testing metrics.
Additionally, for WikiSQL, inference on the test set can be performed in the decaNLP code, 
but the official evaluation system at github.com/salesforce/WikiSQL should be used for the testing metric.

### How can I work more closely with Salesforce Research?
For discussion or collaboration, please reach out to 
[Bryan McCann](https://bmccann.github.io/) ([bmccann@salesforce.com](mailto:bmccann@salesforce.com)) and [Nitish Shirish Keskar](https://keskarnitish.github.io/) ([nkeskar@salesforce.com](mailto:nkeskar@salesforce.com)).
We are also hiring for full time research scientists, research engineers, and research interns. 
See our <a href="https://einstein.ai/careers">careers</a> page for more information.


# Open Questions
<hr>

### Augmenting decaNLP with additional tasks
There are a large number of NLP tasks and datasets that are not included in decaNLP.
Which tasks and datasets would be most helpful to include alongside decaNLP in training a general NLP model?

### Continual Learning in decaNLP
In the original decaNLP paper, all tasks were trained together, but decaNLP can also be used to explore continual learning and transfer learning.
How important is the order of tasks learned in a continual learning setting? 
Multitask learning mitigates the effects of catastrophic forgetting, but in a continual learning setting this again becomes an issue.

### Analysis of relatedness between tasks
Which tasks are most related? Which tasks hurt/help other tasks? Is there any way to measure this kind of relatedness by looking at the internals of the model itself,
or is this kind of analysis restricted to empirical observation through transfer learning?

### Zero-shot and Domain Adaptation
How can decaNLP be adapated to enhance a model's ability to adapt to new tasks or perform them without any fine-tuning?

### Architecture Search for decaNLP
Does the multitask setting provided by decaNLP provide a unique opportunity for architecture search to find general models?

### Multilingual decaNLP
Can we create a model that performs well across all of decaNLP in multiple languages?
This might start out by simply adding additional machine translation pairs to decaNLP, 
but eventually we would like a system that can perform all ten tasks in as many languages as possible.
Ideally, this would not require running an NMT system first before running a question answering system.
A truly multilingual model would be able to capitalize on its understanding of many languages to perform tasks in multiple language despite having only trained on that task in a single language.

### Scaling decaNLP
How can we adapt the training strategies and sampling procedures to scale decaNLP to larger (and more) datasets and also larger models? The viability of scaling strategies employed in single-task cases, such as large-batch training, asynchronous training, and delayed updates, remains to be shown for decaNLP.

### Robustness to paraphrasing
In the original formulation of decaNLP as question answering, some tasks used fixed questions like 'What is the summary?'. 
Ideally, a single model for NLP would also be robust to semantically equivalent questions:
'What is the summarization?',
'Can you summarize the text for me?',
'What are the most important points to remember?'.
Our first approach was robust to minor variations in the kinds of questions that could be asked,
but we found this robustness lacking.

### Analysis of multitask representations
The original decaNLP paper focuses on showing the potential of learning many complex tasks in a single model,
and experiments demonstrated that models trained on decaNLP were better at learning new tasks.
How does multitask learning change the kinds of representations the model learns internally?

# Citation
<hr>
If you use this in your work, please cite [*The Natural Language Decathlon: Multitask Learning as Question Answering*](https://arxiv.org/abs/1806.08730).

```
@article{McCann2018decaNLP,
  title={The Natural Language Decathlon: Multitask Learning as Question Answering},
  author={Bryan McCann and Nitish Shirish Keskar and Caiming Xiong and Richard Socher},
  journal={arXiv preprint arXiv:1806.08730},
  year={2018}
}
```

# Contact
<hr>

Contact: [Bryan McCann](https://bmccann.github.io/) ([bmccann@salesforce.com](mailto:bmccann@salesforce.com)) and [Nitish Shirish Keskar](https://keskarnitish.github.io/) ([nkeskar@salesforce.com](mailto:nkeskar@salesforce.com))
