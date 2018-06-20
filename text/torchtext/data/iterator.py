from __future__ import division
import time

import math
import random
from contextlib import contextmanager
from copy import deepcopy

import torch
from torch.distributed import get_world_size, get_rank

from .batch import Batch
from .dataset import Dataset


class RandomShuffler(object):
    """Use random functions while keeping track of the random state to make it
    reproducible and deterministic."""

    def __init__(self, random_state=None):
        self._random_state = random_state
        self.extra = 0
        if self._random_state is None:
            self._random_state = random.getstate()

    @contextmanager
    def use_internal_state(self):
        """Use a specific RNG state."""
        old_state = random.getstate()
        random.setstate(self._random_state)
        yield
        self._random_state = random.getstate()
        random.setstate(old_state)

    @property
    def random_state(self):
        return deepcopy(self._random_state)

    @random_state.setter
    def random_state(self, s):
        self._random_state = s

    def __call__(self, data, subsample=None):
        """Shuffle and return a new list."""
        with self.use_internal_state():
            return random.sample(data, len(data))

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedShuffler:

    def __init__(self, num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.extra = 0

    def __call__(self, data, subsample=True):

        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = list(torch.randperm(len(data), generator=g))
        if not subsample:
            return [data[i] for i in indices]
        return [data[i] for i in self.subsample(indices)]

    def subsample(self, indices):
        # add extra samples to make it evenly divisible
        num_samples = int(math.ceil(len(indices) * 1.0 / self.num_replicas))
        total_size = num_samples * self.num_replicas
        extras = indices[:(total_size - len(indices))]
        self.extra = len(extras)
        indices += extras
        assert len(indices) == total_size

        # subsample
        offset = num_samples * self.rank
        indices = indices[offset:offset + num_samples]
        assert len(indices) == num_samples

        return indices

    def set_epoch(self, epoch):
        self.epoch = epoch


class Iterator(object):
    """Defines an iterator that loads batches of data from a Dataset.

    Attributes:
        dataset: The Dataset object to load Examples from.
        batch_size: Batch size.
        batch_size_fn: Function of three arguments (new example to add, current
            count of examples in the batch, and current effective batch size)
            that returns the new effective batch size resulting from adding
            that example to a batch. This is useful for dynamic batching, where
            this function would add to the current effective batch size the
            number of tokens in the new example.
        sort_key: A key to use for sorting examples in order to batch together
            examples with similar lengths and minimize padding. The sort_key
            provided to the Iterator constructor overrides the sort_key
            attribute of the Dataset, or defers to it if None.
        train: Whether the iterator represents a train set.
        repeat: Whether to repeat the iterator for multiple epochs.
        shuffle: Whether to shuffle examples between epochs.
        sort: Whether to sort examples according to self.sort_key.
            Note that repeat, shuffle, and sort default to train, train, and
            (not train).
        sort_within_batch: Whether to sort (in descending order according to
            self.sort_key) within each batch. If None, defaults to self.sort.
            If self.sort is True and this is False, the batch is left in the
            original (ascending) sorted order.
        device: Device to create batches on. Use -1 for CPU and None for the
            currently active GPU device.
    """

    def __init__(self, dataset, batch_size, sort_key=None, device=None,
                 batch_size_fn=None, train=True,
                 repeat=None, shuffle=None, sort=None, reverse=False,
                 sort_within_batch=None, distributed=False, num_replicas=None, rank=None):
        self.batch_size, self.train, self.dataset = batch_size, train, dataset
        self.batch_size_fn = batch_size_fn
        self.iterations = 0
        self.epoch = 0
        self.reverse = reverse
        self.repeat = train if repeat is None else repeat
        self.shuffle = train if shuffle is None else shuffle
        self.sort = not train if sort is None else sort
        if sort_within_batch is None:
            self.sort_within_batch = self.sort
        else:
            self.sort_within_batch = sort_within_batch
        if sort_key is None:
            self.sort_key = dataset.sort_key
        else:
            self.sort_key = sort_key
        self.device = device

        self.distributed = distributed
        if distributed:
            self.random_shuffler = DistributedShuffler(num_replicas=num_replicas, rank=rank)
        else:
            self.random_shuffler = RandomShuffler() 

        # For state loading/saving only
        self._iterations_this_epoch = 0
        self._random_state_this_epoch = None
        self._restored_from_state = False

    @classmethod
    def splits(cls, datasets, batch_sizes=None, **kwargs):
        """Create Iterator objects for multiple splits of a dataset.

        Arguments:
            datasets: Tuple of Dataset objects corresponding to the splits. The
                first such object should be the train set.
            batch_sizes: Tuple of batch sizes to use for the different splits,
                or None to use the same batch_size for all splits.
            Remaining keyword arguments: Passed to the constructor of the
                iterator class being used.
        """
        if batch_sizes is None:
            batch_sizes = [kwargs.pop('batch_size')] * len(datasets)
        ret = []
        for i in range(len(datasets)):
            train = i == 0
            ret.append(cls(
                datasets[i], batch_size=batch_sizes[i], train=train, **kwargs))
        return tuple(ret)

    def data(self):
        """Return the examples in the dataset in order, sorted, or shuffled."""
        if self.sort:
            xs = sorted(self.dataset, key=self.sort_key, reverse=self.reverse)
            if self.distributed:
                xs = [xs[i] for i in self.random_shuffler.subsample(list(range(len(xs))))]
        elif self.shuffle:
            xs = self.random_shuffler(list(self.dataset))
        else:
            xs = self.dataset
        self.extra = self.random_shuffler.extra
        return xs

    def init_epoch(self):
        """Set up the batch generator for a new epoch."""
        if not self.distributed:
            if self._restored_from_state:
                self.random_shuffler.random_state = self._random_state_this_epoch
            else:
                self._random_state_this_epoch = self.random_shuffler.random_state

        self.create_batches()

        if not self.distributed:
            if self._restored_from_state:
                self._restored_from_state = False
            else:
                self._iterations_this_epoch = 0
        else:
            self._iterations_this_epoch = 0


        if not self.repeat:
            self.iterations = 0
        self.epoch += 1
        if self.distributed:
            self.random_shuffler.set_epoch(self.epoch)

    def create_batches(self):
        self.batches = batch(self.data(), self.batch_size, self.batch_size_fn)

    def __len__(self):
        if self.batch_size_fn is not None:
            raise NotImplementedError
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a minibatch
                    # be sorted by decreasing order, which requires reversing
                    # relative to typical sort keys
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=self.reverse)
                b =  Batch(minibatch, self.dataset, self.device,
                            self.train)
                yield b
            if not self.repeat:
                return

    def state_dict(self):
        d = {"iterations": self.iterations}
        if not self.distributed:
            d.update({
                "iterations_this_epoch": self._iterations_this_epoch,
                "random_state_this_epoch": self._random_state_this_epoch
            })

    def load_state_dict(self, state_dict):
        self.iterations = state_dict["iterations"]
        self._iterations_this_epoch = state_dict["iterations_this_epoch"]

        if not self.distributed:
            self._random_state_this_epoch = state_dict["random_state_this_epoch"]
            self._restored_from_state = True


class BPTTIterator(Iterator):
    """Defines an iterator for language modeling tasks that use BPTT.

    Provides contiguous streams of examples together with targets that are
    one timestep further forward, for language modeling training with
    backpropagation through time (BPTT). Expects a Dataset with a single
    example and a single field called 'text' and produces Batches with text and
    target attributes.

    Attributes:
        dataset: The Dataset object to load Examples from.
        batch_size: Batch size.
        bptt_len: Length of sequences for backpropagation through time.
        sort_key: A key to use for sorting examples in order to batch together
            examples with similar lengths and minimize padding. The sort_key
            provided to the Iterator constructor overrides the sort_key
            attribute of the Dataset, or defers to it if None.
        train: Whether the iterator represents a train set.
        repeat: Whether to repeat the iterator for multiple epochs.
        shuffle: Whether to shuffle examples between epochs.
        sort: Whether to sort examples according to self.sort_key.
            Note that repeat, shuffle, and sort default to train, train, and
            (not train).
        device: Device to create batches on. Use -1 for CPU and None for the
            currently active GPU device.
    """

    def __init__(self, dataset, batch_size, bptt_len, **kwargs):
        self.bptt_len = bptt_len
        super(BPTTIterator, self).__init__(dataset, batch_size, **kwargs)

    def __len__(self):
        return math.ceil((len(self.dataset[0].text) / self.batch_size - 1) /
                         self.bptt_len)

    def __iter__(self):
        text = self.dataset[0].text
        TEXT = self.dataset.fields['text']
        TEXT.eos_token = None
        text = text + ([TEXT.pad_token] * int(math.ceil(len(text) / self.batch_size) *
                                              self.batch_size - len(text)))
        data = TEXT.numericalize(
            [text], device=self.device, train=self.train)
        data = data.view(self.batch_size, -1).t().contiguous()
        dataset = Dataset(examples=self.dataset.examples, fields=[
            ('text', TEXT), ('target', TEXT)])
        while True:
            for i in range(0, len(self) * self.bptt_len, self.bptt_len):
                seq_len = min(self.bptt_len, len(data) - i - 1)
                yield Batch.fromvars(
                    dataset, self.batch_size, train=self.train,
                    text=data[i:i + seq_len],
                    target=data[i + 1:i + 1 + seq_len])
            if not self.repeat:
                raise StopIteration


class BucketIterator(Iterator):
    """Defines an iterator that batches examples of similar lengths together.

    Minimizes amount of padding needed while producing freshly shuffled
    batches for each new epoch. See pool for the bucketing procedure used.
    """

    def create_batches(self):
        if self.sort:
            self.batches = batch(self.data(), self.batch_size,
                                 self.batch_size_fn, repeat=self.repeat)
        else:
            self.batches = pool(self.data(), self.batch_size,
                                self.sort_key, self.batch_size_fn,
                                random_shuffler=self.random_shuffler, repeat=self.repeat,
                                reverse=self.reverse, shuffle=self.shuffle)


def batch(data, batch_size, batch_size_fn=None, repeat=False):
    """Yield elements from data in chunks of batch_size."""
    if batch_size_fn is None:
        def batch_size_fn(new, count, sofar):
            return count
    minibatch = []
    size_so_far = 0
    for ex in data:
        minibatch.append(ex)
        size_so_far = batch_size_fn(ex, len(minibatch), size_so_far)
        if size_so_far == batch_size: 
            yield minibatch
            minibatch, size_so_far = [], 0
        elif size_so_far > batch_size:
            if len(minibatch) == 1: # if we only have one really big example
                yield minibatch
                minibatch, size_so_far = [], 0
            else:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], batch_size_fn(ex, 1, 0)
                if size_so_far > batch_size: # if we add a really big example that needs to be on its own to a batch
                    yield minibatch
                    minibatch, size_so_far = [], 0
    if minibatch:
        yield minibatch


def pool(data, batch_size, key, batch_size_fn=lambda new, count, sofar: count,
         random_shuffler=None, reverse=False, shuffle=False, repeat=False, leftovers=None):
    """Sort within buckets, then batch, then shuffle batches.

    Partitions data into chunks of size 100*batch_size, sorts examples within
    each chunk using sort_key, then batch these examples and shuffle the
    batches.
    """
    if random_shuffler is None:
        random_shuffler = random.shuffle
    for p in batch(data, batch_size * 100, batch_size_fn):
        p_batch = batch(sorted(p, key=key, reverse=reverse), batch_size, batch_size_fn, repeat=repeat)
        if shuffle:
            for b in random_shuffler(list(p_batch), subsample=False):
                yield b
        else:
            for b in list(p_batch):
                yield b
