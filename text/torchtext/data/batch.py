from collections import defaultdict
from torch.autograd import Variable
from copy import deepcopy



class Batch(object):
    """Defines a batch of examples along with its Fields.

    Attributes:
        batch_size: Number of examples in the batch.
        dataset: A reference to the dataset object the examples come from
            (which itself contains the dataset's Field objects).
        train: Whether the batch is from a training set.

    Also stores the Variable for each column in the batch as an attribute.
    """

    def __init__(self, data=None, dataset=None, device=None, train=True):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            self.dataset = dataset
            self.train = train
            field = list(dataset.fields.values())[0]
            limited_idx_to_full_idx = deepcopy(field.decoder_to_vocab) # should avoid this with a conditional in map to full
            oov_to_limited_idx = {}
            for (name, field) in dataset.fields.items():
                if field is not None:
                    batch = [x.__dict__[name] for x in data]
                    if not field.include_lengths:
                        setattr(self, name, field.process(batch, device=device, train=train))
                    else:
                        entry, lengths, limited_entry, raw = field.process(batch, device=device, train=train, 
                            limited=field.decoder_stoi, l2f=limited_idx_to_full_idx, oov2l=oov_to_limited_idx)
                        setattr(self, name, entry)
                        setattr(self, f'{name}_lengths', lengths)
                        setattr(self, f'{name}_limited', limited_entry)
                        setattr(self, f'{name}_elmo', [[s.strip() for s in l] for l in raw])
            setattr(self, f'limited_idx_to_full_idx', limited_idx_to_full_idx)
            setattr(self, f'oov_to_limited_idx', oov_to_limited_idx)


    @classmethod
    def fromvars(cls, dataset, batch_size, train=True, **kwargs):
        """Create a Batch directly from a number of Variables."""
        batch = cls()
        batch.batch_size = batch_size
        batch.dataset = dataset
        batch.train = train
        for k, v in kwargs.items():
            setattr(batch, k, v)
        return batch
