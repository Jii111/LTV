from .basetask import BaseTask

from .sst2 import SST2
from .dbpedia import DBPedia
from .sst5 import SST5
from .trec import TREC
from .agnews import AGNews
from .subj import Subj
from .rotten_tomatoes import RottenTomatoes
from .hate_speech18 import HateSpeech18


target_datasets = {
    'agnews': AGNews,
    'dbpedia': DBPedia,
    'sst5': SST5,
    'trec': TREC,
    'sst2': SST2,
    'subj': Subj,
    'mr': RottenTomatoes,
    'hate_speech18': HateSpeech18,
}

dataset_dict = {}
dataset_dict.update(target_datasets)


def get_dataset(dataset, *args, **kwargs) -> BaseTask:
    num_train_dataset = None
    if "num_train_dataset" in kwargs:
        num_train_dataset = kwargs.pop("num_train_dataset")
    dataset = dataset_dict[dataset](task_name=dataset, *args, **kwargs)
    if num_train_dataset is not None:
        from torch.utils.data import Subset

        # Create indices for the first N samples
        indices = list(range(min(num_train_dataset, len(dataset.dataset))))
        dataset.dataset = Subset(dataset.dataset, indices)
        dataset.all_data = [data for data in dataset.dataset]
        print(f"Dataset size adjusted: using {len(indices)} samples (requested: {num_train_dataset})")
    return dataset
