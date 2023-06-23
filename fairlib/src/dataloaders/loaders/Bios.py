import numpy as np
from ..utils import BaseDataset
from pathlib import Path
import pandas as pd
import sys

class BiosDataset(BaseDataset):
    embedding_type = "bert_avg_SE"
    text_type = "hard_text"

    def load_data(self):
        self.filename = "bios_{}_df.pkl".format(self.split)

        data = pd.read_pickle(Path(self.args.data_dir) / self.filename)

        # data = data.sample(frac=0.01, random_state=42).reset_index()

        # if self.args.protected_task in ["economy", "both"] and self.args.full_label:
        if self.args.protected_task in ["economy", "both", "intersection"] and self.args.full_label:
            selected_rows = (data["economy_label"] != "Unknown")
            data = data[selected_rows]

        if self.args.encoder_architecture == "Fixed":
            self.X = list(data[self.embedding_type])
        elif self.args.encoder_architecture == "BERT":
            print('here')
            _input_ids, _token_type_ids, _attention_mask = self.args.text_encoder.encoder(list(data[self.text_type]))
            self.X = _input_ids
            self.addition_values["input_ids"] = _input_ids
            self.addition_values['attention_mask'] = _attention_mask
            self.addition_values['bert_representation'] = list(data[self.embedding_type])
        else:
            raise NotImplementedError

        self.y = data["profession_class"].astype(np.float64)  # Profession
        if self.args.protected_task == "gender":
            self.protected_label = data["gender_class"].astype(np.int32)  # Gender
        elif self.args.protected_task == "economy":
            self.protected_label = data["economy_class"].astype(np.int32)  # Economy
        elif self.args.protected_task == "intersection":
            self.protected_label = np.array(
                [2 * _e + _g for _e, _g in zip(list(data["economy_class"]), list(data["gender_class"]))]
            ).astype(np.int32)  # Intersection
        else:
            self.protected_label = data["intersection_class"].astype(np.int32)  # Intersection

"""import numpy as np
from ..utils import BaseDataset
from pathlib import Path
import pandas as pd
import sys


class BiosDataset(BaseDataset):

    embedding_type = "bert_avg_SE"
    text_type = "hard_text"
    demonic_embedding_type = "demonic_avg_SE"  # demonic_cls_SE

    def __init__(self, args, split, dataset_size=1):
        super().__init__(args, split)
        self.filename = None
        self.args = args
        self.split = split
        self.dataset_size = dataset_size

        self.embedding_type = "bert_avg_SE"
        self.text_type = "hard_text"
        self.demonic_embedding_type = "demonic_avg_SE"  # demonic_cls_SE

        self.X = []
        self.demX = []
        self.txt = []
        self.y = []
        self.protected_label = []
        self.instance_weights = []
        self.adv_instance_weights = []
        self.regression_label = []
        self.addition_values = {}

        self.load_data()

        self.regression_init()

        self.X = np.array(self.X)
        if len(self.X.shape) == 3:
            self.X = np.concatenate(list(self.X), axis=0)

        self.demX = np.array(self.demX)
        if len(self.demX.shape) == 3:
            self.demX = np.concatenate(list(self.demX), axis=0)

        self.y = np.array(self.y).astype(int)
        self.protected_label = np.array(self.protected_label).astype(int)

        self.manipulate_data_distribution()

        self.balanced_training()

        self.adv_balanced_training()

        if self.split == "train":
            self.adv_decoupling()

        print("Loaded data shapes: {}, {}, {}".format(self.X.shape, self.y.shape, self.protected_label.shape))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.y)

    def __getitem__(self, index):

        'Generates one sample of data'
        _X = self.X[index]
        _demX = self.demX[index]
        _y = self.y[index]
        _protected_label = self.protected_label[index]
        _instance_weights = self.instance_weights[index]
        _adv_instance_weights = self.adv_instance_weights[index]
        _regression_label = self.regression_label[index]

        data_dict = {
            0: _X,
            1: _y,
            2: _protected_label,
            3: _instance_weights,
            4: _adv_instance_weights,
            5: _regression_label,
            6: _demX,
        }
        for _k in self.addition_values.keys():
            if _k not in data_dict.keys():
                data_dict[_k] = self.addition_values[_k][index]
        return data_dict

    def load_data(self):

        # print("here")
        self.filename = "demon_bios_{}_df.pkl".format(self.split) # demon_

        data = pd.read_pickle(Path(self.args.data_dir) / self.filename)

        # data = data.sample(frac=0.3, random_state=42).reset_index()

        if self.args.protected_task in ["economy", "both"] and self.args.full_label:
            # if self.args.protected_task in ["gender", "economy", "both", "intersection"] and self.args.full_label:
            selected_rows = (data["economy_label"] != "Unknown")
            data = data[selected_rows]

        if self.args.encoder_architecture == "Fixed":
            self.X = list(data[self.embedding_type])
            self.demX = list(data[self.demonic_embedding_type])
            self.txt = list(data[self.text_type])
        elif self.args.encoder_architecture == "BERT":
            _input_ids, _token_type_ids, _attention_mask = self.args.text_encoder.encoder(list(data[self.text_type]))
            self.X = _input_ids
            self.addition_values["input_ids"] = _input_ids
            self.addition_values['attention_mask'] = _attention_mask
        else:
            raise NotImplementedError

        self.y = data["profession_class"].astype(np.float64)  # Profession
        if self.args.protected_task == "gender":
            self.protected_label = data["gender_class"].astype(np.int32)  # Gender
        elif self.args.protected_task == "economy":
            self.protected_label = data["economy_class"].astype(np.int32)  # Economy
        elif self.args.protected_task == "intersection":
            self.protected_label = np.array(
                [2 * _e + _g for _e, _g in zip(list(data["economy_class"]), list(data["gender_class"]))]
            ).astype(np.int32)  # Intersection
        else:
            self.protected_label = data["intersection_class"].astype(np.int32)  # Intersection
"""
