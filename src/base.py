import os
import json

import pandas as pd
import xgboost as xgb
import numpy as np

from collections import defaultdict
from abc import ABC, abstractmethod


class Utilities:
    @staticmethod
    def load_json(json_file):
        if not os.path.exists(json_file):
            print('File does not exist: {}'.format(json_file))
            return None
        with open(json_file, 'r') as f:
            file_contents = json.load(f)
        return file_contents

    @staticmethod
    def create_class_mapping(df, str_columns, class_mapping_file):
        class_mapping_set = defaultdict(set)
        class_mapping = defaultdict(dict)

        for column in str_columns:
            df[column].apply(class_mapping_set[column].add)
            class_mapping[column] = {v: k for k, v in enumerate(sorted(list(map(str, class_mapping_set[column]))))}

        with open(class_mapping_file, 'w') as f:
            json.dump(class_mapping, f, indent=4)
        return class_mapping


class Config:
    def __init__(self):
        self.home = os.path.dirname(os.getcwd())
        self.train_file = os.path.join(self.home, 'data', 'train.csv')
        self.config_file = os.path.join(self.home, 'src', 'base.json')
        self.params_file = os.path.join(self.home, "src", "params.json")
        self.class_mapping_file = os.path.join(self.home, 'data', 'mapping.json')
        self.notable_columns = Utilities.load_json(self.config_file)


class Transformation():
    def __init__(self, config, identifier_string="train", class_mapping=None, _columns=None):
        self.config = config
        file_name = os.path.join(self.config.home, 'data', '{}.csv'.format(identifier_string))

        self.df = pd.read_csv(file_name)
        self._string = identifier_string
        self._columns = _columns
        self.class_mapping = class_mapping

        self.apply_basic_cleaning()
        self.apply_basic_date_transforms()
        self.add_customized_feature_engineering()
        self.remove_unnecessary_columns()
        self.fill_empty_columns()
        self.apply_label_encoding()
        self.transform_for_xgboost()

        self.final_columns = list(self.df.columns)

    def apply_basic_cleaning(self):
        self.df.dropna(how='all', axis="columns", inplace=True)
        self.df.dropna(how='all', axis="rows", inplace=True)

    def apply_basic_date_transforms(self):
        for date_column in self.config.notable_columns["Date"]:
            self.df[date_column] = pd.to_datetime(self.df[date_column])

    def add_customized_feature_engineering(self):
        pass

    def mean_encoding(self):
        pass
        # means = dict()
        # for column in notable_columns["MeanEncode"]:
        #     means[column] = df.groupby([column])[notable_columns["Target"]].mean()
        #     df["{}_mean_target".format(column)] = df[column].map(means[column])

        # for column in notable_columns["MeanEncode"]:
        #     cum_sum = df.groupby(column)[notable_columns["Target"]].cumsum() - df[notable_columns["Target"]]
        #     cum_count = df.groupby(column).cumcount()
        #     df["{}_mean_target".format(column)] = cum_sum/cum_count

    def remove_unnecessary_columns(self):
        self.df = self.df[np.setdiff1d(self.df.columns, self.config.notable_columns["Remove"])]

        if self._columns is not None:
            self.df = self.df[self._columns + [self.config.notable_columns["ID"]]]

        if self._string == 'train':
            self.main_column = self.df[self.config.notable_columns["Target"]]
            self.df = self.df[np.setdiff1d(self.df.columns, self.config.notable_columns["Targets"])]
            self.df = self.df[np.setdiff1d(self.df.columns, [self.config.notable_columns["ID"]])]

        elif self._string == "test":
            self.main_column = self.df[self.config.notable_columns["ID"]]
            self.df = self.df[np.setdiff1d(self.df.columns, [self.config.notable_columns["ID"]])]

    def fill_empty_columns(self):
        has_null_columns = self.df.columns[self.df.isnull().any()]

        str_columns = self.df[has_null_columns].select_dtypes(include=['object']).columns
        int_columns = self.df[has_null_columns].select_dtypes(exclude=["object"]).columns

        self.df[int_columns].fillna(-999, inplace=True)
        self.df[str_columns].fillna('', inplace=True)

    def apply_label_encoding(self):
        str_columns = self.df.select_dtypes(include=['object']).columns

        if self.class_mapping is None:
            self.class_mapping = Utilities.create_class_mapping(self.df, str_columns, self.config.class_mapping_file)

        for column in str_columns:
            self.df[column] = self.df[column].apply(lambda x: self.class_mapping[column][x]
                                                    if x in self.class_mapping[column]
                                                    else 0)

    def transform_for_xgboost(self):
        if self._string == "train":
            self.ddata = xgb.DMatrix(self.df.as_matrix(), self.main_column)
        elif self._string == "test":
            self.ddata = xgb.DMatrix(self.df.as_matrix())



