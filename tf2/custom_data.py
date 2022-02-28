from operator import index
import tensorflow as tf
import os
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import pickle
from glob import glob
from absl import logging


def getBuilder(dataset, *args, **kwargs):
    if dataset == 'mvtech':
        return MVTechBuilder(dataset, *args, **kwargs)
    elif dataset == 'bmw':
        return BMWBuilder(dataset, *args, **kwargs)
    else:
        return tfds.builder(dataset, *args, **kwargs)


class StandardBuilder():

    def __init__(self, *args, use_all_data=True, min_fraction_anomalies=0.8, train_test_ratio=2, **kwargs):
        self.use_all_data = use_all_data #kwargs.get('use_all_data', True)
        self.min_fraction_anomalies = min_fraction_anomalies# kwargs.get('min_fraction_anomalies', 0.8)
        self.train_test_ratio = train_test_ratio# kwargs.get('train_test_ratio', 0.2)
        self._info = None
        

    def split_data_set(self, data_frame, neg_mask, pos_mask):
        if not self.use_all_data:
                dt_df = data_frame[neg_mask]
                self.min_fraction_anomalies = 0.0  # not used
        else:
            dt_df = data_frame

        if self.min_fraction_anomalies <= 0.0:
            train_df = dt_df.sample(frac=1 - self.train_test_ratio, replace=False, axis=0)
        else:
            pos_incl = data_frame[pos_mask]
            pos_incl = pos_incl.sample(frac=self.min_fraction_anomalies, replace=False, axis=0)
            """
            if self.min_fraction_anomalies < self.train_test_ratio:
                # include more IOs
                self.train_test_ratio = self.train_test_ratio + \
                                        (self.train_test_ratio - self.min_fraction_anomalies)
            """
            #
            train_df = data_frame[neg_mask]                
            train_df = train_df.sample(frac=1 - self.train_test_ratio, replace=False, axis=0)
            train_df = pd.concat([train_df, pos_incl])
            train_df = train_df.sample(frac=1)

        test_df = data_frame.drop(index=train_df.index)

        logging.info('total images', data_frame.shape[0])
        with open(os.path.join(self.results_dir, "split.pkl"), "wb") as f:
            pickle.dump((train_df, test_df), f)

        return (train_df, test_df)

    def prepare_dataset(self, train_df, test_df):
        logging.info('train images', train_df.shape[0])
        logging.info('test images', test_df.shape[0])
        
        train_paths = [os.path.join(self.path, file_name) for file_name in train_df.index.values]
        test_paths = [os.path.join(self.path, file_name) for file_name in test_df.index.values]

        # use filename as index
        train_ds = tf.data.Dataset.from_tensor_slices(list(zip(train_paths, train_df.lbl)))
        test_ds = tf.data.Dataset.from_tensor_slices(list(zip(test_paths, test_df.lbl)))

        info = Map({
            'splits': Map({
                'train': Map({
                    'num_examples': train_df.shape[0]
                }),
                'test': Map({
                    'num_examples': test_df.shape[0]
                })
            }),
            'features': Map({
                'label': Map({
                    'num_classes': 2
                })
            })
        })

        return {
            'info': info,
            'train_ds': train_ds,
            'test_df': test_ds
        }


class MVTechBuilder(StandardBuilder):
    """
    This pretends do be a builder.
    `DatasetBuilder` has 3 key methods:
    * `DatasetBuilder.info`: documents the dataset, including feature
        names, types, and shapes, version, splits, citation, etc.
    * `DatasetBuilder.download_and_prepare`: downloads the source data
        and writes it to disk.
    * `DatasetBuilder.as_dataset`: builds an input pipeline using
        `tf.data.Dataset`s.
    """

    def __init__(self, dataset, data_dir, *args, **kwargs):
        super().__init__(self, **kwargs)
        print(kwargs)
        self.dataset = dataset
        self.path = os.path.join(data_dir, '*')

    def download_and_prepare(self):
        self._load_mvtech_dataset()

    @property
    def info(self):
        if self._info == None:
            raise ValueError('info is None. Call download_and_prepare() first.')
        return self._info

    def as_dataset(self, split=None, batch_size=None, shuffle_files=None, as_supervised=False, read_config=None):

        AUTOTUNE = tf.data.AUTOTUNE

        def get_label(status):
            # Convert the path to a list of path components
            # parts = tf.strings.split(file_path, os.path.sep)
            # The second to last is the class-directory
            l = status != 'good'
            # Integer encode the label
            return int(l)  # tf.argmax(one_hot)

        def decode_img(img):
            # Convert the compressed string to a 3D uint8 tensor
            img = tf.io.decode_png(img, channels=3)
            # Resize the image to the desired size for testing
            # this is not needed because it's already done in build input func
            # img = tf.image.resize(img, [64, 64])
            return img

        def process(tpl):
            label = get_label(tpl[1])
            # Load the raw data from the file as a string
            img = tf.io.read_file(tpl[0])
            img = decode_img(img)
            return img, label
        
        if split == 'train':
            dataset = self.train_ds
        elif split == 'test':
            dataset = self.test_ds
        else:
            raise ValueError('Splits needs to be either train or test.')

        return dataset.map(process, num_parallel_calls=AUTOTUNE)

    def _load_mvtech_dataset(self):

        neg_files = glob(os.path.join(self.path, 'train', 'good', '*.png'))
        pos_files = glob(os.path.join(self.path, 'test', '*', '*.png'))

        neg_df = pd.DataFrame(data={'lbl': ['good'] * len(neg_files)}, index=neg_files)
        pos_df = pd.DataFrame(data={'lbl': ['bad'] * len(pos_files)}, index=pos_files)

        df = pd.concat(neg_df, pos_df)
        neg_mask = neg_df.lbl.values == 'good'
        pos_mask = pos_df.lbl.values == 'bad'

        train_df, test_df = self.split_data_set(df, neg_mask, pos_mask)

        res = self.prepare_dataset(train_df, test_df)
        self.train_ds = res['train_df']
        self.test_ds = res['test_df']
        self._info = res['info']


class BMWBuilder(StandardBuilder):
    """
    This pretends do be a builder.
    `DatasetBuilder` has 3 key methods:
    * `DatasetBuilder.info`: documents the dataset, including feature
        names, types, and shapes, version, splits, citation, etc.
    * `DatasetBuilder.download_and_prepare`: downloads the source data
        and writes it to disk.
    * `DatasetBuilder.as_dataset`: builds an input pipeline using
        `tf.data.Dataset`s.
    """

    def __init__(self, dataset, data_dir,
                 load_existing_split=False,
                 results_dir=None,
                 **kwargs,
                 ):
        super().__init__(self, **kwargs)
        self.dataset = dataset
        self.path = data_dir
        self.load_existing_split=load_existing_split
        self.results_dir=results_dir

        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)

    def download_and_prepare(self):
        self._load_bmw_dataset()

    @property
    def info(self):
        if self._info is None:
            raise ValueError('info is None. Call download_and_prepare() first.')
        return self._info

    def as_dataset(self, split=None, batch_size=None, shuffle_files=None, as_supervised=False, read_config=None):

        # AUTOTUNE = tf.data.AUTOTUNE
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        def get_label(status):
            l = status != 'IO'
            # Integer encode the label
            return int(l)  # tf.argmax(one_hot)

        def decode_img(img):
            # Convert the compressed string to a 3D uint8 tensor
            # img = tf.io.decode_png(img, channels=3)
            img = tf.io.decode_jpeg(img, channels=3)
            # Resize the image to the desired size for testing
            # this is not needed because it's already done in build input func
            # img = tf.image.resize(img, [64, 64])
            return img

        def process(tpl):
            label = get_label(tpl[1])
            # Load the raw data from the file as a string
            img = tf.io.read_file(tpl[0])
            img = decode_img(img)
            return img, label

        if split == 'train':
            dataset = self.train_ds
        elif split == 'test':
            dataset = self.test_ds
        else:
            raise ValueError('Splits needs to be either train or test.')

        return dataset.map(process, num_parallel_calls=AUTOTUNE)

    def _load_bmw_dataset(self):
        if not self.load_existing_split:
            annotations = pd.read_csv(os.path.join(self.path, 'annotation.csv'), index_col='file_name')
            neg_mask = annotations.lbl.values == 'IO'
            pos_mask = annotations.lbl.values != 'IO'

            train_df, test_df = self.split_data_set(annotations, neg_mask, pos_mask)
        else:
            logging.info("loading existing split from {}".format(os.path.join(self.results_dir, "split.pkl")))
            with open(os.path.join(self.results_dir, "split.pkl"), "rb") as f:
                (train_df, test_df) = pickle.load(f)

        res = self.prepare_dataset(train_df, test_df)
        self.train_ds = res['train_df']
        self.test_ds = res['test_df']
        self._info = res['info']


class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]
