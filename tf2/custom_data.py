import tensorflow as tf
import os
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd

def getBuilder(dataset, **kwargs):
        if dataset == 'mvtech':
            return MVTechBuilder(dataset, **kwargs)
        elif dataset == 'bmw':
            return BMWBuilder(dataset, **kwargs)
        else:
            return tfds.builder(dataset, **kwargs)

class MVTechBuilder():
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


    def __init__(self, dataset, data_dir=None):
        self.dataset = dataset
        self.path = os.path.join(data_dir, '*')
        self._info = None

    def download_and_prepare(self):
        self._load_mvtech_dataset()

    @property
    def info(self):
        if self._info == None: 
            raise ValueError('info is None. Call download_and_prepare() first.') 
        return self._info

    def as_dataset(self, split=None, batch_size=None, shuffle_files=None, as_supervised=False, read_config=None):

        AUTOTUNE = tf.data.AUTOTUNE

        def get_label(file_path):
            # Convert the path to a list of path components
            parts = tf.strings.split(file_path, os.path.sep)
            # The second to last is the class-directory
            l = parts[-2] != 'good'
            # Integer encode the label
            return int(l) # tf.argmax(one_hot)

        def decode_img(img):
            # Convert the compressed string to a 3D uint8 tensor
            img = tf.io.decode_png(img, channels=3)
            # Resize the image to the desired size for testing
            # this is not needed because it's already done in build input func
            # img = tf.image.resize(img, [64, 64])
            return img

        def process_path(file_path):
            label = get_label(file_path)
            # Load the raw data from the file as a string
            img = tf.io.read_file(file_path)
            img = decode_img(img)
            return img, label

        if split=='train':
            dataset =  self.train_ds
        elif split=='test':
            dataset = self.test_ds
        else:
            raise ValueError('Splits needs to be either train or test.')

        return dataset.map(process_path, num_parallel_calls=AUTOTUNE)

    def _load_mvtech_dataset(self):
        self.train_ds = tf.data.Dataset.list_files(os.path.join(self.path, 'train', 'good', '*.png'))
        self.test_ds = tf.data.Dataset.list_files(os.path.join(self.path, 'test', '*','*.png'))

        self._info = Map({
            'splits': Map({
                'train': Map({
                    'num_examples': self.train_ds.cardinality().numpy()
                }),
                'test': Map({
                    'num_examples': self.test_ds.cardinality().numpy()
                })
            }),
            'features': Map({
                'label': Map({
                    'num_classes': 2
                })
            })
        })


class BMWBuilder():
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


    def __init__(self, dataset, data_dir=None):
        self.dataset = dataset
        self.path = data_dir
        self._info = None

    def download_and_prepare(self):
        self._load_bmw_dataset()

    @property
    def info(self):
        if self._info == None: 
            raise ValueError('info is None. Call download_and_prepare() first.') 
        return self._info

    def as_dataset(self, split=None, batch_size=None, shuffle_files=None, as_supervised=False, read_config=None):

        AUTOTUNE = tf.data.AUTOTUNE

        def get_label(status):
            l = status != 'IO'
            # Integer encode the label
            return int(l) # tf.argmax(one_hot)

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

        if split=='train':
            dataset =  self.train_ds
        elif split=='test':
            dataset = self.test_ds
        else:
            raise ValueError('Splits needs to be either train or test.')

        return dataset.map(process, num_parallel_calls=AUTOTUNE)

    def _load_bmw_dataset(self):
        self.annotations = pd.read_csv(os.path.join(self.path, 'annotation.csv'), index_col='file_name')
        io_mask = self.annotations.lbl.values == 'IO'
        io_df = self.annotations[io_mask]

        frac = 0.2
        train_df = io_df.sample(frac=1-frac, replace=False, axis=0)
        test_df = self.annotations.drop(index=train_df.index)

        print('total images', self.annotations.shape[0])
        print('train images', train_df.shape[0])
        print('test images', test_df.shape[0])

        train_paths = [os.path.join(self.path, file_name) for file_name in train_df.index.values]
        test_paths = [os.path.join(self.path, file_name) for file_name in test_df.index.values]

        # use filename as index
        self.train_ds = tf.data.Dataset.from_tensor_slices(list(zip(train_paths, train_df.lbl)))
        self.test_ds =tf.data.Dataset.from_tensor_slices(list(zip(test_paths, test_df.lbl)))

        self._info = Map({
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

