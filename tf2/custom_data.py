import tensorflow as tf
import os
import tensorflow_datasets as tfds

img_height = 256
img_width = 256
batch_size = 32


class CustomBuilder():
    """This pretends do be a builder
  `DatasetBuilder` has 3 key methods:
    * `DatasetBuilder.info`: documents the dataset, including feature
        names, types, and shapes, version, splits, citation, etc.
    * `DatasetBuilder.download_and_prepare`: downloads the source data
        and writes it to disk.
    * `DatasetBuilder.as_dataset`: builds an input pipeline using
        `tf.data.Dataset`s.
    """

    custom_datasets = [
        'pill'
    ]

    @staticmethod
    def getBuilder(dataset, **kwargs):
        if dataset in CustomBuilder.custom_datasets:
            return CustomBuilder(dataset, **kwargs)
        else:
            return tfds.builder(dataset, **kwargs)


    def __init__(self, dataset, data_dir=None):
        print(dataset)
        self.dataset = dataset
        self.path = f'../{dataset}'
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

        def decode_img(img):
            # Convert the compressed string to a 3D uint8 tensor
            img = tf.io.decode_jpeg(img, channels=3)
            # Resize the image to the desired size
            return img # tf.image.resize(img, [img_height, img_width])

        def process_path(file_path):
            label = 1
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

       
        # self.train_ds = tf.keras.utils.image_dataset_from_directory(
        # os.path.join(self.path, 'train'),
        # labels=None,
        # label_mode=None,
        # seed=123,
        # image_size=(img_height, img_width),
        # batch_size=batch_size)

        self.train_ds = tf.data.Dataset.list_files(os.path.join(self.path, 'train', 'good', '*.png'))
        print(self.train_ds.take(1))

        # self.test_ds = tf.keras.utils.image_dataset_from_directory(
        # os.path.join(self.path, 'test'),
        # labels=None,
        # label_mode=None,
        # validation_split=0.0,
        # seed=123,
        # image_size=(img_height, img_width),
        # batch_size=batch_size)

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

