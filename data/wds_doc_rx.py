from pathlib import Path

import numpy as np
import cv2

from albumentations.pytorch import ToTensorV2

import webdataset as wds

import random


def create_wds_ortho_docs_rx(data_dir,
                             prefix="ortho_coupled",
                             n_val_shards=1,
                             image_transform=None,
                             text_tokenizer=None,
                             max_study_images=10,
                             limit_random_first_n_images=12,
                             length=None,
                             n_samples_per_shard=4096,
                             ):
    """
    Create webdataset for ortho docs rx
    :param data_dir: data directory
    :param prefix: prefix of the webdataset shards files
    :param n_val_shards: number of validation shards
    :param image_transform: image transform
    :param text_tokenizer: text tokenizer
    :param max_study_images: maximum number of study images to return
    :param limit_random_first_n_images: the choice of images to return is made from the first n images of the study,
                                        this parameter limits the number of images to consider and is useful for
                                        studies with a large number of images that may be irrelevant
                                        (e.g. fluoroscopy guided injections)
    :param length: length of the dataset (useful since webdataset does not have a length property by default)
    :param n_samples_per_shard: number of samples per shard (used for length calculation after validation split),
                                we assume that the number of samples per shard is the same for all shards,
                                or at least for the first n_val_shards shards
    """
    file_list = list(Path(data_dir).glob(prefix + ".*.tar"))
    file_list = [str(path) for path in file_list]
    file_list.sort()

    val_file_list = file_list[:n_val_shards]
    train_file_list = file_list[n_val_shards:]

    if image_transform is None:
        image_transform = ToTensorV2()
    else:
        image_transform = image_transform

    def preprocess(sample):
        text = sample["txt"].decode("utf8")
        images = []

        images_keys = [key for key in sample.keys() if ".jpg" in key]
        images_keys = images_keys[:limit_random_first_n_images]
        images_keys = random.sample(images_keys, min(max_study_images, len(images_keys)))

        for image_key in images_keys:
            nparr = np.frombuffer(sample[image_key], np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            images.append(image_transform(image=img_np)['image'])

        if text_tokenizer is not None:
            text = text_tokenizer(text, truncation=True)

        return text, images

    train_dataset = (
        wds.WebDataset(train_file_list)
        .shuffle(1000)
        .map(preprocess)
    )

    val_dataset = (
        wds.WebDataset(val_file_list)
        .map(preprocess)
    )

    if length is not None:
        train_dataset = train_dataset.with_length(length - n_val_shards * n_samples_per_shard)
        val_dataset = val_dataset.with_length(n_samples_per_shard * n_val_shards)

    return train_dataset, val_dataset


class BatchedWebLoaderLen(wds.WebLoader):
    """
    Custom loader for webdataset with length property
    """
    def __init__(self, dataset, **kwargs):
        batch_size = kwargs.get('batch_size', 1)

        kwargs['batch_size'] = None
        super().__init__(dataset, **kwargs)

        if hasattr(dataset, '__len__'):
            if kwargs.get('drop_last', False):
                self.length = len(dataset) // batch_size
            else:
                self.length = (len(dataset) + batch_size - 1) // batch_size

    def __len__(self):
        return self.length
