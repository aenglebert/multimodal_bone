from pathlib import Path

import numpy as np
import cv2

from albumentations.pytorch import ToTensorV2

import webdataset as wds


def create_wds_ortho_docs_rx(data_dir,
                             prefix="ortho_coupled",
                             image_transform=None,
                             text_tokenizer=None,
                             max_study_images=10,
                             length=None,
                             doc_embedding_npz=None,
                             ):
    file_list = list(Path(data_dir).glob(prefix + ".*.tar"))
    file_list = [str(path) for path in file_list]
    file_list.sort()

    if image_transform is None:
        image_transform = ToTensorV2()
    else:
        image_transform = image_transform

    def preprocess(sample):
        text = sample["txt"].decode("utf8")
        images = []

        images_keys = [key for key in sample.keys() if ".jpg" in key]
        for image_key in images_keys[:max_study_images]:
            nparr = np.frombuffer(sample[image_key], np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            images.append(image_transform(image=img_np)['image'])

        if text_tokenizer is not None:
            text = text_tokenizer(text, truncation=True)

        return text, images

    dataset = (
        wds.WebDataset(file_list)
        .shuffle(1000)
        .map(preprocess)
    )

    if length is not None:
        dataset = dataset.with_length(length)

    return dataset


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