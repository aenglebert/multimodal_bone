import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class OrthoDocRx(Dataset):
    def __init__(self,
                 data_dir,
                 image_transform=None,
                 text_tokenizer=None,
                 max_study_images=10,
                ):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "image_files"
        self.text_dir = self.data_dir / "text_files"

        if image_transform is None:
            self.image_transform = ToTensorV2()
        else:
            self.image_transform = image_transform

        self.text_tokenizer = text_tokenizer
        self.max_study_images = max_study_images

        self.studies = pd.read_csv(self.data_dir / "bone_rx_doc.csv")

    def __len__(self):
        return len(self.studies)

    def __getitem__(self,
                     idx,
                    ):
        study = self.studies.loc[idx]

        patient_id = study.patient_id
        doc_file = self.text_dir / patient_id / (study.doc_id + ".txt")
        image_folder = self.image_dir / patient_id / study.study_id

        image_list = []

        for image_path in list(image_folder.glob("*.jpg"))[:self.max_study_images]:
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image_list.append(self.image_transform(image=image)['image'])

        with open(doc_file) as f:
            text = f.read()

        if self.text_tokenizer is not None:
            text = self.text_tokenizer(text)

        return text, image_list
