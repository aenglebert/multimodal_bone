import torch

from transformers import DataCollatorWithPadding, DataCollatorForLanguageModeling


class StudyCollator:
    def __init__(self,
                 tokenizer,
                 pad_to_multiple_of=8,
                 return_tensors="pt",
                 padding="longest",
                 mlm=False,
                 mlm_probability=0.15,
                 ):
        """
        Args:
            tokenizer: tokenizer to use
            pad_to_multiple_of: pad to multiple of
            return_tensors: return tensors type (default: "pt")
            padding: padding type (default: "longest")
            mlm: whether to use masked language modeling (default: False)
            mlm_probability: masked language modeling probability (default: 0.15)
        """
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        self.padding = padding

        if mlm:
            self.mlm = True
            self.mlm_probability = mlm_probability
            self.text_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm_probability=mlm_probability,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
            )
        else:
            self.mlm = False
            self.text_collator = DataCollatorWithPadding(
                tokenizer=tokenizer,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
                padding=padding,
            )

    def __call__(self, batch, return_tensors=None):
        """Creates mini-batch tensors from the list of tuples (text, list of images).

        Since the image list is variable lenght we need a custom collate_fn

        Args:
            data: list of tuple (image_list, label).
                - text: text to tokenize
                - imageList: list of image tensors.
        Returns:
            text: tensor of stacked tokens with padding to the max size
            images: torch tensor of shape (total_images_nb, 3, w, h).
            seq_attr: tensor of size (num_texts, num_images) that map the images to the corresponding text
        """
        if return_tensors is None:
            return_tensors = self.return_tensors

        text_list = []
        seq_sizes = []
        images_list_of_list = []

        for item in batch:
            text, image_list = item

            images_list_of_list.append(image_list)
            seq_sizes.append(len(image_list))

            text_list.append(text)

        # stack images
        images = torch.stack([image for image_list in images_list_of_list for image in image_list], 0)

        # Create a placeholder for the pooling mapping tensor
        pooling_matrix = torch.zeros((len(seq_sizes), images.shape[0]))

        # Fill the placeholder tensor with 1 where an image is present
        idx_x = 0
        for idx_y, size in enumerate(seq_sizes):
            pooling_matrix[idx_y, idx_x:idx_x + size] = 1
            idx_x += size

        texts = self.text_collator([self.tokenizer(text, truncation=True) for text in text_list])

        seq_attr = torch.zeros((len(seq_sizes), images.shape[0]))

        idx_x = 0
        for idx_y, size in enumerate(seq_sizes):
            seq_attr[idx_y, idx_x:idx_x + size] = 1
            idx_x += size

        return {
            **texts,
            "pixel_values": images,
            "pooling_matrix": pooling_matrix if pooling_matrix is not None else None,
            "image_text_pairs": torch.eye(len(seq_sizes), len(seq_sizes)),
            "return_loss": True,
        }
