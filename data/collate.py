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
        images = []
        seq_sizes = []
        images_list_of_list = []
        doc_embedding_list = []

        for item in batch:
            if len(item) == 2:
                text, image_list = item
                doc_embedding = None
            elif len(item) == 3:
                text, image_list, doc_embedding = item
                if return_tensors == "pt":
                    doc_embedding = torch.tensor(doc_embedding)
            else:
                raise ValueError("Batch item must be a tuple of 2 or 3 elements")

            images_list_of_list.append(image_list)
            seq_sizes.append(len(image_list))

            if doc_embedding is not None:
                doc_embedding_list.append(doc_embedding)

            text_list.append(text)

        # Get max image sequence length
        max_seq_size = max(seq_sizes)

        # Get shape of an image
        image_shape = images_list_of_list[0][0].shape

        # Create a placeholder tensor for images
        images = torch.zeros((len(seq_sizes), max_seq_size, image_shape[0], image_shape[1], image_shape[2]))

        # Fill the placeholder tensor with images
        for idx_x, image_list in enumerate(images_list_of_list):
            for idx_y, image in enumerate(image_list):
                images[idx_x, idx_y] = image

        if len(doc_embedding_list) == len(text_list):
            doc_embeddings = torch.stack(doc_embedding_list, 0)
        else:
            doc_embeddings = None

        texts = self.text_collator([self.tokenizer(text, truncation=True) for text in text_list])

        seq_attr = torch.zeros((len(seq_sizes), images.shape[0]))

        idx_x = 0
        for idx_y, size in enumerate(seq_sizes):
            seq_attr[idx_y, idx_x:idx_x + size] = 1
            idx_x += size

        return {
            **texts,
            "pixel_values": images,
            "seq_attr": seq_attr,
            "doc_embeddings": doc_embeddings if doc_embeddings is not None else None,
        }
