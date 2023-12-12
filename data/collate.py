import torch

from transformers import DataCollatorWithPadding, DataCollatorForLanguageModeling


class StudyCollator:
    def __init__(self,
                 tokenizer,
                 pad_to_multiple_of=8,
                 return_tensors="pt",
                 padding="longest",
                 images_sequence="matrix",
                 mlm=False,
                 mlm_probability=0.15,
                 ):
        """
        Args:
            tokenizer: tokenizer to use
            pad_to_multiple_of: pad to multiple of
            return_tensors: return tensors type (default: "pt")
            padding: padding type (default: "longest")
            images_sequence: images sequence type (default: "matrix"). Can be either "matrix" or "5d",
                where "matrix" will return a tensor of shape (n_images_total, channels, height, width) accompanied by a
                tensor of shape (batch_size, n_images_total) that maps the images to the corresponding text.
                "5d" will return a tensor of shape (batch_size, n_images_per_exam, channels, height, width)
                with an additional tensor of shape (batch_size, n_images_per_exam) that is a mask of the valid images
                in each exam (1 for valid, 0 for padding).
            mlm: whether to use masked language modeling (default: False)
            mlm_probability: masked language modeling probability (default: 0.15)
        """
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        self.padding = padding

        assert images_sequence in ["matrix", "5d"], "images_sequence must be either 'matrix' or '5d'"

        self.images_sequence = images_sequence

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

        if self.images_sequence == "5d":
            # Get max image sequence length
            max_seq_size = max(seq_sizes)

            # Get shape of an image
            image_shape = images_list_of_list[0][0].shape

            # Create a placeholder tensor for images
            images = torch.zeros((len(seq_sizes), max_seq_size, image_shape[0], image_shape[1], image_shape[2]))

            # Create a placeholder for images attention mask
            images_attention_mask = torch.zeros((len(seq_sizes), max_seq_size))

            # Fill the placeholder tensor with images
            for idx_x, image_list in enumerate(images_list_of_list):
                for idx_y, image in enumerate(image_list):
                    images[idx_x, idx_y] = image
                    images_attention_mask[idx_x, idx_y] = 1
        else:
            images_attention_mask = None

        if self.images_sequence == "matrix":
            # stack images
            images = torch.stack([image for image_list in images_list_of_list for image in image_list], 0)

            # Create a placeholder for the pooling mapping tensor
            pooling_matrix = torch.zeros((len(seq_sizes), images.shape[0]))

            # Fill the placeholder tensor with 1 where an image is present
            idx_x = 0
            for idx_y, size in enumerate(seq_sizes):
                pooling_matrix[idx_y, idx_x:idx_x + size] = 1
                idx_x += size
        else:
            pooling_matrix = None

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
            "pooling_matrix": pooling_matrix if pooling_matrix is not None else None,
            "images_attention_mask": images_attention_mask if images_attention_mask is not None else None,
            "image_text_pairs": torch.eye(len(seq_sizes), len(seq_sizes)),
            "doc_embeddings": doc_embeddings if doc_embeddings is not None else None,
        }
