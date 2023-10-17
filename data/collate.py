import torch


class StudyCollator:
    def __init__(self,
                 tokenizer,
                 pad_to_multiple_of=8,
                 return_tensors="pt",
                 padding="longest",
                 ):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        self.padding = padding

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

        for text, image_list in batch:
            seq_sizes.append(len(image_list))

            for image in image_list:
                images.append(image)

            text_list.append(text)

        # Merge images (from list of 3D tensor to 4D tensor).
        images = torch.stack(images, 0)

        # Tokenizer texts
        texts = self.tokenizer(text_list,
                               padding=self.padding,
                               return_tensors=return_tensors,
                               pad_to_multiple_of=self.pad_to_multiple_of,
                               truncation=True,
                               )

        seq_attr = torch.zeros((len(seq_sizes), images.shape[0]))

        idx_x = 0
        for idx_y, size in enumerate(seq_sizes):
            seq_attr[idx_y, idx_x:idx_x + size] = 1
            idx_x += size

        return {
            **texts,
            "pixel_values": images,
            "seq_attr": seq_attr,
        }
