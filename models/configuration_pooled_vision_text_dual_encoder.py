from transformers.configuration_utils import PretrainedConfig


class PooledVisionTextDualEncoderConfig(PretrainedConfig):
    def __init__(self,
                 projection_dim=512,
                 logit_scale_init_value=2.6592,
                 loss_type="siglip",
                 pool_image=True,
                 **kwargs):
        super().__init__(**kwargs)

        if "vision_config" not in kwargs:
            raise ValueError("`vision_config` can not be `None`.")

        if "text_config" not in kwargs:
            raise ValueError("`text_config` can not be `None`.")

        vision_config = kwargs.pop("vision_config")
        text_config = kwargs.pop("text_config")

        self.vision_config = vision_config

        self.text_config = text_config

        self.is_composition = True

        self.logit_scale_init_value = logit_scale_init_value

        self.loss_type = loss_type

        self.pool_image = pool_image

    @classmethod
    def from_vision_text_configs(cls, vision_config: PretrainedConfig, text_config: PretrainedConfig, **kwargs):
        r"""
        Instantiate a [`VisionTextDualEncoderConfig`] (or a derived class) from text model configuration and vision
        model configuration.

        Returns:
            [`VisionTextDualEncoderConfig`]: An instance of a configuration object
        """

        return cls(vision_config=vision_config, text_config=text_config, **kwargs)