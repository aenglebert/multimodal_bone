from torch import nn

from transformers import CamembertModel, CamembertForMaskedLM, CamembertConfig


class ProjCamembertConfig(CamembertConfig):
    def __init__(self,
                 projection_size: int = 512,
                 **kwargs):

        model_type = "proj_camembert"

        super().__init__(**kwargs)
        self.projection_size = projection_size


class ProjCamembertModel(CamembertModel):
    config: ProjCamembertConfig

    def __init__(self, config: ProjCamembertConfig):
        super().__init__(config)

        self.projection = nn.Linear(config.hidden_size, config.projection_size)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        base_model_prefix: str = "",
        **kwargs,
    ):
        if base_model_prefix != "":
            cls.base_model_prefix = base_model_prefix

        return super().from_pretrained(pretrained_model_name_or_path,
                                       **kwargs)


class ProjCamembertForMaskedLM(CamembertForMaskedLM):
    config: ProjCamembertConfig

    def __init__(self, config: ProjCamembertConfig):
        super().__init__(config)

        self.projection = nn.Linear(config.hidden_size, config.projection_size)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        base_model_prefix: str = "",
        **kwargs,
    ):
        if base_model_prefix != "":
            cls.base_model_prefix = base_model_prefix

        return super().from_pretrained(pretrained_model_name_or_path,
                                       **kwargs)