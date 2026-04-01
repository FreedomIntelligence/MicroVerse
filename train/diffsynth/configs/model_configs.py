wan_series = [
    {
        "model_hash": "5ec04e02b42d2580483ad69f4e76346a",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': False, 'patch_size': [1, 2, 2], 'in_dim': 16, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06},
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_dit.WanVideoDiTStateDictConverter",
    },
    {
        "model_hash": "9c8818c2cbea55eca56c7b447df170da",
        "model_name": "wan_video_text_encoder",
        "model_class": "diffsynth.models.wan_video_text_encoder.WanTextEncoder",
    },
    {
        "model_hash": "ccc42284ea13e1ad04693284c7a09be6",
        "model_name": "wan_video_vae",
        "model_class": "diffsynth.models.wan_video_vae.WanVideoVAE",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_vae.WanVideoVAEStateDictConverter",
    },
    {
        "model_hash": "5941c53e207d62f20f9025686193c40b",
        "model_name": "wan_video_image_encoder",
        "model_class": "diffsynth.models.wan_video_image_encoder.WanImageEncoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_image_encoder.WanImageEncoderStateDictConverter"
    },
    {
        "model_hash": "9269f8db9040a9d860eaca435be61814",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': False, 'patch_size': [1, 2, 2], 'in_dim': 16, 'dim': 1536, 'ffn_dim': 8960, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 12, 'num_layers': 30, 'eps': 1e-06}
    },
    {
        "model_hash": "aafcfd9672c3a2456dc46e1cb6e52c70",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': False, 'patch_size': [1, 2, 2], 'in_dim': 16, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06}
    },
    {
        "model_hash": "3ef3b1f8e1dab83d5b71fd7b617f859f",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': True, 'patch_size': [1, 2, 2], 'in_dim': 36, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06, 'has_image_pos_emb': True}
    },
    {
        "model_hash": "e1de6c02cdac79f8b739f4d3698cd216",
        "model_name": "wan_video_vae",
        "model_class": "diffsynth.models.wan_video_vae.WanVideoVAE38",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_vae.WanVideoVAEStateDictConverter",
    },
    {
        "model_hash": "a61453409b67cd3246cf0c3bebad47ba",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': False, 'patch_size': [1, 2, 2], 'in_dim': 16, 'dim': 1536, 'ffn_dim': 8960, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 12, 'num_layers': 30, 'eps': 1e-06},
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_dit.WanVideoDiTStateDictConverter",
    },
    {
        "model_hash": "a61453409b67cd3246cf0c3bebad47ba",
        "model_name": "wan_video_vace",
        "model_class": "diffsynth.models.wan_video_vace.VaceWanModel",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_vace.VaceWanModelDictConverter"
    },
    {
        "model_hash": "7a513e1f257a861512b1afd387a8ecd9",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': False, 'patch_size': [1, 2, 2], 'in_dim': 16, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06},
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_dit.WanVideoDiTStateDictConverter",
    },
    {
        "model_hash": "7a513e1f257a861512b1afd387a8ecd9",
        "model_name": "wan_video_vace",
        "model_class": "diffsynth.models.wan_video_vace.VaceWanModel",
        "extra_kwargs": {'vace_layers': (0, 5, 10, 15, 20, 25, 30, 35), 'vace_in_dim': 96, 'patch_size': (1, 2, 2), 'has_image_input': False, 'dim': 5120, 'num_heads': 40, 'ffn_dim': 13824, 'eps': 1e-06},
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_vace.VaceWanModelDictConverter"
    },
]

MODEL_CONFIGS = wan_series
