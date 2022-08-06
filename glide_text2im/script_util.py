import argparse
import inspect

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .text2im_model import (
    SuperResText2ImModel,
    Text2ImModel,
)
 


def model_and_diffusion_defaults(super_res=0):
    """
    Defaults for image training.
    """
    result= dict(
        image_size=64,
        num_channels=192,
        num_res_blocks=3,
        channel_mult="",
        num_heads=1,
        num_head_channels=64,
        num_heads_upsample=-1,
        attention_resolutions="32,16,8",
        dropout=0.1,
        text_ctx=128,
        xf_width=512,
        xf_layers=16,
        xf_heads=8,
        xf_final_ln=True,
        xf_padding=True,
        learn_sigma=True, ##
        sigma_small=False, ##
        diffusion_steps=1000,
        noise_schedule="squaredcos_cap_v2",
        timestep_respacing="",
        use_kl=False, ##
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_fp16=False, ##
        use_scale_shift_norm=True,
        resblock_updown=True,
        cache_text_emb=False,
        inpaint=False,
        super_res=0,
        mode = '',
    )
    if super_res:
        result.update(
        dict(
            image_size=256,
            num_res_blocks=2,
            noise_schedule="linear",
            super_res=super_res,
        ))
    return result


def create_model_and_diffusion(
        image_size=64,
        num_channels=192,
        num_res_blocks=3,
        channel_mult="",
        num_heads=1,
        num_head_channels=64,
        num_heads_upsample=-1,
        attention_resolutions="32,16,8",
        dropout=0.1,
        text_ctx=128,
        xf_width=512,
        xf_layers=16,
        xf_heads=8,
        xf_final_ln=True,
        xf_padding=True,
        learn_sigma=False, ##
        sigma_small=False, ##
        diffusion_steps=1000,
        noise_schedule="squaredcos_cap_v2",
        timestep_respacing="",
        use_kl=False, ##
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_fp16=False, ##
        use_scale_shift_norm=True,
        resblock_updown=True,
        cache_text_emb=False,
        inpaint=False,
        super_res=False,
        mode = '',
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        channel_mult=channel_mult,
        use_fp16=use_fp16,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        text_ctx=text_ctx,
        xf_width=xf_width,
        xf_layers=xf_layers,
        xf_heads=xf_heads,
        xf_final_ln=xf_final_ln,
        xf_padding=xf_padding,
        resblock_updown=resblock_updown,
        cache_text_emb=cache_text_emb,
        inpaint=inpaint,
        super_res=super_res,
        mode = mode
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def create_model(
    image_size,
        num_channels,
        num_res_blocks,
        learn_sigma,
        channel_mult,
        use_fp16,
        attention_resolutions,
        num_heads,
        num_head_channels,
        num_heads_upsample,
        use_scale_shift_norm,
        dropout,
        text_ctx,
        xf_width,
        xf_layers,
        xf_heads,
        xf_final_ln,
        xf_padding,
        resblock_updown,
        cache_text_emb,
        inpaint,
        super_res,
        mode,
):
    if channel_mult == "":
        if image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))
        assert 2 ** (len(channel_mult) + 2) == image_size

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    if super_res:
        model_cls = SuperResText2ImModel
    else:
        model_cls = Text2ImModel

    n_class = 3
    if mode == 'ade20k' or mode == 'coco':
        n_class = 3
    elif mode == 'depth-normal' :
        n_class = 6
    elif mode == 'coco-edge' or mode == 'flickr-edge':
        n_class = 1 

    return model_cls(
        text_ctx=text_ctx,
        xf_width=xf_width,
        xf_layers=xf_layers,
        xf_heads=xf_heads,
        xf_final_ln=xf_final_ln,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        num_head_channels=num_head_channels,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        in_channels=3,
        n_class = n_class,
        image_size = image_size,
    )



def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys=None):
    if keys is None:
        keys=vars(args)
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
