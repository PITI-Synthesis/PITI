import os
from typing import Tuple
from . import dist_util
import PIL
import numpy as np
import torch as th
from .script_util import (
    create_gaussian_diffusion,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

# Sample from the base model.

#@th.inference_mode()
def sample(
    glide_model,
    glide_options,
    side_x,
    side_y,
    prompt,
    batch_size=1,
    guidance_scale=4,
    device="cpu",
    prediction_respacing="100",
    upsample_enabled=False,
    upsample_temp=0.997,
    mode = '',
):

    eval_diffusion = create_gaussian_diffusion(
        steps=glide_options["diffusion_steps"],
        learn_sigma=glide_options["learn_sigma"],
        noise_schedule=glide_options["noise_schedule"],
        predict_xstart=glide_options["predict_xstart"],
        rescale_timesteps=glide_options["rescale_timesteps"],
        rescale_learned_sigmas=glide_options["rescale_learned_sigmas"],
        timestep_respacing=prediction_respacing
    )
 
    # Create the classifier-free guidance tokens (empty)
    full_batch_size = batch_size * 2
    cond_ref   =  prompt['ref']
    uncond_ref = th.ones_like(cond_ref) 
    
    model_kwargs = {}
    model_kwargs['ref'] =  th.cat([cond_ref, uncond_ref], 0).to(dist_util.dev())

    def cfg_model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = glide_model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
 
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)

        eps = th.cat([half_eps, half_eps], dim=0)
        return th.cat([eps, rest], dim=1)


    if upsample_enabled:
        model_kwargs['low_res'] = prompt['low_res'].to(dist_util.dev())
        noise = th.randn((batch_size, 3, side_y, side_x), device=device) * upsample_temp
        model_fn = glide_model # just use the base model, no need for CFG.
        model_kwargs['ref'] =  model_kwargs['ref'][:batch_size]

        samples = eval_diffusion.p_sample_loop(
        model_fn,
        (batch_size, 3, side_y, side_x),  # only thing that's changed
        noise=noise,
        device=device,
        clip_denoised=True,
        progress=False,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:batch_size]

    else:
        model_fn = cfg_model_fn # so we use CFG for the base model.
        noise = th.randn((batch_size, 3, side_y, side_x), device=device) 
        noise = th.cat([noise, noise], 0)  
        samples = eval_diffusion.p_sample_loop(
            model_fn,
            (full_batch_size, 3, side_y, side_x),  # only thing that's changed
            noise=noise,
            device=device,
            clip_denoised=True,
            progress=False,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
    
    return samples

 