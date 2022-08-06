"""
Train a diffusion model on images.
"""
import argparse

from pretrained_diffusion import dist_util, logger
from pretrained_diffusion.image_datasets_mask import load_data_mask
from pretrained_diffusion.image_datasets_sketch import load_data_sketch
from pretrained_diffusion.image_datasets_depth import load_data_depth
from pretrained_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from pretrained_diffusion.train_util import TrainLoop
from pretrained_diffusion.glide_util import sample 
import torch
import os
import torch as th
import torchvision.utils as tvu
import torch.distributed as dist

def main():
    parser, parser_up = create_argparser()
    
    args = parser.parse_args()
    args_up = parser_up.parse_args()
    dist_util.setup_dist()

    options=args_to_dict(args, model_and_diffusion_defaults(0.).keys())
    model, diffusion = create_model_and_diffusion(**options)
 
    options_up=args_to_dict(args_up, model_and_diffusion_defaults(True).keys())
    model_up, diffusion_up = create_model_and_diffusion(**options_up)
 

    if  args.model_path:
        print('loading model')
        model_ckpt = dist_util.load_state_dict(args.model_path, map_location="cpu")

        model.load_state_dict(
            model_ckpt   , strict=True )

    if  args.sr_model_path:
        print('loading sr model')
        model_ckpt2 = dist_util.load_state_dict(args.sr_model_path, map_location="cpu")

        model_up.load_state_dict(
            model_ckpt2   , strict=True ) 

 
    model.to(dist_util.dev())
    model_up.to(dist_util.dev())
    model.eval()
    model_up.eval()
 
########### dataset
    logger.log("creating data loader...")

    if args.mode == 'ade20k' or args.mode == 'coco':
 
        val_data = load_data_mask(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size//2,
            image_size=256,
            train=False,
            deterministic=True,
            low_res=args.super_res,
            uncond_p = 0. ,
            mode = args.mode,
            random_crop=False,
        )

    elif args.mode == 'depth' or args.mode == 'depth-normal':
        val_data = load_data_depth(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size//2,
            image_size=256,
            train=False,
            deterministic=True,
            low_res=args.super_res,
            uncond_p = 0. ,
            mode = args.mode,
            random_crop=False,
        )    


    elif args.mode == 'coco-edge' or args.mode == 'flickr-edge':
        val_data = load_data_sketch(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size//2,
            image_size=256,
            train=False,
            deterministic=True,
            low_res=args.super_res,
            uncond_p = 0. ,
            mode = args.mode,
            random_crop=False,
        )    


 
    logger.log("sampling...")
    gt_path = os.path.join(logger.get_dir(), 'GT')
    os.makedirs(gt_path,exist_ok=True)
    lr_path = os.path.join(logger.get_dir(), 'LR')
    os.makedirs(lr_path,exist_ok=True)    
    hr_path = os.path.join(logger.get_dir(), 'HR')
    os.makedirs(hr_path,exist_ok=True)
    ref_path = os.path.join(logger.get_dir(), 'REF')
    os.makedirs(ref_path,exist_ok=True)    

    img_id = 0
    while (True):
        if img_id >= args.num_samples:
            break

        batch, model_kwargs = next(val_data)    
          
        with th.no_grad():
            samples_lr =sample(
                glide_model= model,
                glide_options= options,
                side_x= 64,
                side_y= 64,
                prompt=model_kwargs,
                batch_size= args.batch_size//2,
                guidance_scale=args.sample_c,
                device=dist_util.dev(),
                prediction_respacing= "250",
                upsample_enabled= False,
                upsample_temp=0.997,
                mode = args.mode,
            )

            samples_lr = samples_lr.clamp(-1, 1)

            tmp = (127.5*(samples_lr + 1.0)).int() 
            model_kwargs['low_res'] = tmp/127.5 - 1.

            samples_hr =sample(
                glide_model= model_up,
                glide_options= options_up,
                side_x=256,
                side_y=256,
                prompt=model_kwargs,
                batch_size=args.batch_size//2,
                guidance_scale=1,
                device=dist_util.dev(),
                prediction_respacing= "fast27",
                upsample_enabled=True,
                upsample_temp=0.997,
                mode = args.mode,
            )


            samples_lr = samples_lr.cpu()
            # ref = model_kwargs['ref'].cpu()
            ref =  model_kwargs['ref_ori'].cpu()
       
            samples_hr = samples_hr.cpu()
            for i in range(samples_lr.size(0)):
                name = model_kwargs['path'][i].split('/')[-1].split('.')[0] + '.png'
                out_path = os.path.join(lr_path, name)
                tvu.save_image(
                    (samples_lr[i]+1)*0.5, out_path)

                out_path = os.path.join(gt_path, name)
                tvu.save_image(
                    (batch[i]+1)*0.5, out_path)

                out_path = os.path.join(ref_path, name)
                tvu.save_image(
                    (ref[i]+1)*0.5, out_path)

      
                out_path = os.path.join(hr_path, name)
                tvu.save_image(
                    (samples_hr[i]+1)*0.5, out_path)

                img_id += 1     


def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        model_path="",
        sr_model_path="",
        encoder_path="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=20000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        sample_c=1.,
        sample_respacing="100",
        uncond_p=0.2,
        num_samples=1,
        finetune_decoder = False,
        mode = '',
        )

    defaults_up = defaults
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    defaults_up.update(model_and_diffusion_defaults(True))
    parser_up = argparse.ArgumentParser()
    add_dict_to_argparser(parser_up, defaults_up)

    return parser, parser_up


if __name__ == "__main__":
    main()
 
 