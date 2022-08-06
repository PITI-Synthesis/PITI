"""
Train a diffusion model on images.
"""
import argparse
import torch.distributed as dist
from pretrained_diffusion import dist_util, logger
from pretrained_diffusion.image_datasets_mask import load_data_mask
from pretrained_diffusion.image_datasets_sketch import load_data_sketch
from pretrained_diffusion.image_datasets_depth import load_data_depth
from pretrained_diffusion.resample import create_named_schedule_sampler
from pretrained_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,)
from pretrained_diffusion.train_util import TrainLoop
import torch

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()

    options=args_to_dict(args, model_and_diffusion_defaults(args.super_res).keys())
    model, diffusion = create_model_and_diffusion(**options)

    options=args_to_dict(args)
    if dist.get_rank() == 0:
        logger.save_args(options)

##### scratch #####
    if  args.model_path:
        print('loading decoder')
        model_ckpt = dist_util.load_state_dict(args.model_path, map_location="cpu")

        for k  in list(model_ckpt.keys()):
            if k.startswith("transformer") and 'transformer_proj'  not in k:
                # print(f"Removing key {k} from pretrained checkpoint")
                del model_ckpt[k]
            if k.startswith("padding_embedding") or k.startswith("positional_embedding") or k.startswith("token_embedding") or k.startswith("final_ln"):
                # print(f"Removing key {k} from pretrained checkpoint")
                del model_ckpt[k]

        model.decoder.load_state_dict(
            model_ckpt   , strict=True )


    if args.encoder_path:
        print('loading encoder')
        encoder_ckpt = dist_util.load_state_dict(args.encoder_path, map_location="cpu")
        model.encoder.load_state_dict(
            encoder_ckpt   , strict=True )        

    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)


########### dataset selection 
    logger.log("creating data loader...")
 
    if args.mode == 'ade20k' or args.mode == 'coco':
        data = load_data_mask(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            train=True,
            low_res=args.super_res,
            uncond_p = args.uncond_p,
            mode = args.mode,
            random_crop=True,
        )

        val_data = load_data_mask(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size//2,
            image_size=args.image_size,
            train=False,
            deterministic=True,
            low_res=args.super_res,
            uncond_p = 0. ,
            mode = args.mode,
            random_crop=False,
        )

    elif args.mode == 'depth' or args.mode == 'depth-normal':
        data = load_data_depth(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            train=True,
            low_res=args.super_res,
            uncond_p = args.uncond_p,
            mode = args.mode,
            random_crop=True,
        )

        val_data = load_data_depth(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size//2,
            image_size=args.image_size,
            train=False,
            deterministic=True,
            low_res=args.super_res,
            uncond_p = 0. ,
            mode = args.mode,
            random_crop=False,
        )    


    elif args.mode == 'coco-edge' or args.mode == 'flickr-edge':
        data = load_data_sketch(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            train=True,
            low_res=args.super_res,
            uncond_p = args.uncond_p,
            mode = args.mode,
            random_crop=True,
        )

        val_data = load_data_sketch(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size//2,
            image_size=args.image_size,
            train=False,
            deterministic=True,
            low_res=args.super_res,
            uncond_p = 0. ,
            mode = args.mode,
            random_crop=False,
        )    

    logger.log("training...")
    TrainLoop(
        model,
        options,
        diffusion,
        data=data,
        val_data=val_data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        finetune_decoder = args.finetune_decoder,
        mode =  args.mode,
        use_vgg = args.super_res,
        use_gan = args.super_res,
        uncond_p = args.uncond_p,
        super_res = args.super_res,
    ).run_loop()

 
def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        model_path="",
        encoder_path="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=200,
        save_interval=20000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        super_res=0,
        sample_c=1.,
        sample_respacing="100",
        uncond_p=0.2,
        num_samples=1,
        finetune_decoder = False,
        mode =  "",
        )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
