import torch as th
import torch.nn as nn
import torch.nn.functional as F
import random
from .nn import timestep_embedding
from .unet import UNetModel
from .xf import LayerNorm, Transformer, convert_module_to_f16
from timm.models.vision_transformer import PatchEmbed

class Text2ImModel(nn.Module):
    def __init__(
        self,
        text_ctx,
        xf_width,
        xf_layers,
        xf_heads,
        xf_final_ln,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout,
        channel_mult,
        use_fp16,
        num_heads,
        num_heads_upsample,
        num_head_channels,
        use_scale_shift_norm,
        resblock_updown, 
        in_channels = 3,  
        n_class = 3,
        image_size = 64,
    ):
        super().__init__()
        self.encoder = Encoder(img_size=image_size, patch_size=image_size//16, in_chans=n_class,
                 xf_width=xf_width, xf_layers=8, xf_heads=xf_heads, model_channels=model_channels)

        self.in_channels = in_channels
        self.decoder = Text2ImUNet(
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=dropout,
        channel_mult=channel_mult,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        num_head_channels=num_head_channels,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        encoder_channels=xf_width
    )


    def forward(self, xt, timesteps, ref=None, uncond_p=0.0):
        latent_outputs =self.encoder(ref, uncond_p)
        pred = self.decoder(xt, timesteps, latent_outputs)
        return pred


class Text2ImUNet(UNetModel):
    def __init__(
        self,
        *args,
        **kwargs,  
    ):
        super().__init__(*args, **kwargs)
        self.transformer_proj = nn.Linear(512, self.model_channels * 4) ###
  
    def forward(self, x, timesteps, latent_outputs):
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        xf_proj, xf_out = latent_outputs["xf_proj"], latent_outputs["xf_out"]

        xf_proj = self.transformer_proj(xf_proj) ###
        emb = emb + xf_proj.to(emb)
 
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, xf_out)
            hs.append(h)
        h = self.middle_block(h, emb, xf_out)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, xf_out)
        h = h.type(x.dtype)
        h = self.out(h)
        return h


class Encoder(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        xf_width,
        xf_layers,
        xf_heads, 
        model_channels,
    ): 
        super().__init__( )
        self.transformer = Transformer(
            xf_width,
            xf_layers,
            xf_heads,
        )

        self.cnn = CNN(in_chans)
        self.final_ln = LayerNorm(xf_width)
  
        self.cls_token = nn.Parameter(th.empty(1, 1, xf_width, dtype=th.float32))
        self.positional_embedding = nn.Parameter(th.empty(1, 256 + 1, xf_width, dtype=th.float32))

    def forward(self, ref, uncond_p=0.0):
        x = self.cnn(ref)
        x = x.flatten(2).transpose(1, 2)
        
        x = x + self.positional_embedding[:, 1:, :]

        cls_token = self.cls_token + self.positional_embedding[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = th.cat((x, cls_tokens), dim=1)

        xf_out = self.transformer(x)
        if self.final_ln is not None:
            xf_out = self.final_ln(xf_out)
    
        xf_proj = xf_out[:, -1]
        xf_out = xf_out[:, :-1].permute(0, 2, 1)  # NLC -> NCL

        outputs = dict(xf_proj=xf_proj, xf_out=xf_out)
        return outputs


class SuperResText2ImModel(Text2ImModel):
    """
    A text2im model that performs super-resolution.
    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, *args, **kwargs):
        if "in_channels" in kwargs:
            kwargs = dict(kwargs)
            kwargs["in_channels"] = kwargs["in_channels"] * 2
        else:
            # Curse you, Python. Or really, just curse positional arguments :|.
            args = list(args)
            args[1] = args[1] * 2
        super().__init__(*args, **kwargs)
        

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(
            low_res, (new_height, new_width), mode="bilinear", align_corners=False
        )

        # ##########
        # upsampled = upsampled + th.randn_like(upsampled)*0.0005*th.log(1 + 0.1* timesteps.reshape(timesteps.shape[0], 1,1,1))  
        # ##########

        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)



def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=True)


def conv7x7(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=7, 
                     stride=stride, padding=3, bias=True)                     

class CNN(nn.Module):
    def __init__(self, in_channels=3):
        super(CNN, self).__init__()
        self.conv1 = conv7x7(in_channels, 32) #256
        self.norm1 = nn.InstanceNorm2d(32, affine=True)
        self.LReLU1 = nn.LeakyReLU(0.2)

        self.conv2 = conv3x3(32, 64, 2)  #128
        self.norm2 = nn.InstanceNorm2d(64, affine=True)
        self.LReLU2 = nn.LeakyReLU(0.2)
 
        self.conv3 = conv3x3(64, 128, 2)  #64
        self.norm3 = nn.InstanceNorm2d(128, affine=True)
        self.LReLU3 = nn.LeakyReLU(0.2)
        
        self.conv4 = conv3x3(128, 256, 2)  #32
        self.norm4 = nn.InstanceNorm2d(256, affine=True)
        self.LReLU4 = nn.LeakyReLU(0.2)
  
        self.conv5 = conv3x3(256, 512, 2)  #16
        self.norm5 = nn.InstanceNorm2d(512, affine=True)
        self.LReLU5 = nn.LeakyReLU(0.2)
    
        self.conv6 = conv3x3(512, 512, 1)
 
        
    def forward(self, x):
        x = self.LReLU1(self.norm1(self.conv1(x)))
        x = self.LReLU2(self.norm2(self.conv2(x)))
        x = self.LReLU3(self.norm3(self.conv3(x)))
        x = self.LReLU4(self.norm4(self.conv4(x)))
        x = self.LReLU5(self.norm5(self.conv5(x)))
        x = self.conv6(x) 
        return x     