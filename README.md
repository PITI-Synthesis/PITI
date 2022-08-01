# PITI: Pretraining is All You Need for Image-to-Image Translation 
Official PyTorch implementation  
> Pretraining is All You Need for Image-to-Image Translation    
> 2022
    

[paper](https://arxiv.org/abs/2205.12952) | [project website](https://tengfei-wang.github.io/PITI/index.html) | [video]()

## Introduction
We present a simple and universal framework that brings the power of the pretraining to various
image-to-image translation tasks.  

Diverse samples synthesized by our approach.   
<img src="figure/diverse.jpg" height="380px"/>   

Comparison with other methods.   
<img src="figure/1.jpg" height="690px"/>

## Set up
### Installation
```
git clone https://github.com/PITI-Synthesis/PITI.git
cd PITI
```

### Environment
```
conda env create -f environment.yml
```

## Quick Start
### Pretrained Models
Please download our pre-trained models for both ```Base``` model and ```Upsample``` model, and put them in ```./ckpt```.
| Model | Task  | Dataset
| :--- | :----------  | :----------
|[Base-64x64](https://drive.google.com/file/d/1iqv0u0j5b4OH7t2fnnKkHSXfCQOY1kDX/view?usp=sharing)  | Mask-to-Image | Trained on COCO.
|[Upsample-64-256](https://drive.google.com/file/d/1HmYS-mXz-oFRKkmBWTX9hVn_YbB8vkJz/view?usp=sharing) | Mask-to-Image | Trained on COCO.
|[Base-64x64](https://drive.google.com/file/d/1QphMoGR9cojO_tc9khTXeLJMaJ4jQhI0/view?usp=sharing) | Sketch-to-Image | Trained on COCO.
|[Upsample-64-256](https://drive.google.com/file/d/1ND3LF5OEkjZpaEbSBBqGxG2iwRVE67Za/view?usp=sharing)| Sketch-to-Image  | Trained on COCO.

### Prepare Images
We put some example images in `./test_imgs`, and you can quickly try them.  
#### COCO
For COCO dataset, download the images and annotations from the [COCO webpage](https://cocodataset.org/#home).

For mask-to-image synthesis, we use the semantic maps in RGB format as inputs. To obtain such semantic maps, run ```./preprocess/preprocess_mask.py```.  Note that we do not need instant masks like previous works.

For sketch-to-image synthesis, we use sketch maps extracted by HED as inputs. To obtain such sketch maps, run ```./preprocess/preprocess_sketch.py```.


### Inference
#### Interactive Inference
Run the following script, and it would create an interactive GUI built by gradio. You can upload input masks or sketches and generate images.   
```
pip install gradio
python inference.py
```

#### Batch Inference
Modify `sample.sh` according to the follwing instructions, and run:   
```
bash sample.sh
```
| Args | Description
| :--- | :----------
| --model_path | the path of ckpt for base model.
| --sr_model_path | the path of ckpt for upsample model.
| --val_data_dir | the path of a txt file that contains the paths for images.
| --num_samples | number of images that you want to sample.
| --sample_c | Strength of classifier-free guidance.
| --mode | The input type.

## Training
### Preparation
1. Download and preprocess datasets.
2. Download pretrained GLIDE models from [the webpage](https://github.com/openai/glide-text2im).

### Start Training
Taking mask-to-image synthesis as an example: (sketch-to-image is the same)
#### Finetune the Base Model
Modify  `mask_finetune_base.sh`  and run:
```
bash mask_finetune_base.sh
```
#### Finetune the Upsample Model
Modify  `mask_finetune_upsample.sh`  and run:
```
bash mask_finetune_upsample.sh
```
## Citation
If you find this work useful for your research, please cite:

``` 
@inproceedings{wang2022pretraining,
 title = {Pretraining is All You Need for Image-to-Image Translation},
  author = {Wang, Tengfei and Zhang, Ting and Zhang, Bo and Ouyang, Hao and Chen, Dong and Chen, Qifeng and Wen, Fang},
  booktitle = {arXiv},
  year = {2022},
}
```
