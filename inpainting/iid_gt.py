import torch
from flux_fill_pipeline import FluxFillPipeline
from diffusers.utils import load_image
import glob
import os
from tqdm import tqdm

pipe = FluxFillPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Fill-dev",
    torch_dtype=torch.bfloat16
)
pipe.remove_text_models()
pipe = pipe.to("cuda")
seed = 100000000

image_root = 'images'
image_paths = glob.glob(os.path.join(image_root, '*'))
image_names = [os.path.basename(image_path).split('.')[0] for image_path in image_paths]
for image_name in image_names:
    output_dir = f'images_iid/{image_name}'
    os.makedirs(output_dir, exist_ok=True)

    image = load_image(os.path.join(image_root, f'{image_name}.jpg'))
    mask = load_image('mask.png')

    for repeat_id in tqdm(range(1000)):
        image_list, trajectories = pipe(
            image=image,
            mask_image=mask,
            height=512,
            width=512,
            num_inference_steps=28,
            generator=torch.Generator("cpu").manual_seed(seed+repeat_id),
            num_images_per_prompt=1,
            diversity_type='none',
            diversity_f_0=0.0,
        )
        image_list[0].save(os.path.join(output_dir, f'{repeat_id}.png'))
        samples = trajectories['samples']
        torch.save(samples, os.path.join(output_dir, f'samples_{repeat_id}.pth'))
