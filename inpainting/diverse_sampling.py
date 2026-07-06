import argparse
import torch
from flux_fill_pipeline import FluxFillPipeline
from diffusers.utils import load_image
import os

parser = argparse.ArgumentParser(description="FLUX.1-Fill-dev diverse joint inpainting")
parser.add_argument("--s_proj", choices=["none", "soft", "hard"], required=True,
                    help="score-regularization mode (SR): none, soft, or hard")
args = parser.parse_args()

pipe = FluxFillPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Fill-dev", 
    torch_dtype=torch.bfloat16
)
pipe.remove_text_models()
pipe.to("cuda")

seed = 0
N = 10
f_0 = 0.3

image_root = 'images'
for repeat_time in range(100):
    for image_idx in range(10):
        s_proj_options = [args.s_proj]

        for s_proj in s_proj_options:
            output_dir = f'images_pred_{f_0}/{s_proj}/{image_idx}'
            os.makedirs(f'{output_dir}/repeat_{repeat_time}', exist_ok=True)
            image = load_image(os.path.join(image_root, f'{image_idx}.jpg'))
            mask = load_image('mask.png')

            if os.path.exists(f'{output_dir}/trajectory_{repeat_time}.pth'):
                try:
                    torch.load(f'{output_dir}/trajectory_{repeat_time}.pth')
                    has_trajectory = True
                except:
                    has_trajectory = False
            else:
                has_trajectory = False
            if has_trajectory:
                print(f"{output_dir}/trajectory_{repeat_time}.pth already exists")
                continue
            else:
                print(f'now, computing: {output_dir}/trajectory_{repeat_time}.pth')

            image_list, trajectories = pipe(
                image=image,
                mask_image=mask,
                height=512,
                width=512,
                num_inference_steps=28,
                generator=torch.Generator("cpu").manual_seed(seed+repeat_time),
                num_images_per_prompt=N,
                diversity_type="dpp",
                diversity_f_0=f_0,
                diversity_s_proj=s_proj,
            )
            for i, image in enumerate(image_list):
                image.save(os.path.join(output_dir, f'repeat_{repeat_time}/{i}.png'))

            torch.save(trajectories, f'{output_dir}/trajectory_{repeat_time}.pth')
