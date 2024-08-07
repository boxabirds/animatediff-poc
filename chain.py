import torch
from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler, EulerAncestralDiscreteScheduler
import cv2
import numpy as np
import argparse
from PIL import Image

def generate_frames(prompt, negative_prompt="bad quality, worse quality", num_frames=16, num_inference_steps=20):
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        guidance_scale=7.5,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cuda").manual_seed(666),
    )
    return output.frames[0]

def export_to_mp4(frames, output_path, fps=10):
    # Convert the first frame to numpy array to get dimensions
    first_frame = np.array(frames[0])
    height, width, layers = first_frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        # Convert PIL Image to numpy array and change color space
        frame_array = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        video.write(frame_array)

    video.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate animated video using AnimateDiffPipeline")
    parser.add_argument("--output", type=str, default="animation.mp4", help="Output file name")
    args = parser.parse_args()

    #adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-3")
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
    model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
    #model_id = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
    pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16).to("cuda")
    # pipe.scheduler = DDIMScheduler.from_pretrained(
    #     model_id,
    #     subfolder="scheduler",
    #     beta_schedule="linear",
    #     clip_sample=False,
    #     timestep_spacing="linspace",
    #     steps_offset=1
    # )
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        beta_schedule="linear",
    )

    # enable memory savings
    pipe.enable_vae_slicing()
    #pipe.enable_vae_tiling()

    # enable FreeInit
    pipe.enable_free_init(method="butterworth", use_fast_sampling=False)

    # generate frames
    frames = generate_frames(
        prompt="cute ginger kitten on sunny day walking in a forest towards the camera",
        negative_prompt="bad quality, worse quality",
        num_frames=32,
        num_inference_steps=20
    )

    # disable FreeInit
    pipe.disable_free_init()

    # export to mp4
    export_to_mp4(frames, args.output)
