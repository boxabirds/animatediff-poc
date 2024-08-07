import torch
import argparse
from diffusers import MotionAdapter, AnimateDiffPipeline, EulerAncestralDiscreteScheduler
from diffusers.utils import export_to_video

# Add command line argument parsing
parser = argparse.ArgumentParser(description="Generate animated video using AnimateDiff")
parser.add_argument("--output", type=str, default="output.mp4", help="Output filename (default: output.mp4)")
args = parser.parse_args()

# Load the motion adapter
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-3")
# load SD 1.5 based finetuned model
#model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
model_id = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter)
scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    beta_schedule="linear",
)
pipe.scheduler = scheduler

# enable memory savings
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

output = pipe(
    prompt=(
        "cute ginger kitten on sunny day walking in a forest towards the camera"
    ),
    negative_prompt="bad quality, worse quality",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=25,
    generator=torch.Generator("cpu").manual_seed(42),
)

frames = output.frames[0]
export_to_video(frames, args.output)
