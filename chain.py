import torch
from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler
from PIL import Image
from torchvision import transforms

def image_to_latent(pipe, image):
    if isinstance(image, Image.Image):
        image = transforms.ToTensor()(image).unsqueeze(0)
    image = image.to(pipe.device)
    latent = pipe.vae.encode(image).latent_dist.sample()
    latent = latent * pipe.vae.config.scaling_factor
    return latent

def generate_chained_video(pipe, prompts, num_frames_per_segment=16, seed=42):
    all_frames = []
    last_latent = None
    generator = torch.Generator().manual_seed(seed)

    for prompt in prompts:
        if last_latent is not None:
            # Replicate the last latent across the time dimension
            latents = last_latent.unsqueeze(2).repeat(1, 1, num_frames_per_segment, 1, 1)
        else:
            latents = None

        output = pipe(
            prompt=prompt,
            latents=latents,
            num_frames=num_frames_per_segment,
            generator=generator,
        )
        
        all_frames.extend(output.frames[0])
        
        # Encode the last frame to use as the starting point for the next segment
        last_frame = output.frames[0][-1]
        last_latent = image_to_latent(pipe, last_frame)

    return all_frames

# Set up the pipeline
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter)
pipe.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe.enable_vae_slicing()

# Move the pipeline to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# Generate the chained video
prompts = [
    "A serene beach at sunrise, gentle waves",
    "The same beach at midday, busy with people",
    "The beach at sunset, golden light on the water",
    "The beach at night under a starry sky"
]

frames = generate_chained_video(pipe, prompts)

# Export to GIF
from diffusers.utils import export_to_gif
export_to_gif(frames, "chained_beach_day.gif")
