import os
import PIL
import numpy as np
import torch
from diffusers import DDIMScheduler
from tqdm import tqdm
import ptp_utils
from StableDiffusionPipelineWithDDIMInversion import StableDiffusionPipelineWithDDIMInversion
from attention_store import AttentionStore
from PerMIRS.visualization_utils import bbox_from_mask

def center_crop(im, min_obj_x=None, max_obj_x=None, offsets=None):
    if offsets is None:
        width, height = im.size  # Get dimensions
        min_dim = min(width, height)
        left = (width - min_dim) / 2
        top = (height - min_dim) / 2
        right = (width + min_dim) / 2
        bottom = (height + min_dim) / 2

        if min_obj_x < left:
            diff = abs(left - min_obj_x)
            left = min_obj_x
            right = right - diff
        if max_obj_x > right:
            diff = abs(right - max_obj_x)
            right = max_obj_x
            left = left + diff
    else:
        left, top, right, bottom = offsets

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im, (left, top, right, bottom)


def load_im_into_format_from_path(im_path, size=512, offsets=None):
    im, offsets = center_crop(PIL.Image.open(im_path), offsets=offsets)
    return im


def load_im_into_format_from_image(image, size=512, min_obj_x=None, max_obj_x=None):
    im, offsets = center_crop(image, min_obj_x=min_obj_x, max_obj_x=max_obj_x)
    return im.resize((size, size)), offsets

def extract_attention(sd_model, image, prompt):
    controller = AttentionStore()
    inv_latents = sd_model.invert(prompt, image=image, guidance_scale=1.0).latents
    ptp_utils.register_attention_control_efficient(sd_model, controller)
    # recon_image = sd_model(prompt, latents=inv_latents, guidance_scale=1.0).images[0]
    key_format = f"up_blocks_3_attentions_1_transformer_blocks_0_attn1_self"
    timestamp = 49
    return [controller.attention_store[timestamp]["Q_" + key_format][0].to("cuda"),
               controller.attention_store[timestamp]["K_" + key_format][0].to("cuda"),
           ], sd_model.unet.up_blocks[3].attentions[1].transformer_blocks[0].attn1

if __name__ == "__main__":
    model_id = "CompVis/stable-diffusion-v1-4"

    device = "cuda"  # if torch.cuda.is_available() else "cpu"

    sd_model = StableDiffusionPipelineWithDDIMInversion.from_pretrained(
        model_id,
        safety_checker=None,
        scheduler=DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    )
    sd_model = sd_model.to(device)
    img_size = 512
    attn_size = 64

    dataset_dir = "/PerMIRS"
    for vid_id in tqdm(os.listdir(dataset_dir)):
        try:
            frames = []
            masks = []
            masks_img_size = []
            masks_np = np.load(f"{dataset_dir}/{vid_id}/masks.npz.npy", allow_pickle=True)
            for f in range(3):
                xmin, ymin, xmax, ymax = [int(x) for x in bbox_from_mask(list(masks_np[f].values())[0])]
                m, curr_offsets = load_im_into_format_from_image(
                    PIL.Image.fromarray(np.uint8(list(masks_np[f].values())[0])),
                    min_obj_x=xmin, max_obj_x=xmax)
                masks += [np.asarray(m.resize((attn_size, attn_size)))]
                masks_img_size += [np.asarray(m.resize((img_size, img_size)))]
                frames += [
                    load_im_into_format_from_path(f"{dataset_dir}/{vid_id}/{f}.jpg", offsets=curr_offsets).resize((img_size, img_size)).convert("RGB")]

            all_attn = []
            for f in frames:
                curr_attn, _ = extract_attention(sd_model, f, "A photo")
                all_attn += [curr_attn]
            torch.save(all_attn, f"{dataset_dir}/{vid_id}/diff_feats.pt")
        except Exception as e:
            f = open(f"{dataset_dir}/{vid_id}/error.txt", "w")
            f.writelines(str(e))
            f.close()