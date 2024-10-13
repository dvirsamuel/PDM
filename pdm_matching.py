import PIL
import numpy as np
import torch
from diffusers import DDIMScheduler
from matplotlib import pyplot as plt
from torch import nn
from transformers import SamModel, SamProcessor
from attention_store import AttentionStore
from StableDiffusionPipelineWithDDIMInversion import StableDiffusionPipelineWithDDIMInversion
#from sam_utils import show_masks_on_image, show_points_on_image
import ptp_utils


def center_crop(im):
    width, height = im.size  # Get dimensions
    min_dim = min(width, height)
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im


def load_im_into_format_from_path(im_path):
    return center_crop(PIL.Image.open(im_path)).resize((512, 512))

def get_masks(raw_image, input_points):
    inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )
    # visualize sam segmentation
    # scores = outputs.iou_scores
    # show_masks_on_image(raw_image, masks[0], scores)
    # show_points_on_image(raw_image, input_points[0])
    return masks


def extract_attention(sd_model, image, prompt):
    controller = AttentionStore()
    inv_latents = sd_model.invert(prompt, image=image, guidance_scale=1.0).latents
    ptp_utils.register_attention_control_efficient(sd_model, controller)
    recon_image = sd_model(prompt, latents=inv_latents, guidance_scale=1.0).images[0]
    key_format = f"up_blocks_3_attentions_2_transformer_blocks_0_attn1_self"
    timestamp = 49
    res_attn = 64
    return [controller.attention_store[timestamp]["Q_" + key_format][0].to("cuda"),
            controller.attention_store[timestamp]["K_" + key_format][0].to("cuda"),
            ], sd_model.unet.up_blocks[3].attentions[2].transformer_blocks[0].attn1, res_attn

def heatmap(sd_model, ref_img, tgt_img):
    image_size = 512
    prompt = f"A photo"
    # extract query PDM features (Q and K)
    ref_attn_ls, _, res_attn = extract_attention(sd_model, ref_img, prompt)
    h = w = res_attn

    # get query mask using SAM or use provided mask from user
    source_masks = get_masks(ref_img.resize(size=(h, w)), [[[h // 2, w // 2]]])
    source_mask = source_masks[0][:, 1:2, :, :].squeeze(dim=0).squeeze(dim=0)
    mask_idx_y, mask_idx_x = torch.where(source_mask)

    # extract target PDM features (Q and K)
    target_attn_ls, _, _ = extract_attention(sd_model, tgt_img, prompt)

    # apply matching and show heatmap
    for attn_idx, (ref_attn, target_attn) in enumerate(zip(ref_attn_ls, target_attn_ls)):
        heatmap = torch.zeros(ref_attn.shape[0]).to("cuda")
        for x, y in zip(mask_idx_x, mask_idx_y):
            t = np.ravel_multi_index((y, x), dims=(h, w))
            source_vec = ref_attn[t].reshape(1, -1)
            euclidean_dist = torch.cdist(source_vec, target_attn)
            idx = torch.sort(euclidean_dist)[1][0][:100]
            heatmap[idx] += 1
        heatmap = heatmap / heatmap.max()
        heatmap_img_size = \
            nn.Upsample(size=(image_size, image_size), mode='bilinear')(heatmap.reshape(1, 1, 64, 64))[0][
                0]
        plt.imshow(tgt_img)
        plt.imshow(heatmap_img_size.cpu(), alpha=0.6)
        plt.show()


if __name__ == "__main__":
    model_id = "CompVis/stable-diffusion-v1-4"

    device = "cuda"

    sd_model = StableDiffusionPipelineWithDDIMInversion.from_pretrained(
        model_id,
        safety_checker=None,
        scheduler=DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    )
    sd_model = sd_model.to(device)
    img_size = 512

    model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    ref_img = load_im_into_format_from_path("dogs/source.png").convert("RGB")
    plt.imshow(ref_img)
    plt.show()
    tgt_img = load_im_into_format_from_path("dogs/target1.png").convert("RGB")
    plt.imshow(tgt_img)
    plt.show()
    heatmap(sd_model, ref_img, tgt_img)

