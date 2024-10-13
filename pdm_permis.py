import os
import PIL
import numpy as np
import torch
from diffusers import DDIMScheduler
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm
from transformers import SamModel, SamProcessor
import ptp_utils
from StableDiffusionPipelineWithDDIMInversion import StableDiffusionPipelineWithDDIMInversion
from PerMIRS.eval_miou import AverageMeter, intersectionAndUnion
from attention_store import AttentionStore
from PerMIRS.visualization_utils import bbox_from_mask
from sam_utils import show_points_on_image, show_masks_on_image, show_single_mask_on_image
import torch.nn.functional as F


def get_sam_masks(sam_model, processor, raw_image, input_points, input_labels, attn, ref_embeddings, input_masks=None, box=None,
                  multimask_output=True, verbose=False):
    inputs = processor(raw_image, input_points=input_points, input_labels=input_labels, input_boxes=box,
                       return_tensors="pt",
                       attention_similarity=attn).to("cuda")
    inputs["attention_similarity"] = attn
    inputs["target_embedding"] = ref_embeddings
    inputs["input_masks"] = input_masks
    inputs["multimask_output"] = multimask_output
    with torch.no_grad():
        outputs = sam_model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )
    scores = outputs.iou_scores
    logits = outputs.pred_masks
    if verbose:
        if not multimask_output:
            scores = scores.reshape(1, -1)
            masks = masks[0].reshape(1, masks[0].shape[-2], masks[0].shape[-1])
            show_single_mask_on_image(raw_image, masks[0], scores)
        else:
            show_masks_on_image(raw_image, masks[0], scores)
        show_points_on_image(raw_image, input_points[0], input_labels[0])
    return masks, scores, logits

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


def diff_attn_images(sd_model, sam_model, sam_processor, ref_img, ref_mask, ref_attn_ls, tgt_img, target_attn_ls,
                     verbose=True):
    image_size = 512
    prompt = f"A photo"
    h = w = 64
    mask_idx_y, mask_idx_x = torch.where(ref_mask)

    all_points = []
    all_labels = []
    for attn_idx, (ref_attn, target_attn) in enumerate(zip(ref_attn_ls, target_attn_ls)):
        heatmap = torch.zeros(ref_attn.shape[0]).to("cuda")
        for x, y in zip(mask_idx_x, mask_idx_y):
            t = np.ravel_multi_index((y, x), dims=(h, w))
            source_vec = ref_attn[t].reshape(1, -1)
            cs_similarity = F.normalize(source_vec) @ F.normalize(target_attn).T
            idx = torch.sort(cs_similarity, descending=True)[1][0][:100]
            heatmap[idx] += 1
        heatmap = heatmap / heatmap.max()
        heatmap_img_size = \
            nn.Upsample(size=(image_size, image_size), mode='bilinear')(heatmap.reshape(1, 1, 64, 64))[0][
                0]
        if verbose:
            plt.imshow(tgt_img)
            plt.imshow(heatmap_img_size.cpu(), alpha=0.6)
            plt.axis("off")
            plt.show()

        all_points += [(heatmap_img_size == torch.max(heatmap_img_size)).nonzero()[0].reshape(1, -1).cpu().numpy()]
        all_labels += [1]

    all_points = np.concatenate(all_points)
    all_points[:, [0, 1]] = all_points[:, [1, 0]]
    pred_masks, masks_scores, mask_logits = get_sam_masks(sam_model, sam_processor, tgt_img,
                                                          input_points=[list(all_points)],
                                                          input_labels=[all_labels],
                                                          attn=None,
                                                          ref_embeddings=None,
                                                          verbose=verbose
                                                          )
    best_mask = 2
    y, x = pred_masks[0][0][best_mask].nonzero().T
    x_min = x.min().item()
    x_max = x.max().item()
    y_min = y.min().item()
    y_max = y.max().item()
    input_box = [[x_min, y_min, x_max, y_max]]
    pred_masks, masks_scores, mask_logits = get_sam_masks(sam_model, sam_processor, tgt_img,
                                                          input_points=[list(all_points)],
                                                          input_labels=[all_labels],
                                                          attn=None,
                                                          ref_embeddings=None,
                                                          # input_masks=mask_logits[0, 0, best_mask: best_mask + 1, :,:],
                                                          box=[input_box],
                                                          multimask_output=True,
                                                          verbose=verbose
                                                          )
    best_idx = 2
    final_pred_mask = pred_masks[0][:, best_idx, :, :].squeeze().numpy()
    return final_pred_mask


if __name__ == "__main__":
    model_id = "CompVis/stable-diffusion-v1-4"

    device = "cuda"

    sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    sd_model = StableDiffusionPipelineWithDDIMInversion.from_pretrained(
        model_id,
        safety_checker=None,
        scheduler=DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    )
    sd_model = sd_model.to(device)
    img_size = 512
    attn_size = 64

    dataset_dir = "/PerMIRS"
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    for vid_id in tqdm(os.listdir(dataset_dir)):
        frames = []
        masks = []
        masks_img_size = []
        masks_np = np.load(f"{dataset_dir}/{vid_id}/masks.npz.npy", allow_pickle=True)
        attn_ls = torch.load(f"{dataset_dir}/{vid_id}/diff_feats.pt")
        for f in range(3):
            xmin, ymin, xmax, ymax = [int(x) for x in bbox_from_mask(list(masks_np[f].values())[0])]
            m, curr_offsets = load_im_into_format_from_image(
                PIL.Image.fromarray(np.uint8(list(masks_np[f].values())[0])),
                min_obj_x=xmin, max_obj_x=xmax)
            masks += [np.asarray(m.resize((attn_size, attn_size)))]
            masks_img_size += [np.asarray(m.resize((img_size, img_size)))]
            frames += [
                load_im_into_format_from_path(f"{dataset_dir}/{vid_id}/{f}.jpg", offsets=curr_offsets).resize(
                    (img_size, img_size)).convert("RGB")]

        ref_idx = 0
        tgt_idx = 1

        # remove tiny objects
        if masks_img_size[tgt_idx].sum() / (img_size * img_size) < 0.005:
            continue
        pred_mask = diff_attn_images(sd_model, sam_model, sam_processor, frames[ref_idx], torch.tensor(masks[ref_idx]),
                                     attn_ls[ref_idx][0:2],
                                     frames[tgt_idx], attn_ls[tgt_idx][0:2], verbose=False)

        pred_mask = np.uint8(pred_mask)
        gt_mask = np.uint8(masks_img_size[tgt_idx])

        intersection, union, target = intersectionAndUnion(pred_mask, gt_mask)
        print(vid_id, intersection, union, target)
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)

    print("\nmIoU: %.2f" % (100 * iou_class))
    print("mAcc: %.2f\n" % (100 * accuracy_class))
