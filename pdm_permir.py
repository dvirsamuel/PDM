import PIL
import numpy as np
import torch
import os
from diffusers import DDIMScheduler
from sklearn.metrics import average_precision_score
from torchvision.transforms import PILToTensor
from tqdm import tqdm
from StableDiffusionPipelineWithDDIMInversion import StableDiffusionPipelineWithDDIMInversion
from dift import SDFeaturizer, get_correspondences_seg
from PerMIRS.visualization_utils import bbox_from_mask
import torch.nn.functional as F


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

if __name__ == "__main__":
    model_id = "CompVis/stable-diffusion-v1-4"

    device = "cuda"
    sd_model = StableDiffusionPipelineWithDDIMInversion.from_pretrained(
        model_id,
        safety_checker=None,
        scheduler=DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    )
    sd_model = sd_model.to(device)

    dift = SDFeaturizer(sd_model)
    img_size = 512
    attn_size = 64
    dift_size = 32

    dataset_dir = "/PerMIRS"

    if "red_data.pth" not in os.listdir(dataset_dir):
        # organize data for retrieval
        query_frames = []
        query_labels = []
        query_dift_features = []
        query_dift_mask = []
        query_perdiff_features = []
        gallery_frames = []
        gallery_labels = []
        gallery_dift_features = []
        gallery_perdiff_features = []

        for vid_idx, vid_id in tqdm(enumerate(os.listdir(dataset_dir))):
            masks_attn = []
            masks_dift = []
            masks_relative_size = []
            masks_np = np.load(f"{dataset_dir}/{vid_id}/masks.npz.npy", allow_pickle=True)
            attn_ls = torch.load(f"{dataset_dir}/{vid_id}/diff_feats.pt")
            dift_features = []
            frame_paths = []
            for f in range(3):
                xmin, ymin, xmax, ymax = [int(x) for x in bbox_from_mask(list(masks_np[f].values())[0])]
                m, curr_offsets = load_im_into_format_from_image(
                    PIL.Image.fromarray(np.uint8(list(masks_np[f].values())[0])),
                    min_obj_x=xmin, max_obj_x=xmax)

                masks_attn += [np.asarray(m.resize((attn_size, attn_size)))]
                masks_dift += [np.asarray(m.resize((dift_size, dift_size)))]
                masks_relative_size += [np.asarray(m).sum() / (img_size * img_size)]
                path = f"{dataset_dir}/{vid_id}/{f}.jpg"
                frame_paths += [path]
                frame = load_im_into_format_from_path(path, offsets=curr_offsets).resize(
                    (img_size, img_size)).convert("RGB")
                curr_dift = dift.forward((PILToTensor()(frame) / 255.0 - 0.5) * 2,
                                         prompt="A photo",
                                         ensemble_size=2)
                dift_features += [curr_dift]

            masks_relative_size = np.array(masks_relative_size)
            # remove small frames
            if len(np.where(np.array(masks_relative_size) < 0.005)[0]) > 0:
                continue

            query_idx = masks_relative_size.argmax()
            for i in range(len(dift_features)):
                if i == query_idx:
                    # query
                    query_dift_features += [dift_features[i]]
                    query_dift_mask += [masks_dift[i]]
                    query_perdiff_features += [attn_ls[i]]
                    query_labels += [vid_idx]
                    query_frames += [frame_paths[i]]
                else:
                    # gallery
                    gallery_dift_features += [dift_features[i]]
                    gallery_perdiff_features += [attn_ls[i]]
                    gallery_labels += [vid_idx]
                    gallery_frames += [frame_paths[i]]

        query_labels = torch.tensor(query_labels)
        gallery_labels = torch.tensor(gallery_labels)

        torch.save([query_frames, query_labels, query_dift_features, query_dift_mask, query_perdiff_features,
                    gallery_frames, gallery_labels, gallery_dift_features, gallery_perdiff_features], f"{dataset_dir}/ret_data.pt")
    else:
        # retrieval performance on PerMIR
        query_frames, query_labels, query_dift_features, query_dift_mask, query_perdiff_features, \
        gallery_frames, gallery_labels, gallery_dift_features, gallery_perdiff_features = torch.load(
            f"{dataset_dir}/ret_data.pt")
        topk = 1
        recall_dict = {1: 0, 5: 0, 10: 0, 50: 0}
        ap = []
        for q_idx in tqdm(range(len(query_dift_features))):
            scores = []
            for g_idx in range(len(gallery_dift_features)):
                # First, extract correspondences using DIFT
                unfiltered_ref_points, unfiltered_tgt_points, cs_sim = get_correspondences_seg(query_dift_features[q_idx],
                                                                                               gallery_dift_features[g_idx],
                                                                                               query_dift_mask[q_idx],
                                                                                               img_size=attn_size,
                                                                                               topk=topk)
                # for better matching, we use all corresponding points found by DIFT for retrieval
                ref_points = unfiltered_ref_points
                tgt_points = unfiltered_ref_points
                dift_scores = cs_sim

                total_maps_scores = []
                total_maps_scores_cs = []
                for attn_idx, (ref_attn, target_attn) in enumerate(
                        zip(query_perdiff_features[q_idx], gallery_perdiff_features[g_idx])):
                    all_point_scores = []
                    all_point_scores_cs = []
                    for p_ref, p_tgt, dift_score in zip(ref_points, tgt_points, dift_scores):
                        source_vec = ref_attn[
                            torch.tensor(np.ravel_multi_index(p_ref, dims=(attn_size, attn_size))).to("cuda")].reshape(1,
                                                                                                                       -1)
                        target_vec = target_attn[
                            torch.tensor(np.ravel_multi_index(p_tgt.T, dims=(attn_size, attn_size))).to(
                                "cuda")].reshape(topk, -1)
                        euclidean_dist = torch.cdist(source_vec, target_vec)
                        all_point_scores += [dift_score * euclidean_dist.mean().cpu().numpy()]
                        all_point_scores_cs += [
                            dift_score * (F.normalize(source_vec) @ F.normalize(target_vec).T).mean().cpu().numpy()]
                    total_maps_scores += [np.mean(all_point_scores)]
                    total_maps_scores_cs += [np.mean(all_point_scores_cs)]
                total_score = np.mean(total_maps_scores)
                total_score_cs = np.mean(total_maps_scores_cs)
                scores += [total_score_cs]

            pred_scores_idx = torch.argsort(torch.tensor(scores), descending=True)  # change to false fro euclidean
            pred_g_labels = gallery_labels[pred_scores_idx]
            curr_query_lbl = query_labels[q_idx]

            ap += [average_precision_score((gallery_labels == curr_query_lbl).int().numpy(),scores)]
            for r in [1, 5, 10, 50]:
                if curr_query_lbl in pred_g_labels[:r]:
                    recall_dict[r] += 1

        print("MAP:", np.array(ap).mean())
        for k in recall_dict.keys():
            print(f"Recall@{k}", recall_dict[k] / len(query_dift_features))
