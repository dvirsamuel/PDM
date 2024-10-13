import gc

import numpy as np
import torch
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from diffusers import DDIMScheduler
from diffusers import StableDiffusionPipeline
from torch import nn
import torch.nn.functional as F

from DIFT.dift import MyUNet2DConditionModel


class OneStepSDPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
            self,
            img_tensor,
            t,
            up_ft_indices,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None
    ):
        device = self._execution_device
        latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        t = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        unet_output = self.unet(latents_noisy,
                                t,
                                up_ft_indices,
                                encoder_hidden_states=prompt_embeds,
                                cross_attention_kwargs=cross_attention_kwargs)
        return unet_output


class SDFeaturizer:
    def __init__(self, sd_model):
        unet = MyUNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
        onestep_pipe = OneStepSDPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", unet=unet,
                                                         safety_checker=None)
        onestep_pipe.vae.decoder = None
        onestep_pipe.scheduler = sd_model.scheduler

        gc.collect()
        onestep_pipe = onestep_pipe.to("cuda")
        onestep_pipe.enable_attention_slicing()
        onestep_pipe.enable_xformers_memory_efficient_attention()
        self.pipe = onestep_pipe

    @torch.no_grad()
    def forward(self,
                img_tensor,  # single image, [1,c,h,w]
                prompt,
                t=261,
                up_ft_index=1,
                ensemble_size=8):
        img_tensor = img_tensor.repeat(ensemble_size, 1, 1, 1).cuda()  # ensem, c, h, w

        prompt_embeds = self.pipe._encode_prompt(
            prompt=prompt,
            device='cuda',
            num_images_per_prompt=1,
            do_classifier_free_guidance=False)  # [1, 77, dim]

        prompt_embeds = prompt_embeds.repeat(ensemble_size, 1, 1)
        torch.manual_seed(42)
        unet_ft_all = self.pipe(
            img_tensor=img_tensor,
            t=t,
            up_ft_indices=[up_ft_index],
            prompt_embeds=prompt_embeds)
        unet_ft = unet_ft_all['up_ft'][up_ft_index]  # ensem, c, h, w
        unet_ft = unet_ft.mean(0, keepdim=True)  # 1,c,h,w
        return unet_ft


def get_correspondences(src_ft, tgt_ft, ref_bbox, img_size=512, topk=10):
    num_channel = src_ft.shape[1]
    ref_object_start_token = np.ravel_multi_index([(ref_bbox[1]), (ref_bbox[0])], (img_size, img_size))
    bbox_w, bbox_h =  ref_bbox[2] - ref_bbox[0], ref_bbox[3] - ref_bbox[1]
    with torch.no_grad():
        src_ft = nn.Upsample(size=(img_size, img_size), mode='bilinear')(src_ft)
        src_vec = F.normalize(src_ft.view(num_channel, -1))  # C, HW
        tgt_ft = nn.Upsample(size=(img_size, img_size), mode='bilinear')(tgt_ft)
        trg_vec = F.normalize(tgt_ft.view(num_channel, -1))  # C, HW
        # For efficient computation on high-memory GPUs, process all tokens simultaneously rather than per row.
        all_ref_points = []
        all_tgt_points = []
        all_cosine_similarity = []
        for i in range(bbox_h):
            curr_ref_tokens = list(range(ref_object_start_token + img_size*i, ref_object_start_token + img_size*i+bbox_w+1))
            all_ref_points += [np.array(np.unravel_index(curr_ref_tokens, shape=(img_size, img_size))).T]
            cos_map = torch.matmul(src_vec.T[curr_ref_tokens], trg_vec)
            #tgt_tokens = cos_map.argmax(dim=-1).cpu().numpy()
            res = cos_map.topk(k=topk,dim=-1)
            all_cosine_similarity += [res[0].cpu().numpy()]
            tgt_tokens = res[1].cpu().numpy()
            all_tgt_points += [np.array(np.unravel_index(tgt_tokens, shape=(img_size, img_size))).T]

    return np.concatenate(all_ref_points), np.concatenate(all_tgt_points, axis=1).reshape(-1,topk,2), np.concatenate(all_cosine_similarity).reshape(-1)

def get_correspondences_seg(src_ft, tgt_ft, src_mask, img_size=512, topk=10):
    num_channel = src_ft.shape[1]
    with torch.no_grad():
        src_ft = nn.Upsample(size=(img_size, img_size), mode='bilinear')(src_ft)
        src_vec = F.normalize(src_ft.view(num_channel, -1))  # C, HW
        tgt_ft = nn.Upsample(size=(img_size, img_size), mode='bilinear')(tgt_ft)
        trg_vec = F.normalize(tgt_ft.view(num_channel, -1))  # C, HW
        # For efficient computation on high-memory GPUs, process all tokens simultaneously rather than per row.
        all_tgt_points = []
        all_cosine_similarity = []
        all_ref_points = np.column_stack(src_mask.nonzero())
        for p in all_ref_points:
            curr_ref_tokens = np.ravel_multi_index([(p[1]), (p[0])], (img_size, img_size))
            cos_map = torch.matmul(src_vec.T[curr_ref_tokens], trg_vec)
            #tgt_tokens = cos_map.argmax(dim=-1).cpu().numpy()
            res = cos_map.topk(k=topk,dim=-1)
            all_cosine_similarity += [res[0].cpu().numpy()]
            tgt_tokens = res[1].cpu().numpy()
            all_tgt_points += [np.array(np.unravel_index(tgt_tokens, shape=(img_size, img_size))).T]

    return all_ref_points, np.concatenate(all_tgt_points, axis=1).reshape(-1,topk,2), np.concatenate(all_cosine_similarity).reshape(-1)
