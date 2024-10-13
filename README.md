# Where's Waldo: Diffusion Features For Personalized Segmentation and Retrieval (NeurIPS 2024)
> Dvir Samuel, Rami Ben-Ari, Matan Levy, Nir Darshan, Gal Chechik    
> Bar Ilan University, The Hebrew University of Jerusalem, NVIDIA Research

>
>
> Personalized retrieval and segmentation aim to locate specific instances within a dataset based on an input image and a short description of the reference instance. While supervised methods are effective, they require extensive labeled data for training. Recently, self-supervised foundation models have been introduced to these tasks showing comparable results to supervised methods. However, a significant flaw in these models is evident: they struggle to locate a desired instance when other instances within the same class are presented. In this paper, we explore text-to-image diffusion models for these tasks. Specifically, we propose a novel approach called PDM for Personalized Diffusion Features Matching, that leverages intermediate features of pre-trained text-to-image models for personalization tasks without any additional training. PDM demonstrates superior performance on popular retrieval and segmentation benchmarks, outperforming even supervised methods. We also highlight notable shortcomings in current instance and segmentation datasets and propose new benchmarks for these tasks.


<a href="https://arxiv.org/abs/2405.18025"><img src="https://img.shields.io/badge/arXiv-2405.18025-b31b1b.svg" height=22.5></a>
<a href="https://dvirsamuel.github.io/pdm.github.io/" rel="nofollow"><img src="https://camo.githubusercontent.com/ef82193f89c1e8f821031c916df3beccd5dd2c335309055d265d647a89e064e8/68747470733a2f2f696d672e736869656c64732e696f2f7374617469632f76313f6c6162656c3d50726f6a656374266d6573736167653d5765627369746526636f6c6f723d726564" height="20.5" data-canonical-src="https://img.shields.io/static/v1?label=Project&amp;message=Website&amp;color=red" style="max-width: 100%;"></a></p>

![image](https://github.com/user-attachments/assets/c90fcb80-52f3-4a1e-9b08-7c93528d3c6d)
Personalized segmentation task involves segmenting a specific reference object in a new scene. Our method is capable to accurately identify the specific reference instance in the target image, even when other objects from the same class are present. While other methods capture visually or semantically similar objects, our method can successfully extract the identical instance, by using a new personalized feature map and fusing semantic and appearance cues. Red and green indicate incorrect and correct segmentations respectively.


<br>

## Requirements

Quick installation using pip:
```
torch==2.0.1
torchvision==0.15.2
diffusers==0.18.2
transformers==4.32.0.dev0
```

## Personalized Diffusion Features Matching (PDM)

To run PDM visualization between two images run the following:

```
python pdm_matching.py
```

## PerMIR and PerMIS Datasets

The PerMIR and PerMIS datasets were sourced from the [BURST](https://github.com/Ali2500/BURST-benchmark) repository. 

### Instructions:
1. Download the datasets from the BURST repository.
2. Run the script `PerMIRS/permirs_gen_dataset.py` to prepare the personalization datasets.
3. Execute `PerMIRS/extract_diff_features.py` to extract PDM and DIFT features from each image in the dataset.


## Evaluation on PerMIR

For PDM evaluation on PerMIR dataset (personalized retrieval) run:

```
python pdm_permir.py
```

## Evaluation on PerMIS

For PDM evaluation on PerMIS dataset (personalized segmentation) run:

```
python pdm_permis.py
```



## Cite Our Paper
If you find our paper and repo useful, please cite:
```
@article{Samuel2024Waldo,
  title={Where's Waldo: Diffusion Features For Personalized Segmentation and Retrieval},
  author={Dvir Samuel and Rami Ben-Ari and Matan Levy and Nir Darshan and Gal Chechik},
  journal={NeurIPS},
  year={2024}
}
```
