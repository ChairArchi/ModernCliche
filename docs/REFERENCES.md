# Method references

This project adapts the following methods; it does not reproduce the full papers or train a new foundation model.

| Source | Role here | Boundary |
|---|---|---|
| Cootes, Taylor, Cooper & Graham (1995), [Active Shape Models—Their Training and Application](https://doi.org/10.1006/cviu.1995.1004) · [author PDF](https://personalpages.manchester.ac.uk/staff/timothy.f.cootes/papers/cootes_cviu95.pdf) | Mean + PCA shape directions; coefficients in standard deviations | Our PCA operates on aligned signed-distance grids, not corresponding landmarks or the full ASM fitting algorithm |
| Blum (1967), [A Transformation for Extracting New Descriptors of Shape](https://pageperso.lis-lab.fr/~edouard.thiel/rech/1967-blum.pdf) | Medial representation and measured inscribed radii | Extending discrete maximal circles into 3D ellipsoids is our explicit depth prior |
| Nguyen, Yosinski & Clune (2016), [Multifaceted Feature Visualization](https://arxiv.org/abs/1602.03616) | Motivation for separating multiple appearances before synthesising a representative | Conceptual guidance; no neuron activation maximisation is implemented |
| Qin et al. (2020), [U²-Net](https://arxiv.org/abs/2005.09007) · [rembg](https://github.com/danielgatis/rembg) | Pretrained foreground saliency, compact `u2netp`, CPU inference | Saliency does not label anatomical parts; masks require review |
| Oquab et al. (2023), [DINOv2](https://arxiv.org/abs/2304.07193) | Optional visual embeddings for foreground crops | Does not generate images or reconstruct 3D |
| Radford et al. (2021), [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) | Optional CLIP embeddings and query cosine similarity | Similarity is not a probability or human recognition score |
| Wang et al. (2025), [MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision](https://arxiv.org/abs/2410.19115), CVPR 2025 · [official code](https://github.com/microsoft/MoGe) | Direct camera-space point-map and camera-intrinsics prediction | Pretrained inference, not category-specific training or reconstruction of unseen surfaces |
| Wang et al. (2025), [MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details](https://arxiv.org/abs/2507.02546) · [Base checkpoint](https://huggingface.co/Ruicheng/moge-2-vitb-normal) | Default embedded depth/point-map inference; CPU Small/Base, optional Large | Metric scale remains an unverified prediction; output fabrication size is user normalised. Large is implemented but not locally inference-tested. |
| Yang et al. (2024), [Depth Anything V2](https://arxiv.org/abs/2406.09414), NeurIPS 2024 · [official code](https://github.com/DepthAnything/Depth-Anything-V2) · [Transformers API](https://huggingface.co/docs/transformers/model_doc/depth_anything_v2) | Alternative embedded relative-disparity inference | The 60° camera and inverse-depth offset used to backproject are project assumptions; this is not calibrated metric depth. |
| [scikit-learn silhouette analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html), [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html), [ARI](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html) | Clustering, practical cluster-count selection, resampling consistency, PCA | Feature-space quality is not semantic truth |
| [scikit-image medial axis](https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.medial_axis), [marching cubes](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.marching_cubes) | Skeleton/radii and implicit-field surface extraction | Mesh topology and projection are checked separately |

## CAPTCHA data

[Sivakorn, Polakis & Keromytis (2016) archive](https://github.com/ssivakorn/img-CAPTCHA-challenge-dataset) supplies image challenges and human category labels, including `unclear`. Preserve labels through a `filename,label` CSV alongside selected image uploads. Dataset labels and discovered morphology groups remain distinct.

CAPTCHA images have their own rights; an archive's code license is not a blanket image license. No CAPTCHA dataset is bundled. This application neither collects live authentication challenges nor solves them.

## Pinned inference runtime and licenses

MoGe-2 code: [925b8ed](https://github.com/microsoft/MoGe/tree/925b8ed835a7a9cdb7578ba15c658a0afc969030), MIT, with Apache-2.0 DINOv2 components. Model ID and exact checkpoint revision are recorded in each depth NPZ and `run.json`. Depth Anything V2 **Small** is Apache-2.0; the project does not use the differently licensed larger Depth Anything V2 weights. Refer to upstream model cards for the model terms and source datasets. No model weights or source photographs are committed to this repository.

The project implements image-grid triangulation and explicit shell closure. It does not implement MoGe-3, DUSt3R/MASt3R camera registration, multi-view fusion or a generative 360° completion model.
