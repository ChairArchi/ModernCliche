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
| [scikit-learn silhouette analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html), [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html), [ARI](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html) | Clustering, practical cluster-count selection, resampling consistency, PCA | Feature-space quality is not semantic truth |
| [scikit-image medial axis](https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.medial_axis), [marching cubes](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.marching_cubes) | Skeleton/radii and implicit-field surface extraction | Mesh topology and projection are checked separately |

## CAPTCHA data

[Sivakorn, Polakis & Keromytis (2016) archive](https://github.com/ssivakorn/img-CAPTCHA-challenge-dataset) supplies image challenges and human category labels, including `unclear`. Preserve labels through a `filename,label` CSV alongside selected image uploads. Dataset labels and discovered morphology groups remain distinct.

CAPTCHA images have their own rights; an archive's code license is not a blanket image license. No CAPTCHA dataset is bundled. This application neither collects live authentication challenges nor solves them.
