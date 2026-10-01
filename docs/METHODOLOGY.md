# Reproducible morphology experiments

## Research question

What morphological variation appears among images retrieved under the same object name, and how can that variation drive explicit, traceable 3D hypotheses?

This instrument discovers patterns in a sample of **projected silhouettes**, not universal object grammar. The researcher chooses collection scope, audits masks and interprets contrasts. No animal, gargoyle, ornament, monster or preferred final form is hard-coded in the geometry pipeline.

## Collection and curation

Commons/Openverse returns candidates, not verified class membership. The target is never silently changed; a small visible Korean vocabulary is translated. Title, source URL, provider, creator and available license fields are stored. Downloads are bounded in count, bytes and timeout.

Duplicate detection combines content IDs, 63-bit perceptual hashes, aspect ratio and a conservative RGBA pixel-difference check. Similar morphology alone is not discarded. U²-Net, selected GrabCut or binary extraction yields a foreground candidate. Transparent alpha takes precedence. Small regions are removed and the largest connected object retained; discarded area and boundary contact are reported. Suspect masks start excluded. Individual exclusion and matching-size manual masks are supported. All failures and duplicates are logged.

The method assumes one dominant connected object per image. Wall-attached sculpture, occlusions and multiple animals may violate this assumption. Curated transparent images or manual masks are appropriate when automatic extraction fails.

## Representation

Crop → aspect-preserving scale normalisation → centre by mass → empty margin. Orientation and pose remain unchanged; front and side images are not anatomically registered. Procrustes registration requires correspondences we do not have.

For foreground mask M: `D(M) = EDT(M) - EDT(not M)`. Positive signed-distance values lie inside. These grids connect analysis directly to synthesis.

Twelve descriptors measure aspect ratio, occupancy, solidity, eccentricity, horizontal/vertical symmetry, perimeter²/(4π area), holes, skeleton endpoints, branch regions, mean medial radius and radius variation. Branch counts are grid measurements, not anatomy.

The default feature space balances a standardised descriptor block and up to six spatial field PCA scores. Optional DINOv2/CLIP group foreground-only crops; all backends retain the same shape measurements and field synthesis. Model downloads are explicit; pretrained features are not retrained.

## Sorting and group discovery

Candidate K-means counts: 2…min(6, floor(n/2)); fixed seed and multiple starts. Reject singleton groups. Select by silhouette score. For each candidate, refit eight times on 80% subsamples and compare overlapping partitions using ARI. Automatic selection keeps one group if silhouette<0.18 or mean ARI<0.35. These are declared **practical exploratory thresholds**, not significance tests. Manual group count is recorded; degenerate partitions still fall back.

The scatter's PCA projection is display-only. Clustering uses the pre-projection feature matrix. Each group has a medoid, member masks, feature means and standardised differences relative to the whole dataset. Numeric group names avoid invented semantics. Imported labels, including CAPTCHA labels, are a separate cross-tabulation; `unclear` is not a monster score.

## Statistical shape variation

Global and within-group field PCA retain up to 12 non-degenerate axes. With mean μ, direction v_j, score standard deviation s_j and coefficient b_j:

`D_generated = μ + Σ (b_j × s_j × v_j)`

Axis previews show −2σ, mean, +2σ and descriptor correlations. Beyond ±2σ is marked exploratory extrapolation, without assuming a Gaussian population.

Three sources: an observed member/medoid; within-group field variation; weighted means of two groups. Threshold at zero. Empty outputs are rejected; disconnected components are counted, retained and warned about rather than joined by invented geometry.

With ≥8 samples, five 75/25 holdout splits fit field PCA only on training samples and reconstruct held-out fields with up to six axes. Compare RMSE and mask IoU with the training-mean-only baseline. Poor results remain visible. This evaluates sample reconstruction, not category recognition, external generalisation or 3D accuracy.

This adapts the statistical-shape principle of Cootes et al., not landmark ASM. Different viewpoints may blur the mean; group inspection is essential. Correlations and feature contrasts do not establish causality or population laws.

## 3D hypothesis

Measure a medial axis and its inscribed radii. Discrete thinning sometimes omits extremities; compute a maximal inscribed-disc cover from all foreground points, removing discs analytically contained within larger discs. Use distance-to-background minus half a pixel, with a 0.6-pixel minimum to retain narrow connected structures. Radius measurements come from the silhouette, not brightness.

Lift each disc centre to the central depth plane. An ellipsoid has planar radius r and depth radius `depth_scale × r`. Union implicit ellipsoid fields and extract the zero isosurface with marching cubes. The silhouette's branches, holes and thickness determine the volume; no image rectangles or base plate are included.

**Depth is an explicit prior**, not recovered object anatomy. Default depth_scale=1 assumes locally isotropic thickness. The user can vary it and records the choice. Maximum physical dimension is scaled in mm. The app measures foreground projection IoU, mesh components, bounds, volume and watertightness. Thin features remain resolution-limited; disconnected components are not automatically fabrication-ready.

STL/OBJ coordinates use mm (those formats carry no inherent unit). GLB is exported in metres for glTF interoperability. Scale and measurements are saved in run.json.

## Reproducibility and scope

The ZIP includes input PNGs, raw/aligned masks, provenance, exclusions, labels, features, global field PCA arrays, group contrasts, validation, seed, dependency versions, generation settings, generated mask and meshes. CLI reproduction uses saved masks and features, without collection or segmentation model downloads. Match package versions for numeric comparison.

Earlier opaque payloads are removed from the active tree; Git history preserves the old implementation. The entrypoint stays `streamlit_app.py`.

Supported claim: data analysis can drive a traceable shape experiment before a final design is selected. Not supported: human uncanny response, a newly trained GAN, genuine mode collapse, semantic anatomy or exact 3D reconstruction. These require separate studies.
