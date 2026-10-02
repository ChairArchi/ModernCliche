# Reproducible morphology experiments

## Research question

What morphological variation appears among images retrieved under the same object name, and how can that variation drive explicit, traceable 3D hypotheses?

This instrument discovers patterns in a sample of **projected silhouettes and inferred visible-surface geometry**, not universal object grammar. The researcher chooses collection scope, audits masks and interprets contrasts. No animal, gargoyle, ornament, monster or preferred final form is hard-coded in the geometry pipeline.

## Collection and curation

Commons/Openverse returns candidates, not verified class membership. The target is never silently changed; a small visible Korean vocabulary is translated. Title, source URL, provider, creator and available license fields are stored. Downloads are bounded in count, bytes and timeout.

Duplicate detection combines content IDs, 63-bit perceptual hashes, aspect ratio and a conservative RGBA pixel-difference check. Similar morphology alone is not discarded. U²-Net, selected GrabCut or binary extraction yields a foreground candidate. Transparent alpha takes precedence. Small regions are removed and the largest connected object retained; discarded area and boundary contact are reported. Suspect masks start excluded. Individual exclusion and matching-size manual masks are supported. All failures and duplicates are logged.

The method assumes one dominant connected object per image. Wall-attached sculpture, occlusions and multiple animals may violate this assumption. Curated transparent images or manual masks are appropriate when automatic extraction fails.

## Representation

Crop → aspect-preserving scale normalisation → centre by mass → empty margin. Orientation and pose remain unchanged; front and side images are not anatomically registered. Procrustes registration requires correspondences we do not have.

For foreground mask M: `D(M) = EDT(M) - EDT(not M)`. Positive signed-distance values lie inside. These grids connect analysis directly to synthesis.

Twelve descriptors measure aspect ratio, occupancy, solidity, eccentricity, horizontal/vertical symmetry, perimeter²/(4π area), holes, skeleton endpoints, branch regions, mean medial radius and radius variation. Branch counts are grid measurements, not anatomy.

The silhouette feature space balances a standardised descriptor block and up to six spatial field PCA scores. Optional DINOv2/CLIP use foreground-only crops. With depth enabled, normalise the silhouette/visual block to unit mean squared row norm and append six standardised depth descriptors divided by sqrt(6). This gives balanced block energy when descriptors vary; constant descriptors contribute zero. Saved features already include the depth block and are not appended twice on reproduction.

Depth descriptors: robust 5–95% camera-Z extent / maximum robust X/Y extent; (depth q95−q05)/median depth; median adjacent relative depth difference; fraction of adjacent pairs with relative jump >0.03; median absolute Z component of adjacent 3D directions; valid predicted pixels / selected foreground pixels. These are inferred geometry descriptors, not measured distances or calibrated uncertainty. Group contrasts are reported separately for depth and silhouette. Cohort inference requires the same model ID, revision and depth scale convention. An inference failure aborts the cohort analysis; it never substitutes artificial depth for one member.

All foundation models are pretrained and used for inference. No search-specific 3D network is trained. Different poses, occlusion and camera estimates affect depth statistics; category-wide depth averages are not merged into an individual 3D object.

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

## Embedded monocular geometry

Infer from the original RGB image with its background/context retained. Foreground masks only select predicted object pixels after inference. Resize with preserved aspect ratio to a configurable maximum of 384, 512 or 768; do not feed the 64–128 aligned binary grids to the depth network.

Default: **MoGe-2 Base**, fixed checkpoint revision, resolution level 9, maximum source dimension 768, fp32 on CPU. Small is the lower-resource option. Large shares the adapter but has not been exercised here. CUDA is selected automatically when available, using fp16 mixed precision. The pinned MoGe-2 runtime predates MoGe-3's GPU-specific dependencies.

MoGe predicts camera-space points, depth, validity and normalised camera intrinsics. Its projection-constrained inference relates depth Z to image pixel centres through:

`X = ((x+0.5)/W − cx) Z / fx`

`Y = ((y+0.5)/H − cy) Z / fy`

MoGe's metre scale is a model estimate and is not verified against a ruler or scan. Keep it in the saved point/depth maps; normalise exported geometry to the user's requested maximum size. Do not describe exported millimetres as measured real size.

Alternative: **Depth Anything V2 Small** provides relative disparity, without calibrated focal length or inverse-depth offset. Store its raw disparity. For an explicit comparison surface, normalise disparity with foreground 1st/99th percentiles, clip to [0,1], then assume `Z = 1/(1+d)`. Assume horizontal FoV=60° and centre=(0.5,0.5), derive normalised fx/fy, and backproject. This conversion is a documented modelling assumption, not recovered metric geometry. Use MoGe as the primary reconstruction path.

Cache inference by RGB bytes, reviewed mask, model ID/revision, max resolution, detail, device and algorithm version. Serialize arrays without pickle. Cache corruption triggers a recomputation. Model or download errors are displayed; there is no automatic silhouette-radius substitution.

## Visible surface meshing

Use the predicted point map, not a silhouette distance-to-boundary radius. Triangulate adjacent image-grid vertices after intersecting model validity with the reviewed foreground. Reject triangles crossing invalid pixels or a depth discontinuity:

`(max triangle Z − min triangle Z) / min triangle Z > edge_threshold`

Default threshold=0.03; record user changes. Remove degenerate triangles. Keep disconnected patches and report their count. When a cut leaves surface fans meeting at one vertex, duplicate that vertex per edge-connected fan. Coordinates stay identical; topology becomes suitable for shell closure. No invented bridge or global smoothing is applied.

Map OpenCV right/down/forward to app right/forward/up with a handedness-preserving transform. The default is an **open visible surface**. Source RGB becomes vertex colours, with a neutral geometry view available to avoid confusing photographic texture with geometric detail. GLB/PLY store vertex colours; STL does not.

Optional fabrication shell: translate the visible surface along the camera-depth axis by the requested mm thickness, reverse the back faces and join each oriented boundary. Preserve depth cuts and holes. Scale the front so the shell's final maximum extent equals the user size while keeping the selected thickness in mm. Require watertight topology. The added surface is an explicit parallel shell, not a prediction of the hidden back of the object. Watertightness does not establish a single connected printable part, absence of self-intersection or anatomical accuracy.

Metrics include valid-pixel coverage, rejected discontinuity faces, contact splits, disconnected components, topology, area, extents and optional shell volume. These are implementation diagnostics, not ground-truth 3D accuracy. Output STL/OBJ/PLY uses mm; GLB uses metres.

## Retained silhouette variation experiment

The observed silhouette, within-group signed-field PCA and between-group mean interpolation remain available under the explicitly labelled legacy path. A maximal inscribed-disc cover is lifted into ellipsoids with depth radius `depth_scale × r`, and marching cubes extracts their union. This still uses the old depth prior; **it does not inherit the monocular depth model**. Viewpoint registration and meaningful 3D correspondences would be required before PCA/averaging of category members' point maps can be treated as a shared 3D shape model. Do not present a mean of unrelated 2.5D point maps as complete 3D reconstruction.

## Reproducibility and scope

The ZIP includes input PNGs, raw/aligned masks, provenance, exclusions, labels, features, global field PCA arrays, group contrasts, validation, seed, dependency versions, generation settings, generated mask, selected/cohort depth and point arrays, fixed model revisions and meshes. CLI reproduction uses saved masks, features and depth arrays, without collection, segmentation or depth model downloads. Match package versions for numeric comparison.

Earlier opaque payloads are removed from the active tree; Git history preserves the old implementation. The entrypoint stays `streamlit_app.py`.

Supported claim: data analysis can drive a traceable shape experiment before a final design is selected. Not supported: human uncanny response, a newly trained GAN, genuine mode collapse, semantic anatomy or exact 3D reconstruction. These require separate studies.
