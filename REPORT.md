# Camera Traps in New Places: How Few Unlabeled Images Does a Species Classifier Need to Adapt to an Unseen Camera?
CSE-414 Machine Learning and Deep Learning Lab - [Team members] - [Supervisor] - November 2026

## Abstract
Camera-trap species classifiers lose accuracy on cameras they have not seen. We ask how many unlabeled images from a new camera a lightweight test-time adaptation method needs, and whether the answer depends on the backbone. On the iWildCam benchmark (WILDS splits, a subsample at 224x224) we compare a supervised ResNet-50 and a self-supervised DINO ViT-S/16 under no adaptation, a BatchNorm-statistics update and TENT-style entropy minimization, using 16 to 128 unlabeled images per camera. For the ViT with TENT, 16 images give +1.2 points accuracy and the oracle (all images) +1.7 points; about 70% of the 27 unseen cameras improve, but macro F1 changes by under 0.01. For the ResNet-50, both methods collapse accuracy from 0.567 to about 0.12 on every camera. We report both outcomes and discuss why.

## 1. Introduction and motivation
Camera traps take a photo when an animal moves in front of them. One camera produces thousands of images and naming the species needs expert time. A model trained on some cameras meets a new background, light and angle at a new camera, and accuracy can drop sharply; the WILDS authors report macro F1 falling from about 47 on seen cameras to about 31 on unseen ones for the standard baseline [1]. Labeling every new camera is not realistic, so a method that adapts from a few unlabeled photos would make species counting cheaper. This connects to SDG 15 (life on land): Bangladesh has forests where camera traps can support monitoring, although the public dataset used here contains no Bangladeshi cameras.

## 2. Related work
WILDS and iWildCam define the train / unseen-camera setting [1]. WILDS 2.0 adds unlabeled data and benchmarks adaptation methods [2]. Test-time BatchNorm adaptation re-estimates normalization statistics on the shifted data [5]. Tent minimizes prediction entropy on unlabeled test data and updates only normalization parameters [4]. DINO is a self-supervised vision transformer [6]. Adaptation itself is therefore not new; we study a small-data question. [TODO: add the fuller literature search promised in the proposal.]

## 3. Research question and hypotheses
How many unlabeled images from an unseen camera does a lightweight test-time adaptation method need to recover accuracy, and does the answer change with the type of pretrained backbone?
* H1: adapting with a small batch of unlabeled images from each new camera improves macro F1, and most of the gain appears with few images.
* H2: a self-supervised backbone loses less on unseen cameras than a supervised ImageNet backbone under the same compute.

## 4. Method
**4.1 Data.** iWildCam from WILDS with the official splits, loaded with the `wilds` package, cached as 224x224 images. Subset sizes (images / cameras): train 8,337 / 227; id_val 3,000 / 138; id_test 3,000 / 156; val (unseen) 6,542 / 22; test (unseen) 9,246 / 27. Leakage check: no camera is shared between train, val and test; 153 of 156 id_test cameras appear in train.
**4.2 Backbones.** timm `resnet50.tv_in1k` (supervised; layer4 and head trained) and `vit_small_patch16_224.dino` (self-supervised; last 2 blocks, final norm and head trained). Training: ERM, AdamW, OneCycle, mixed precision; checkpoint selected on val cameras.
**4.3 Adaptation.** (a) none; (b) BN-statistics update from the unlabeled batch (ResNet only; a ViT has no BatchNorm); (c) TENT-style entropy minimization over normalization-layer scale and shift (for the ResNet, BN statistics are re-estimated from the adapt batch afterwards). Learning rate and steps are tuned on val cameras only.
**4.4 Protocol.** For each unseen camera and seed we draw n in {16, 32, 64, 128} unlabeled images, adapt a copy of the model, predict the remaining images of that camera, and reset. No-adaptation predictions are stored on the same held-out images, so all comparisons are paired. The oracle adapts on all images of the camera (transductive upper bound).
**4.5 Metrics.** Macro F1 over classes present in the ground truth (WILDS style) and accuracy; 95% t-intervals over 5 seeds (3 for the oracle and ablations); per-camera accuracy change with a bootstrap over cameras.

## 5. Results (27 unseen test cameras; full tables in `results_reported/summary.md`)
**5.1 Without adaptation.** ResNet-50: accuracy 0.567, macro F1 about 0.27. ViT-S DINO: accuracy 0.572, macro F1 about 0.31. A majority-class guess reaches about 0.27 accuracy. [Insert seen (id_test) vs unseen (test) numbers from `results/baseline.json` and `fig_seen_unseen.png`.]
**5.2 How many images (H1).** ViT-S + TENT, change in accuracy (percentage points, n = 16 / 32 / 64 / 128 / oracle): +1.2 / +1.4 / +1.6 / +1.5 / +1.7 with intervals of about +-0.4. Change in macro F1: +0.007 / +0.008 / +0.008 / +0.007 / +0.011. The gain is small, appears already at 16 images (about 70% of the oracle gain), and grows little beyond that.
**5.3 ResNet-50 collapse.** With BN update accuracy falls from 0.567 to 0.116 (n = 16), 0.121 (n = 64), 0.122 (n = 128) and 0.128 (oracle); macro F1 from 0.27 to about 0.06. TENT behaves the same (0.117 to 0.136). 0 of 27 cameras improve (mean change -0.438 BN, -0.432 TENT).
**5.4 Backbone (H2).** Unadapted, the two backbones are close (accuracy 0.572 vs 0.567, macro F1 0.31 vs 0.27), so H2 is only partly supported by the baseline. The clear difference is the response to adaptation: the ViT gains slightly, the ResNet collapses.
**5.5 Per camera.** ViT + TENT at n = 64: mean accuracy change +1.7 points (95% interval +0.5 to +3.0), 70.4% of cameras improved. Rank correlation of the gain with image brightness is -0.08 and with the share of empty frames 0.09, so neither explains which cameras gain.
**5.6 Ablations (n = 64).** Adapting only the last blocks: ViT +0.7 points (vs +1.6 with all norm layers); ResNet BN -35.5 and TENT -33.3 points (vs about -44). Share of empty frames in the adapt batch (0 / 25 / 50 / 75%, cameras 22 / 8 / 5 / 6): ViT gain +1.4 / +2.9 / +3.9 / +3.7 points; the groups are different cameras and small, so this is only a hint. A BN-prior sweep that shrinks the new statistics towards the source statistics only walks back to "no adaptation" on val cameras.

## 6. Discussion
H1 holds for the ViT with TENT: a small batch of 16 unlabeled images gives most of the available gain, but the gain is small in absolute terms (macro F1 under +0.01), so we do not claim adaptation is a large improvement. For the ResNet-50, H1 fails: both methods collapse accuracy on every camera. We ruled out several causes: a code bug (a huge shrinkage prior reproduces the unadapted model), too few images (the oracle collapses as well), the share of empty frames (all shares drop 0.3 to 0.45 in accuracy) and TENT hyperparameters (BN update has none). Our untested hypothesis is that a single camera shows few species and many near-identical burst frames, so its BatchNorm statistics differ strongly from the pooled training statistics, which a ResNet relies on and a LayerNorm ViT does not. We have not proved this cause.

## 7. Limitations
Subsample of iWildCam at 224x224, so numbers are not comparable to the official leaderboard; one training run per backbone; two backbones and two methods; macro F1 is noisy because of rare species; the empty-frame ablation uses few cameras; no Bangladeshi cameras were tested (a model shown Bangladeshi species will be wrong because those species are not in iWildCam); the ResNet collapse is unexplained.

## 8. Conclusion
A few unlabeled images from a new camera help a self-supervised ViT slightly, and 16 are enough for most of the gain. The same adaptation breaks a supervised ResNet-50. Adaptation is therefore not safe by default; it should be tested per backbone and per camera before use. Future work: test the BatchNorm hypothesis, other test-time methods and larger backbones, the full training set, and a pilot on Bangladeshi camera-trap images.

## References
[1] Koh et al. WILDS: A benchmark of in-the-wild distribution shifts. ICML 2021.
[2] Sagawa et al. Extending the WILDS benchmark for unsupervised adaptation. ICLR 2022.
[4] Wang et al. Tent: Fully test-time adaptation by entropy minimization. ICLR 2021.
[5] Schneider et al. Improving robustness against common corruptions by covariate shift adaptation. NeurIPS 2020.
[6] Caron et al. Emerging properties in self-supervised vision transformers. ICCV 2021.
[7] Beery et al. The iWildCam 2021 competition dataset. arXiv 2021.
