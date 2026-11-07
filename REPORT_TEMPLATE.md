# Camera Traps in New Places: How Few Unlabeled Images Does a Species Classifier Need to Adapt to an Unseen Camera?
CSE-414 - Team members - Date

## Abstract (150 words)
Question, method (2 backbones x 3 settings, n=16..128), main finding in one sentence (including "did not help" if true), limitation.

## 1. Introduction & motivation  (reuse proposal sections 1-2; SDG 15)
## 2. Related work  (WILDS / WILDS 2.0, TENT, test-time BN, DINO)  -- do the fuller literature search the proposal promised
## 3. Problem statement, research question, hypotheses H1 / H2
## 4. Method
4.1 Data and subset (numbers from build_cache output: #images per split, #unseen cameras used)
4.2 Backbones and what is trained
4.3 Adaptation methods (BN update, TENT; ViT has no BN)
4.4 Protocol (adapt batch / held-out images / reset), oracle, seeds
4.5 Metrics
## 5. Results   (copy from results/summary.md and figures)
5.1 Seen vs unseen gap (fig_seen_unseen)            -> does the WILDS-style gap appear in our subset?
5.2 How many images are enough? (fig_curve, table)   -> H1
5.3 Backbone comparison (baseline gap + gain)        -> H2
5.4 Per-camera analysis (fig_per_camera, fig_conditions, bootstrap table)
5.5 Ablations: layers adapted, empty-frame share
## 6. Discussion  what worked / what did not (state failures plainly), why (empty frames, rare species, noisy BN stats at n=16)
## 7. Limitations  subset + low resolution, 1 training seed per backbone, macro F1 noise, no Bangladeshi cameras, results not comparable to the leaderboard
## 8. Conclusion
## References  [1] WILDS (Koh et al.)  [2] WILDS 2.0 / unlabeled data  [3] ...  [4] TENT (Wang et al.)  + DINO
## Appendix: demo screenshots, full summary.csv, hyper-parameters (tent_config_*.json)
