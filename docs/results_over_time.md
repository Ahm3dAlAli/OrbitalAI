# Results over time — every prediction dir, ranked

Scored against `OrbitSight_Dataset/Testing_sets` with the frozen evaluator (mAP = mean of per-sequence AP @ IoU 0.5).

| Rank | Prediction dir | mAP @ IoU 0.5 | seqs |
|---:|---|---:|---:|
| 1 | `router_ctta` ⭐ | **0.6750** | 4 |
| 2 | `router_best2` | **0.6600** | 4 |
| 3 | `test_ctx` | **0.6507** | 4 |
| 4 | `test_ctx_stack` | **0.6482** | 4 |
| 5 | `router_ctx` | **0.6295** | 4 |
| 6 | `test_ctx_tta` | **0.6013** | 3 |
| 7 | `router_final` | **0.5542** | 4 |
| 8 | `test_ens` | **0.5474** | 4 |
| 9 | `test_ens_stack` | **0.5474** | 4 |
| 10 | `test_xg` | **0.5225** | 4 |
| 11 | `test_xg_stack` | **0.5157** | 4 |
| 12 | `test_g192_stack` | **0.4951** | 4 |
| 13 | `test_g192` | **0.4896** | 4 |
| 14 | `router_stack` | **0.4543** | 4 |
| 15 | `router_best` | **0.4488** | 4 |
| 16 | `test_aug` | **0.4120** | 4 |
| 17 | `testing_router2` | **0.3983** | 4 |
| 18 | `testing_cnet_aug` | **0.3797** | 4 |
| 19 | `router_g192` | **0.3608** | 4 |
| 20 | `router_big` | **0.3478** | 3 |
| 21 | `router_aug` | **0.3255** | 4 |
| 22 | `testing_router` | **0.3152** | 4 |
| 23 | `testing_cnet` | **0.2890** | 4 |
| 24 | `testing` | **0.2491** | 4 |
| 25 | `testing_cnet_trk` | **0.1650** | 4 |
| 26 | `test_noaug` | **0.0856** | 4 |
| 27 | `test_noaug128` | **0.0856** | 4 |
| 28 | `test_ens_big_stack` | **0.0736** | 2 |
| 29 | `testing_evt2` | **0.0164** | 4 |
| 30 | `testing_pointnet` | **0.0162** | 4 |
| 31 | `testing_evt` | **0.0076** | 4 |
| 32 | `testing_snn` | **0.0002** | 4 |
