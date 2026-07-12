# Results over time — every prediction dir, ranked

Scored against `OrbitSight_Dataset/Testing_sets` with the frozen evaluator (mAP = mean of per-sequence AP @ IoU 0.5).

| Rank | Prediction dir | mAP @ IoU 0.5 | seqs |
|---:|---|---:|---:|
| 1 | `testing_router2` ⭐ | **0.3983** | 4 |
| 2 | `testing_cnet_aug` | **0.3797** | 4 |
| 3 | `testing_router` | **0.3152** | 4 |
| 4 | `testing_cnet` | **0.2890** | 4 |
| 5 | `testing` | **0.2491** | 4 |
| 6 | `testing_cnet_trk` | **0.1650** | 4 |
| 7 | `testing_evt2` | **0.0164** | 4 |
| 8 | `testing_pointnet` | **0.0162** | 4 |
| 9 | `testing_evt` | **0.0076** | 4 |
| 10 | `testing_snn` | **0.0002** | 4 |
