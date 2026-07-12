# End-to-end latency (evt_centernet_aug.pt, CPU)

Batch size = 1  (streaming latency uses batch=1)

| Sensor | Grid | Ctx | Windows |   Vox |   Fwd |   Dec | **Total** | <40ms? | win/s |
|--------|------|-----|---------|-------|-------|-------|-----------|--------|-------|
| EVK4   |  128 |   0 |     400 |  0.60 |  4.15 |  0.05 | **  4.80** | ✅ |    208 |
| DAVIS  |  128 |   0 |     400 |  0.31 |  4.29 |  0.05 | **  4.65** | ✅ |    215 |
| DVX    |  128 |   0 |     400 |  0.26 |  3.91 |  0.05 | **  4.21** | ✅ |    238 |
| DVX    |  128 |   0 |     400 |  0.25 |  3.89 |  0.04 | **  4.18** | ✅ |    239 |

Real-time target (<40 ms end-to-end per window): **REAL-TIME**
