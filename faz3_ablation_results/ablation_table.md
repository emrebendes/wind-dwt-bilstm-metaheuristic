# Ablation Karsilastirmasi - GWO

| Senaryo | Mean RMSE | Std | $\Delta$ % | Mann-Whitney $p$ | Cohen's $d$ | Cliff's $\delta$ | Effect |
|---|---|---|---|---|---|---|---|
| **GWO Full** | 0.037117 | 0.002026 | -- | -- | -- | -- | baseline |
| GWO No DWT | 0.055984 | 0.000229 | +50.83 | <0.0001 \* | 13.09 | 1.00 | large |
| GWO No Bidir. | 0.036694 | 0.001345 | -1.14 | 0.6740 | -0.25 | -0.07 | small |

**Friedman test** (k=3 senaryo, n=30 koSu): 
$\chi^2 = 45.07$, $p = 1.64e-10$

**Mean ranks (lower = better):**
- Full: 1.53
- No DWT: 3.00
- No Bidir.: 1.47
