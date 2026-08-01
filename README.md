# sam3_DGX_GB10

SAM 3 分割模型在 DGX GB10 上的部署。

## 程式功能說明

| 腳本 | 功能 |
|---|---|
| `example.py` | SAM3 範例 |
| `sam3_slurm.sh` | Slurm 提交腳本 |

## 依賴環境

| 項目 | 需求 |
|---|---|
| Python | 3.x + SAM3, PyTorch |
| GPU | NVIDIA GB10 |

## AI Agent 操控指南

```
任務: 執行 SAM3 分割
步驟:
1. sbatch sam3_slurm.sh 提交
2. python example.py 本地執行
```
