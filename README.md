# TDScaler
This repository provides the TDScaler implementation, workload traces, and performance profiling data for the GPU instances used in the experiments. 

# GPU Instance Requirements
- Operating System: Ubuntu 22.04.5
- CUDA Version: 12.8
- vLLM Version: 0.26.0
- LLM Models: Llama-2-7B and Qwen3-14B

## Experiment groups

- `A.3x`: Llama2-7B, BurstGPT/ShareGPT, three instance types.
- `A.5x`: Llama2-7B, Azure, five instance types.
- `B.3x`: Qwen3-14B, BurstGPT/ShareGPT, three instance types.
- `SQL-A.3x`: Llama2-7B, BurstGPT/BIRD, three instance types.
- `SQL-A.5x`: Llama2-7B, Azure/BIRD, five instance types.
- `SQL-B.3x`: Qwen3-14B, BurstGPT/BIRD, three instance types.

## Training

```bash
python3 -m train.run \
  --group SQL-B.3x \
  --episodes x \
```

## Evaluation

```bash
python3 -m evaluate.run
```

## Plotting


```bash
python3 figures/Train_curve.py --workload original/bird
python3 figures/TDScaler_System_plt.py --workload original/bird
python3 figures/Scaling_Interval.py
```
