## 1. Analysis

## 2. Editing Code
Done.
```shell
$env:PYTHONPATH = "C:\Users\soono\OneDrive\Ongoing\CS285-Fall2023-Homework\hw1"
```
## 3. Behavior Cloning

### Ant
```shell
python hw1/cs285/scripts/run_hw1.py `
    --env_name Ant-v4 --exp_name bc_ant `
    --ep_len 1000 --eval_batch_size 5000 `
    --video_log_freq -1
```
|                 Metric                 |         Value         |
|:--------------------------------------:|:---------------------:|
|           Eval_AverageReturn           |  1003.3518676757812   |
|             Eval_StdReturn             |   274.9608459472656   |
|             Eval_MaxReturn             |   1492.29638671875    |
|             Eval_MinReturn             |    738.24658203125    |
|           Eval_AverageEpLen            |        1000.0         |
|          Train_AverageReturn           |   4681.891673935816   |
|            Train_StdReturn             |   30.70862278765526   |
|            Train_MaxReturn             |   4712.600296723471   |
|            Train_MinReturn             |   4651.18305114816    |
|           Train_AverageEpLen           |        1000.0         |
|             Training Loss              | 0.034366387873888016  |
|          Train_EnvstepsSoFar           |           0           |
|             TimeSinceStart             |  11.873703241348267   |
|  Initial_DataCollection_AverageReturn  |   4681.891673935816   |


## 4. DAgger

### Ant
```shell
python hw1/cs285/scripts/run_hw1.py `
  --env_name Ant-v4 --exp_name dagger_ant --do_dagger `
  --n_iter 10 --ep_len 1000 --eval_batch_size 5000 `
  --video_log_freq -1
```
```shell
tensorboard --logdir C:\Users\soono\OneDrive\Ongoing\CS285-Fall2023-Homework\hw1\run_logs
```
