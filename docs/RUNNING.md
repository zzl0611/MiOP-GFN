# 运行手册

以下命令均在 `MiOP-GFN` 根目录执行，示例使用 PowerShell。

## 1. 准备训练环境

```powershell
conda env create -f .\environment-amp.yml
conda activate amp-moo-gfn
python .\run.py doctor
```

推荐 Python 3.9。`doctor` 返回 `READY` 才表示当前解释器具备训练依赖与仓库内训练资源。它不会强制要求本机安装 HemoPI2 或 ToxinPred3，因为奖励服务可以部署在远端或其他 Conda 环境。

## 2. 准备奖励预测器

### 使用远程服务

```powershell
$env:MIC_BERT_URL = 'http://SERVER:5010/predict'
$env:HEMO_URL = 'http://SERVER:5006/predict'
$env:TOX_URL = 'http://SERVER:5008/predict'
```

`mic-hemo` 只需要前两个地址；`mic-hemo-tox` 需要三个地址。变量只对当前 PowerShell 会话生效。

### 使用本地服务

三个预测器可以使用不同环境。每个服务占用一个终端：

```powershell
# 终端 1：仓库内 predictors/bert_mic 中的 BERT MIC
python .\run.py serve mic --python 'D:\envs\mic\python.exe'

# 终端 2：先配置本地 HemoPI2 模型目录
$env:HEMO_MODEL_DIR = 'D:\models\hemopi2\Model'
python .\run.py serve hemo --python 'D:\envs\hemo\python.exe'

# 终端 3：三目标训练才需要
$env:TOXINPRED3_BIN = 'D:\tools\ToxinPred3\toxinpred3.exe'
python .\run.py serve tox --python 'D:\envs\tox\python.exe'
```

模型路径、端口和请求格式详见 [LOCAL_REWARD_SERVICES.md](LOCAL_REWARD_SERVICES.md)。

## 3. 检查奖励服务

检查命令会发送一条真实序列，而不是只判断端口是否打开：

```powershell
python .\run.py check --preset mic-hemo
python .\run.py check --preset mic-hemo-tox
```

所有目标均返回有限数值后才会成功。正式训练也会自动执行同类预检；不要在真实实验中设置 `ORACLE_ALLOW_FALLBACK=1`，否则服务故障可能被伪奖励掩盖。

## 4. 检查最终训练命令

`--dry-run` 只显示将执行的命令，不加载模型、不访问服务：

```powershell
python .\run.py train --preset mic-hemo --dry-run
python .\run.py train --preset mic-hemo-tox --dry-run
```

## 5. 冒烟测试与正式训练

先以独立输出目录运行少量步数：

```powershell
python .\run.py train `
  --preset mic-hemo `
  --steps 10 `
  --log-dir .\outputs\training\smoke-mic-hemo
```

然后启动完整训练：

```powershell
python .\run.py train --preset mic-hemo
```

该预设是与论文方法对齐的 BERT-MIC + Hemo 配置，不包含离线 STOP、在线 STOP 或 motif diversity 辅助损失。底层脚本的无参数默认值也已同步；仍推荐使用根入口，因为它会检查输出目录、启用无缓冲日志并把运行产物集中到 `outputs/`。

该历史流程的训练 `global_batch_size` 是 64。`offline_reward_batch_size=64` 是另一个参数，只控制首次计算离线奖励缓存时的分块大小。

三目标训练：

```powershell
python .\run.py train --preset mic-hemo-tox
```

默认输出位于根目录 `outputs/training/`。不要让两个进程共用同一个 `log_dir`；当前训练器会在该目录写入配置、TensorBoard 日志、Pareto 数据和 checkpoint。入口检测到目录已经存在时会停止，只有显式加入 `--overwrite` 才允许底层训练器替换原目录。

如需覆盖少数底层参数：

```powershell
python .\run.py train --preset mic-hemo `
  --log-dir .\outputs\training\mic-hemo-seed42 `
  -- --seed 42 --num_training_steps 20000
```

`--` 之后的参数会追加到预设命令末尾。长期保留的实验配置应复制一份 JSON 并命名，而不是只保存在终端历史中。

## 6. 采样

采样只恢复生成模型并输出原始候选，不调用奖励预测器：

```powershell
python .\run.py sample `
  --preset mic-hemo `
  --checkpoint .\outputs\training\paper_mic_hemo_10000_off020_rand015_seed0\model_state.pt `
  --output .\outputs\samples\mic-hemo-seed0.csv `
  --seed 0 `
  --num-samples 5000 `
  --batch-size 64 `
  --device cuda
```

新 checkpoint 会校验 motif 文件哈希、动作映射、门控与长度参数。旧 checkpoint 只有在明确接受无法完整验证复现信息时才使用 `--allow-legacy-checkpoint`。

## 7. 测试

```powershell
python .\run.py test
```

单元测试不调用奖励服务，主要验证奖励客户端、缓存命名和 checkpoint 复现清单。

## 常见问题

### `doctor` 报缺少 `torch_geometric` 等包

说明当前终端不是完整训练环境。激活 `amp-moo-gfn` 后重新执行；不要用能够运行 MIC 服务的环境代替训练环境。

### `check` 报 connection refused 或 timeout

确认对应服务已经启动、环境变量是完整的 `/predict` URL、端口可从训练机器访问，并检查服务器防火墙。

### Hemo 服务启动时报模型目录不存在

设置 `HEMO_MODEL_DIR`，其值应是包含 tokenizer 与模型文件的 Hugging Face 风格 `Model` 目录。

### Tox 服务启动时报找不到程序

设置 `TOXINPRED3_BIN` 为 ToxinPred3 可执行文件的绝对路径，或把它加入 `PATH`。

### 想跳过奖励服务直接训练

只要训练包含在线样本，就必须有真实奖励。已有离线缓存不能替代后续在线序列的打分。`--skip-oracle-preflight` 仅适合定位问题，不会让缺失的服务在在线训练阶段自动可用。
