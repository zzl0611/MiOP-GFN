# MiOP-GFN 项目结构说明

## 组织原则

项目将稳定入口、实验配置、训练数据、实现源码和外部预测服务分开：

1. `run.py` 是用户入口，负责环境检查、训练、采样、服务启动与测试；
2. `configs/` 固化实验参数，`datasets/` 明确展示训练数据与 motif 资源；
3. `src/gflownet/` 只承载 Python 包源码；
4. `predictors/` 集中保存奖励预测器的 HTTP 适配层与本地运行资源；
5. HemoPI2、ToxinPred3 仍可位于仓库外或远程服务器；
6. `scripts/` 放置非训练入口脚本，`tests/` 放置验证代码；
7. 所有运行产物集中写入不入库的 `outputs/`。

该分层将配置、数据、预测器、训练源码和运行输出明确分离，同时保留 `gflownet` Python 包内部稳定的导入关系。

## 完整目录

```text
MiOP-GFN/
├── run.py
├── README.md
├── environment-amp.yml
├── LICENSE.md
├── configs/
│   ├── README.md
│   └── train/
│       ├── mic-hemo.json
│       └── mic-hemo-tox.json
├── datasets/
│   ├── README.md
│   ├── offline/
│   │   ├── bert_ec_final_offline_mic_priority_top2000.fasta
│   │   ├── ...mic_hemo..._rewards.pt
│   │   └── ...mic_hemo_tox..._rewards.pt
│   └── motifs/
│       ├── bpe_motif_vocab_bert_ec_final_mic_hemo_notox_effect005.txt
│       ├── bpe_motif_vocab_bert_ec_final_mic_hemo_tox_effect005.txt
│       └── bpe_motif_vocab_final.txt
├── predictors/
│   ├── bert_mic/
│   │   ├── server.py
│   │   ├── regressor.py
│   │   └── artifacts/
│   │       ├── backbone/        # ProtBERT 配置、词表和基础权重
│   │       ├── best_model_state_dict.pkl
│   │       └── score_calibration.json
│   ├── hemolysis/
│   │   └── server.py
│   └── toxicity/
│       └── server.py
├── src/gflownet/
│   ├── tasks/                   # AMP 任务及奖励服务客户端
│   ├── algo/                    # OP-TB 等训练算法
│   ├── envs/                    # 序列和图环境
│   ├── models/                  # 序列和图模型
│   ├── data/                    # 采样迭代器与 replay buffer
│   └── utils/                   # 条件编码、指标与复现工具
├── scripts/
│   └── sample_trained_amp.py
├── tests/
│   ├── test_project_entry.py
│   ├── test_amp_reproducibility.py
│   └── test_reward_oracles.py
├── docs/
└── outputs/                     # 训练/采样时创建，Git 忽略
```

## 入口与调用关系

| 用户命令 | 实现位置 | 是否访问奖励服务 |
| --- | --- | --- |
| `python run.py doctor` | `run.py` | 否 |
| `python run.py serve mic` | `predictors/bert_mic/server.py` | 启动服务 |
| `python run.py serve hemo` | `predictors/hemolysis/server.py` | 启动服务 |
| `python run.py serve tox` | `predictors/toxicity/server.py` | 启动服务 |
| `python run.py check` | `run.py` | 是 |
| `python run.py train` | `src/gflownet/tasks/amp_moo.py` | 是 |
| `python run.py sample` | `scripts/sample_trained_amp.py` | 否 |
| `python run.py test` | `tests/` | 否 |

`run.py` 在子进程环境中自动加入 `src/` 到 `PYTHONPATH`，工作目录保持为仓库根目录。因此 JSON 中的 `./datasets/...` 与 `./outputs/...` 都以仓库根目录为基准。

## 奖励链路

```text
MiOP-GFN 生成序列
├── BERT MIC HTTP 服务 ────────> MIC reward
├── HemoPI2 HTTP 服务 ─────────> non-hemolysis reward
└── ToxinPred3 HTTP 服务（可选）> non-toxicity reward
```

`AMP_Evaluators` 不在训练和原始采样链路中。离线奖励缓存只能覆盖离线 FASTA；训练产生的新序列仍必须由在线服务打分。

## 底层入口

需要直接调试训练模块时，在仓库根目录执行：

```powershell
$env:PYTHONPATH = '.\src'
python -m gflownet.tasks.amp_moo --help
python .\scripts\sample_trained_amp.py --help
```

日常实验仍推荐 `run.py`，因为它统一处理路径、配置和输出目录检查。
