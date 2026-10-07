# Reward predictors

本目录集中保存 MiOP-GFN 训练所需的奖励预测器代码与本地资源，避免与 `src/gflownet/models/` 中的生成模型实现混淆。

```text
predictors/
├── bert_mic/
│   ├── server.py
│   ├── regressor.py
│   └── artifacts/               # backbone、checkpoint、校准文件和模型卡
├── hemolysis/
│   └── server.py                # HemoPI2 HTTP 适配器
└── toxicity/
    └── server.py                # ToxinPred3 HTTP 适配器
```

BERT MIC 的运行资源随项目保存在 `bert_mic/artifacts/`。HemoPI2 与 ToxinPred3 的第三方模型或可执行程序不复制进仓库，分别通过 `HEMO_MODEL_DIR` 和 `TOXINPRED3_BIN` 指定。

统一启动方式：

```powershell
python .\run.py serve mic
python .\run.py serve hemo
python .\run.py serve tox
```
