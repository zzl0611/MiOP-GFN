# BERT MIC service

本目录集中保存 MiOP-GFN 训练使用的 E. coli BERT MIC 预测器：

- `server.py`：Flask 服务和输入、输出协议；
- `regressor.py`：与微调 checkpoint 匹配的 `REG` 网络定义；
- `artifacts/`：ProtBERT backbone、E. coli MIC 微调 checkpoint、校准参数和模型卡。

服务默认读取同目录的 `artifacts/`。从仓库根目录启动：

```powershell
python .\run.py serve mic
```

默认监听 `127.0.0.1:5010`，接口为：

- `GET /health`：服务与模型信息；
- `POST /predict`：请求 `{"seqs": ["KIVRIFFKILKF"]}`，训练读取响应中的 `mic_score`。

可用环境变量：`MIC_BERT_HOST`、`MIC_BERT_PORT`、`BERT_MODEL_PATH`、`BERT_TOKENIZER_PATH`、`BERT_EC_CKPT`、`BERT_EC_CALIB`、`BERT_EC_MAX_LEN` 和 `BERT_EC_BATCH_SIZE`。
