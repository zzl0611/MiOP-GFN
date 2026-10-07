# 训练配置

`train/` 中的 JSON 保存可复用实验参数，由根目录 `run.py` 读取。

| 配置 | 目标 | 说明 |
| --- | --- | --- |
| `mic-hemo.json` | MIC、溶血性 | 与论文方法对齐，不使用 STOP 或 motif diversity 辅助损失 |
| `mic-hemo-tox.json` | MIC、溶血性、毒性 | 与仓库保留的三目标 motif 词表和奖励缓存匹配 |

执行方式：

```powershell
python .\run.py train --preset mic-hemo
python .\run.py train --preset mic-hemo-tox
```

配置内的训练资源路径相对于仓库根目录，不是相对于 JSON 文件。离线数据位于 `datasets/offline/`，motif 词表位于 `datasets/motifs/`，默认训练产物集中写入 `outputs/training/`。

命令行适合临时覆盖：

```powershell
python .\run.py train --preset mic-hemo --steps 10 --log-dir .\outputs\training\smoke
python .\run.py train --preset mic-hemo -- --seed 1
```

若一组参数需要用于正式实验，复制已有 JSON、改成有意义的配置名并记录独立 `log_dir`。添加新配置后还需在 `run.py` 的 `--preset` 选项中注册。
