# Datasets

本目录只保存 MiOP-GFN 训练直接使用的数据资源。

## `offline/`

- `bert_ec_final_offline_mic_priority_top2000.fasta`：离线训练肽序列；
- `*_mic_hemo_*_rewards.pt`：上述 FASTA 在 MIC、Hemo 两个目标上的奖励缓存；
- `*_mic_hemo_tox_*_rewards.pt`：上述 FASTA 在 MIC、Hemo、Tox 三个目标上的奖励缓存。

缓存文件名由 FASTA 文件名、`--objectives` 的顺序和 `--reward_cache_tag` 共同确定。修改这些参数会选择另一个缓存；不存在时，程序需要调用在线奖励服务重新计算。

## `motifs/`

- `bpe_motif_vocab_bert_ec_final_mic_hemo_notox_effect005.txt`：双目标配置使用的 motif 词表；
- `bpe_motif_vocab_bert_ec_final_mic_hemo_tox_effect005.txt`：三目标配置使用的 motif 词表；
- `bpe_motif_vocab_final.txt`：保留的通用历史词表。

motif 的顺序会影响动作编号。训练和采样必须使用同一词表；新 checkpoint 中的复现清单会校验词表哈希与动作映射。

## 数据路径

配置文件中的路径相对于仓库根目录，例如：

```text
./datasets/offline/bert_ec_final_offline_mic_priority_top2000.fasta
./datasets/motifs/bpe_motif_vocab_bert_ec_final_mic_hemo_notox_effect005.txt
```
