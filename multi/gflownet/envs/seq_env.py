import os
import math
import torch
from typing import List, Dict, Optional

class SeqBuildingEnvContext:
    """
    经典 OP-GFN 单层序列生成环境 (Baseline 对比组)。
    剥离了所有 Motif 机制，纯粹使用 21 维动作空间（20种氨基酸 + 1个Stop）。
    """
    def __init__(self, max_length: int = 50, num_cond_dim: int = 0,motif_vocab_path: Optional[str] = None,max_motif_actions: int = 2,min_length: int = 5,):
        self.max_length = max_length
        self.num_cond_dim = num_cond_dim
        
        # 1. 定义抗菌肽的氨基酸词表 (20种天然氨基酸)
        self.vocab = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                      'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        self.vocab_size = len(self.vocab)
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.vocab)}

        self.min_length = min_length
        self.max_motif_actions = max_motif_actions

        # Pad token 仍然用于 Transformer 输入，不是动作
        self.pad_idx = self.vocab_size + 1  # 21

        # 2. 动作空间：20种氨基酸 + 1个Stop动作(索引为20)
        # 读取 motif
        self.motifs = self._load_motifs(motif_vocab_path)
        self.num_motifs = len(self.motifs)

        # 动作编号规则：
        # 0 ~ 19：氨基酸动作
        # 20 ~ 20 + num_motifs - 1：motif 动作
        # 20 + num_motifs：STOP
        self.motif_action_start = self.vocab_size
        self.motif_action_end = self.vocab_size + self.num_motifs

        self.stop_action_idx = self.vocab_size + self.num_motifs
        self.num_actions = self.stop_action_idx + 1

        print(
            f"🔧 [MotifAction Env] 动作空间：20 aa + {self.num_motifs} motifs + STOP "
            f"= {self.num_actions}；STOP idx={self.stop_action_idx}；最多 motif={self.max_motif_actions}"
        )

        self.device = torch.device("cpu")
        print("🔧 已切换至纯 121 维motif OP-GFN 动作空间！")
    def _load_motifs(self, motif_vocab_path: Optional[str]) -> List[str]:
        """
        读取 motif 词表。
        每行一个 motif，例如 GLRKRLRK。
        """
        if motif_vocab_path is None or motif_vocab_path == "":
            print("⚠️ 未提供 motif_vocab_path，退化为 baseline：20 aa + STOP")
            return []

        if not os.path.exists(motif_vocab_path):
            raise FileNotFoundError(f"找不到 motif 词表文件: {motif_vocab_path}")

        motifs = []
        seen = set()

        with open(motif_vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                motif = line.strip().upper()
                if not motif:
                    continue

                # 只保留标准氨基酸组成的 motif
                if any(ch not in self.aa_to_idx for ch in motif):
                    print(f"⚠️ 跳过非法 motif: {motif}")
                    continue

                # 长度为 1 的 motif 和氨基酸动作重复，跳过
                if len(motif) <= 1:
                    continue

                if motif not in seen:
                    motifs.append(motif)
                    seen.add(motif)

        print(f"✅ 成功读取 motif 数量: {len(motifs)}")
        return motifs

    def get_initial_state(self) -> List[int]:
        """
        初始化状态：保留这个函数以防万一底层框架某些分支强制调用。
        对于经典的 OP-GFN，起点就是一个干净的空列表。
        """
        return []

    def seq_to_tensor(self, seq: List[int]) -> torch.Tensor:
        """把当前状态（整数列表）变成 Tensor"""
        return torch.tensor(seq, dtype=torch.long)

    def seq_str_to_idx(self, seq: str) -> List[int]:
        seq = seq.strip().upper()
        return [self.aa_to_idx[ch] for ch in seq if ch in self.aa_to_idx]

    def mol_to_graph(self, obj) -> List[int]:
        # 兼容旧框架名字：这里 graph 实际就是 token 序列
        if isinstance(obj, str):
            return self.seq_str_to_idx(obj)
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu().tolist()
        return [int(i) for i in obj if 0 <= int(i) < self.vocab_size]
    
    def graph_to_mol(self, state: List[int]) -> List[int]:
        """
        伪装术 1：拦截底层的化学转换。
        由于我们不做小分子，直接把生成的氨基酸数字列表原样返回给裁判。
        """
        return state

    def is_stop_action(self, action_idx: int) -> bool:
        return int(action_idx) == self.stop_action_idx


    def is_aa_action(self, action_idx: int) -> bool:
        return 0 <= int(action_idx) < self.vocab_size


    def is_motif_action(self, action_idx: int) -> bool:
        return self.motif_action_start <= int(action_idx) < self.motif_action_end
    
    def num_valid_parents(self, state: List[int], motif_count: int) -> int:
        """
        计算当前状态有多少个合法父状态。

        加入 motif 后，同一个序列可能有多种生成路径。
        例如 GLRKRLRK 可能来自：
        1. 逐氨基酸生成
        2. motif GLRKRLRK 一步生成

        所以反向概率不能再全部设为 0。
        """
        if len(state) == 0:
            return 0

        seq_str = self.idx_to_seq_str(state)
        motif_count = int(motif_count)

        # 当前序列非空时，至少可以删除最后一个氨基酸
        num_parents = 1

        # 只有当前 motif_count > 0，才可能说明最后一步是 motif 动作
        if motif_count > 0:
            for motif in self.motifs:
                if len(motif) <= len(seq_str) and seq_str.endswith(motif):
                    num_parents += 1

        return max(num_parents, 1)


    def backward_logprob_after_action(self, new_state: List[int], new_motif_count: int, action_idx: int) -> float:
        """
        计算某一步动作后的 logP_B。

        STOP 动作：
            反向唯一，所以 logP_B = 0

        氨基酸 / motif 动作：
            logP_B = -log(合法父状态数量)
        """
        if self.is_stop_action(action_idx):
            return 0.0

        n_parents = self.num_valid_parents(new_state, new_motif_count)
        return -math.log(float(n_parents))

    def action_to_string(self, action_idx: int) -> str:
        """
        把动作编号转成字符串。
        例如：
        0 -> A
        20 -> 第一个 motif
        stop_action_idx -> <STOP>
        """
        action_idx = int(action_idx)

        if self.is_aa_action(action_idx):
            return self.vocab[action_idx]

        if self.is_motif_action(action_idx):
            return self.motifs[action_idx - self.motif_action_start]

        if self.is_stop_action(action_idx):
            return "<STOP>"

        return "<INVALID>"


    def action_to_aa_indices(self, action_idx: int) -> List[int]:
        """
        把动作转成真正要 append 到状态里的氨基酸索引。
        """
        action_idx = int(action_idx)

        if self.is_stop_action(action_idx):
            return []

        if self.is_aa_action(action_idx):
            return [action_idx]

        if self.is_motif_action(action_idx):
            motif = self.motifs[action_idx - self.motif_action_start]
            return [self.aa_to_idx[ch] for ch in motif]

        raise ValueError(f"非法 action_idx: {action_idx}")


    def apply_action(self, state: List[int], motif_count: int, action_idx: int):
        """
        执行动作。
        氨基酸动作：append 1 个氨基酸
        motif 动作：append 多个氨基酸
        STOP：终止，不改变 state
        """
        action_idx = int(action_idx)
        motif_count = int(motif_count)

        if self.is_stop_action(action_idx):
            return list(state), motif_count, True

        append_tokens = self.action_to_aa_indices(action_idx)
        new_state = list(state) + append_tokens

        if self.is_motif_action(action_idx):
            motif_count += 1

        return new_state, motif_count, False

    def object_to_log_repr(self, state: List[int]) -> str:
        """
        伪装术 2：拦截底层的 SMILES 字符串日志记录。
        原样翻译成氨基酸字符串存进日志。
        """
        return self.idx_to_seq_str(state)

    def collate(self, states: List[List[int]], motif_counts: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        """
        核心函数：把多个长短不一的序列状态打包成一个 Batch，喂给 Transformer。
        """
        batch_size = len(states)
        if motif_counts is None:
            motif_counts = [0 for _ in states]

        motif_counts_tensor = torch.tensor(motif_counts, dtype=torch.long)
        max_len_in_batch = max([len(s) for s in states]) if states else 0
        
        # 使用 vocab_size + 1 (即21) 作为 Padding 的 token 索引
        pad_idx = self.pad_idx 
        
        # 处理初始空状态的情况
        if max_len_in_batch == 0:
            padded_seqs = torch.full((batch_size, 1), pad_idx, dtype=torch.long)
            lengths = torch.zeros(batch_size, dtype=torch.long)
        else:
            padded_seqs = torch.full((batch_size, max_len_in_batch), pad_idx, dtype=torch.long)
            lengths = torch.tensor([len(s) for s in states], dtype=torch.long)
            for i, s in enumerate(states):
                if len(s) > 0:
                    padded_seqs[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        
        # 生成 Mask：告诉 Transformer 哪些合法的动作可以做
        # action_mask 维度: [batch_size, 21] (1 代表允许，0 代表屏蔽)
        action_mask = torch.ones((batch_size, self.num_actions), dtype=torch.float32)
        
        # 先全部设为 0，再逐类打开合法动作
        action_mask = torch.zeros((batch_size, self.num_actions), dtype=torch.float32)

        for i, length in enumerate(lengths.tolist()):
            motif_count = int(motif_counts[i])

            # 1. 单氨基酸动作：只要没到最大长度就允许
            if length < self.max_length:
                action_mask[i, :self.vocab_size] = 1.0

            # 2. motif 动作：
            # 条件：还没超过最大 motif 数，并且 append 后不超长
            if motif_count < self.max_motif_actions:
                for m_idx, motif in enumerate(self.motifs):
                    action_idx = self.motif_action_start + m_idx
                    if length + len(motif) <= self.max_length:
                        action_mask[i, action_idx] = 1.0

            # 3. STOP：长度达到 min_length 才允许
            if length >= self.min_length:
                action_mask[i, self.stop_action_idx] = 1.0

            # 4. 如果已经达到最大长度，只能 STOP
            if length >= self.max_length:
                action_mask[i, :] = 0.0
                action_mask[i, self.stop_action_idx] = 1.0

            # 5. 防御：如果没有任何合法动作，强制允许 STOP
            if action_mask[i].sum() == 0:
                action_mask[i, self.stop_action_idx] = 1.0

        return {
            "x": padded_seqs,           # [Batch, Seq_len] 喂给模型的输入矩阵
            "lengths": lengths,         # [Batch] 记录每个序列的真实长度
            "motif_counts": motif_counts_tensor,
            "action_mask": action_mask  # [Batch, 21] 动作掩码
        }

    def idx_to_seq_str(self, seq_idx: List[int]) -> str:
        return "".join([self.vocab[int(i)] for i in seq_idx if 0 <= int(i) < self.vocab_size])