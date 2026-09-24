import torch
import torch.nn as nn

class SeqTransformerGFN(nn.Module):
    """
    经典 OP-GFN 序列生成模型 (Baseline 对比组)。
    单层 Transformer 架构，直接输出 21 维动作分数。
    完美兼容双目标/三目标的动态条件特征 (cond_info) 注入。
    """
    def __init__(self, vocab_size=20, max_len=50, d_model=256, nhead=8, num_layers=3, num_cond_dim=64,num_actions=None,pad_idx=None,max_motif_actions=2,use_motif_gate=False, motif_use_gate_strength=1.0, motif_select_gate_strength=1.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_actions = num_actions if num_actions is not None else vocab_size + 1
        self.pad_idx = pad_idx if pad_idx is not None else vocab_size + 1
        self.max_motif_actions = max_motif_actions

        self.motif_action_start = self.vocab_size
        self.motif_action_end = self.num_actions - 1  # 最后一个动作是 STOP
        self.num_motifs = max(0, self.motif_action_end - self.motif_action_start)

        self.use_motif_gate = use_motif_gate and self.num_motifs > 0
        self.motif_use_gate_strength = motif_use_gate_strength
        self.motif_select_gate_strength = motif_select_gate_strength
        
        # 1. 词嵌入层 (Embedding)
        # 容量：20个氨基酸 + 1个Stop + 1个Pad = 22。为了防止越界，留足 32 的空间
        self.embedding = nn.Embedding(32, d_model)
        
        # 2. 位置编码层 (Positional Encoding)
        # 加 10 的缓冲，防止序列溢出
        self.pos_embedding = nn.Embedding(self.max_len + 10, d_model)
        
        # 3. 核心 Transformer 编码器 (采用较深的网络结构以保证公平对比)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            batch_first=True, 
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. 多目标偏好处理网络
        # 动态接收外部传入的 num_cond_dim (双目标通常是64，三目标通常是80)
        self.cond_mlp = nn.Sequential(
            nn.Linear(num_cond_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # 额外输入：当前轨迹已经使用了几个 motif
        # 因为最多允许 max_motif_actions 个 motif，所以取值是 0, 1, 2
        self.motif_count_emb = nn.Embedding(max_motif_actions + 1, d_model)

        gate_dim = d_model * 3

        if self.use_motif_gate:
            # 门控 1：判断当前状态整体上适不适合使用 motif
            self.motif_use_head = nn.Sequential(
                nn.Linear(gate_dim, d_model),
                nn.SiLU(),
                nn.Linear(d_model, 1),
            )

            # 门控 2：判断每个 motif 当前适不适合被选
            self.motif_select_head = nn.Sequential(
                nn.Linear(gate_dim, d_model),
                nn.SiLU(),
                nn.Linear(d_model, self.num_motifs),
            )
        else:
            self.motif_use_head = None
            self.motif_select_head = None

        # 5. 输出预测头 (Linear Head)
        # 接收 Transformer 提取的序列特征 (d_model) + 多目标偏好特征 (d_model) = d_model * 2
        self.action_head = nn.Linear(d_model * 3, self.num_actions)
        
        print(
            f"🧠 [MotifAction Model] 单层 Transformer 已就位！"
            f"(cond_dim={num_cond_dim}, num_actions={self.num_actions}, max_motif_actions={max_motif_actions})"
        )

    def forward(self, batch):
        """
        前向传播
        :param batch: 由 seq_env.collate() 传过来的字典
        :return: [Batch, 21] 形状的动作分数 (Logits)
        """
        x = batch["x"]                  # [B, L]
        lengths = batch["lengths"]      # [B]
        action_mask = batch["action_mask"] # [B, 21]

        # 新增：读取 motif_counts
        if isinstance(batch, dict):
            motif_counts = batch.get("motif_counts", None)
        else:
            motif_counts = getattr(batch, "motif_counts", None)
        
        B, L = x.shape

        if motif_counts is None:
            motif_counts = torch.zeros(B, dtype=torch.long, device=x.device)
        else:
            motif_counts = motif_counts.to(x.device).long()

        # 防止越界，只允许 0, 1, 2
        motif_counts = motif_counts.clamp(min=0, max=self.max_motif_actions)

        
        # 安全获取多目标条件特征
        ci = None
        if isinstance(batch, dict):
            ci = batch.get("cond_info")
        else:
            ci = getattr(batch, "cond_info", None)
            
        # --- 特征提取阶段 ---
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        x_emb = self.embedding(x) + self.pos_embedding(positions)
        
        # 告诉 Transformer 忽略 Padding 的部分
        padding_mask = (x == self.pad_idx)
        
        # 🚨 终极防御：修复 PyTorch Transformer 全 Pad 崩溃输出 NaN 的 Bug
        all_masked = padding_mask.all(dim=1)
        padding_mask[all_masked, 0] = False
        
        # 过 Transformer 提取特征 -> [B, L, d_model]
        out = self.transformer(x_emb, src_key_padding_mask=padding_mask)
        
        # --- 特征聚合阶段 (Mean Pooling) ---
        mask_float = (~padding_mask).float().unsqueeze(-1)
        sum_out = (out * mask_float).sum(dim=1)
        valid_lens = lengths.float().unsqueeze(-1).clamp(min=1.0) 
        state_repr = sum_out / valid_lens  # [B, d_model]

        # --- 多目标特征融合阶段 ---
        if ci is not None:
            cond_repr = self.cond_mlp(ci)  # [B, d_model]
        else:
            cond_repr = torch.zeros_like(state_repr)

        # 新增：motif_count 表示
        motif_repr = self.motif_count_emb(motif_counts)
        state_repr = torch.cat([state_repr, cond_repr, motif_repr], dim=-1)

        logits = self.action_head(state_repr)

        if self.use_motif_gate and self.num_motifs > 0:
            motif_use_score = self.motif_use_head(state_repr)
            motif_select_scores = self.motif_select_head(state_repr)

            motif_slice = slice(self.motif_action_start, self.motif_action_end)

            logits[:, motif_slice] = (
                logits[:, motif_slice]
                + self.motif_use_gate_strength * motif_use_score
                + self.motif_select_gate_strength * motif_select_scores
            )

            self.last_motif_use_score = motif_use_score.detach()
            self.last_motif_select_scores = motif_select_scores.detach()

        logits = logits.masked_fill(action_mask == 0.0, -1e9)

        return logits