from torch import nn
from torch import softmax
import torch


class Expert(nn.Module):
    def __init__(self, emb_size):
        super(Expert, self).__init__()
        self.seq = nn.Sequential(nn.Linear(emb_size, emb_size),
                                 nn.ReLU(),
                                 nn.Linear(emb_size, emb_size))
        
    def forward(self, x):
        return self.seq(x)
    

class MoE(nn.Module):
    def __init__(self, emb_size, num_experts, top_k):
        super().__init__()
        self.experts = nn.ModuleList([Expert(emb_size) for _ in range(num_experts)])
        self.top_k = top_k
        self.gate = nn.Linear(emb_size, num_experts)

    def forward(self, x):
        # (batch, seq_len, emb_size)
        x_shape = x.shape

        # (batch * seq_len, emb_size)
        x = x.reshape(-1, x_shape[-1])

        # gates, (batch * seq_len, num_experts)
        gate_logits = self.gate(x)
        gate_probs = softmax(gate_logits, dim=-1)

        # get top_k experts
        top_k_gate_probs, top_k_gate_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_gate_probs = softmax(top_k_gate_probs, dim=-1)

        # (batch * seq_len, top_k)
        top_k_gate_probs = top_k_gate_probs.view(-1)
        top_k_gate_indices = top_k_gate_indices.view(-1)

        # (batch * seq_len, top_k, emb_size)
        x = x.unsqueeze(1).expand(x.size(0), self.top_k, x.size(-1)).reshape(-1, x.size(-1))
        y = torch.zeros_like(x)

        # per expert run
        for idx, expert in enumerate(self.experts):
            # (batch * seq_len, top_k, emb_size)
            x_expert = x[top_k_gate_indices == idx]
            y_expert = expert(x_expert)

            add_idx = (top_k_gate_indices == idx).nonzero().flatten()
            y = y.index_add(dim=0, index=add_idx, source=y_expert)

        # weighted sum
        top_k_gate_probs = top_k_gate_probs.view(-1, 1).expand(-1, x.size(-1))
        y = y * top_k_gate_probs
        y = y.view(-1, self.top_k, x.size(-1)).sum(dim=1)
        return y.view(x_shape)
    

class MNIST_MoE(nn.Module):
    def __init__(self, input_size, emb_size, num_experts, top_k):
        super().__init__()
        self.emb = nn.Linear(input_size, emb_size)
        self.moe = MoE(emb_size, num_experts, top_k)
        self.cls = nn.Linear(emb_size, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        y = self.cls(self.moe(self.emb(x)))
        return y
    

if __name__ == '__main__':
    model = MNIST_MoE(28 * 28, 128, 12, 3)
        