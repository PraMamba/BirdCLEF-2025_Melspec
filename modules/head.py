from math import sqrt
from torch import Tensor
import torch
import torch.nn.functional as F
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        input_channel: int,
        head_size: int,
        num_heads: int,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        hidden_dim = head_size * num_heads
        self.hidden_dim = hidden_dim

        self.head_size = head_size
        self.num_heads = num_heads

        self.key = nn.Linear(input_channel, hidden_dim)
        self.query = nn.Linear(input_channel, hidden_dim)

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.sqrt_head_size = sqrt(head_size)

        self.value = nn.Linear(input_channel, hidden_dim)

    def tranpose_for_scores(self, x: Tensor) -> Tensor:
        # [BS; T; H] -> [BS; T; K, M]
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3) # [BS; K; T; M]

    def get_key(self, x: Tensor) -> Tensor:
        return self.tranpose_for_scores(self.key(x))

    def get_query(self, x: Tensor) -> Tensor:
        return self.tranpose_for_scores(self.query(x))

    def get_value(self, x: Tensor) -> Tensor:
        return self.tranpose_for_scores(self.value(x))

    def forward(self, x: Tensor) -> Tensor:
        # [BS; L; C]
        key = self.get_key(x)
        query = self.get_query(x)
        value = self.get_value(x)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) 
        attention_scores /= self.sqrt_head_size
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.attention_dropout(attention_scores)
        x = torch.matmul(attention_scores, value) # [BS; K; N; M]
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.shape[:2] + (self.hidden_dim,))
        return x

class MultiHeadAttentionClassifier(MultiHeadSelfAttention):
    def __init__(
        self,
        input_channel: int,
        head_size: int = 32,
        num_heads: int = 24,
        attention_dropout: float = 0.3,
        num_classes: int = 264,
        dropout: float = 0.3
    ) -> None:
        super().__init__(input_channel, head_size, num_heads, attention_dropout)
        self.query = nn.Parameter(
            torch.empty(num_heads, num_classes, head_size),
            requires_grad=True
        )
        nn.init.normal_(self.query)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(num_classes, num_classes, kernel_size=self.hidden_dim, groups=num_classes)
        )

    def get_query(self, x: Tensor) -> Tensor:
        return self.query

    def forward(self, x: Tensor) -> Tensor:
        # x.shape == [BS; C; F; T] // output from backbone
        x = x.view(x.shape[:-2] + (-1,))
        # [BS; C; L]
        x = x.permute(0, 2, 1)
       # [BS; L; C]

        out = super().forward(x)
        return self.classifier(out)[:,:,0]