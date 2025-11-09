---
title: Transformer 理解
description: >-

author: Yu Fu (付彧)
date: 2025-11-09 19:23:00 +0800
categories: [Tech, Deep Learning Algorithm]
tags: [Transformer, LLM]
pin: true
math: true
---

<p align="center">
  <img src="/assets/img/posts/2025-11-09-Transformer/Transformer.png" alt="一个重要的图表" width="70%" >
</p>



#### Multi-head Attention
```python
import torch
import torch.nn as nn
import torch.nn.function as F

class MultiHeadAttention(nn.Module):
	def __init__(self, d_model, num_heads):
		assert d_model % num_heads == 0
		self.d_model = d_model
		self.num_heads = num_heads
		self.d_k = d_model // num_heads
		
		self.W_Q = nn.Linear(d_model, d_model)
		self.W_K = nn.Linear(d_model, d_model)
		slef.W_V = nn.Linear(d_model, d_model)
		
		self.W_O = nn.Linear(d_model, d_model)
		
	def forward(self, query, key, value, mask=None):
		batch_size = query.size(0)
		Q = self.W_Q(query)
		K = self.W_K(key)
		V = self.W_V(value)
		def split_heads(x):
			x = x.view(batch_size, -1, self.num_heads, self.d_k)
			return x.transpose(1, 2)
			
	Q = split_heads(Q)
	K = split_heads(K)
	V = split_heads(V)
	
	scores = torch.matmul(Q, K.transpose(-2, -1))/(self.d_k**0.5)
	if mask is not None:
		score = scores.masked_fill(mask == 0, -1e9)
	attention_weights = F.softmax(scores, dim = -1)
	attention_output = torch.matmul(attention_weights, V)
	
	attention_output = attention_output.transpose(1, 2).contiguous()
	attention_output = attention_output.view(batch_size, -1, self.d_model)
	
	output = self.W_O(attention_output)
	return output, attention_weight 
	
```
