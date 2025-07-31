import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
import numpy as np
from tqdm.auto import tqdm
from contextlib import nullcontext
import os
import tiktoken
import time
import json
from typing import Optional, Tuple, Dict, List

# ===================== CONFIGURATION =====================
@dataclass
class GPTConfig:
    block_size: int = 512
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 1024
    dropout: float = 0.1
    bias: bool = True
    # MoE specific
    moe_enabled: bool = False
    n_experts: int = 64
    n_shared_experts: int = 2
    expert_capacity_factor: float = 0.25  # Each expert has 1/4 capacity
    top_k_experts: int = 6
    load_balancing_gamma: float = 0.01
    # MLA specific
    compression_ratio: float = 0.5  # r/d ratio
    
    def get_active_params(self):
        """Calculate active parameters per forward pass"""
        if not self.moe_enabled:
            return self.get_total_params()
        
        # Attention params (same for both)
        attn_params = self.n_layer * (
            3 * self.n_embd * self.n_embd +  # Q, K, V projections
            self.n_embd * self.n_embd  # Output projection
        )
        
        # MoE FFN params (only active ones)
        active_experts = self.n_shared_experts + self.top_k_experts
        expert_dim = int(4 * self.n_embd * self.expert_capacity_factor)
        ffn_params = self.n_layer * active_experts * (
            self.n_embd * expert_dim +  # c_fc
            expert_dim * self.n_embd    # c_proj
        )
        
        # Other params
        other_params = (
            self.vocab_size * self.n_embd +  # wte
            self.block_size * self.n_embd +   # wpe
            2 * self.n_layer * self.n_embd    # layernorms
        )
        
        return attn_params + ffn_params + other_params
    
    def get_total_params(self):
        """Calculate total parameters"""
        # Similar to above but with all experts counted
        pass  # Implementation similar to get_active_params but with all experts

# Model configurations from paper
MODEL_CONFIGS = {
    'xs': GPTConfig(n_layer=6, n_embd=256, n_head=8),
    's': GPTConfig(n_layer=6, n_embd=512, n_head=8),
    'm': GPTConfig(n_layer=9, n_embd=512, n_head=8),
    'l': GPTConfig(n_layer=12, n_embd=768, n_head=12),
    'xl': GPTConfig(n_layer=12, n_embd=1024, n_head=16),
}

# ===================== UTILITY FUNCTIONS =====================
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, cos, sin):
    rope_dim = min(x.shape[-1], cos.shape[-1] * 2)
    x_rope = x[..., :rope_dim]
    x_pass = x[..., rope_dim:] if x.shape[-1] > rope_dim else None
    
    x_rotated = (x_rope * cos[..., :rope_dim//2].repeat_interleave(2, dim=-1)) + \
                (rotate_half(x_rope) * sin[..., :rope_dim//2].repeat_interleave(2, dim=-1))
    
    if x_pass is not None:
        x_rotated = torch.cat([x_rotated, x_pass], dim=-1)
    
    return x_rotated

# ===================== MOE COMPONENTS =====================
class TopKRouter(nn.Module):
    """Gradient-free load balanced router"""
    def __init__(self, dim, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.register_buffer('expert_bias', torch.zeros(num_experts))
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('total_counts', torch.tensor(0.0))
        
    def forward(self, x):
        """
        x: [batch, seq_len, dim]
        Returns: (indices, weights) 
            indices: [batch*seq_len, top_k]
            weights: [batch*seq_len, top_k]
        """
        batch, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)
        
        # Compute logits with bias (bias doesn't affect gradients)
        logits = self.gate(x_flat) + self.expert_bias.detach()
        
        # Top-k selection
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        
        # Update counts for load balancing (no gradient flow)
        if self.training:
            with torch.no_grad():
                # Count tokens per expert
                expert_counts = torch.zeros_like(self.expert_counts)
                ones = torch.ones_like(top_k_indices, dtype=expert_counts.dtype)
                expert_counts.scatter_add_(0, top_k_indices.flatten(), ones.flatten())
                
                # Update running counts
                self.expert_counts += expert_counts
                self.total_counts += batch * seq_len * self.top_k
        
        return top_k_indices, top_k_weights

    def update_bias(self, gamma):
        """Update bias for load balancing (gradient-free)"""
        if self.total_counts > 0:
            # Calculate load per expert
            load_per_expert = self.expert_counts / self.total_counts
            target_load = 1.0 / self.num_experts
            
            # Update bias to discourage overused experts
            self.expert_bias -= gamma * (load_per_expert - target_load)
            
            # Reset counts
            self.expert_counts.zero_()
            self.total_counts.zero_()

class Expert(nn.Module):
    """Single FFN expert with reduced capacity"""
    def __init__(self, dim_in, dim_hidden, dropout=0.0):
        super().__init__()
        self.c_fc = nn.Linear(dim_in, dim_hidden)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(dim_hidden, dim_in)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class MoELayer(nn.Module):
    """Fine-grained MoE layer with shared experts"""
    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        self.n_shared = config.n_shared_experts
        self.n_routed = config.n_experts - config.n_shared_experts
        self.top_k = config.top_k_experts
        
        # Expert dimension (1/4 of standard FFN)
        expert_dim = int(4 * config.n_embd * config.expert_capacity_factor)
        
        # Shared experts (always active)
        self.shared_experts = nn.ModuleList([
            Expert(config.n_embd, expert_dim, config.dropout)
            for _ in range(self.n_shared)
        ])
        
        # Routed experts
        self.routed_experts = nn.ModuleList([
            Expert(config.n_embd, expert_dim, config.dropout)
            for _ in range(self.n_routed)
        ])
        
        # Router
        self.router = TopKRouter(config.n_embd, self.n_routed, self.top_k)
        
    def forward(self, x):
        """
        x: [batch, seq_len, dim]
        """
        batch, seq_len, dim = x.shape
        
        # Process shared experts (always active)
        shared_output = sum(expert(x) for expert in self.shared_experts) / self.n_shared
        
        # Route to top-k experts
        x_flat = x.view(-1, dim)
        indices, weights = self.router(x)  # [batch*seq_len, top_k]
        
        # Process routed experts
        routed_output = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            # Get expert indices and weights for this position
            expert_idx = indices[:, i]
            expert_weight = weights[:, i].unsqueeze(-1)
            
            # Group by expert
            for j in range(self.n_routed):
                mask = (expert_idx == j)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_out = self.routed_experts[j](expert_input)
                    routed_output[mask] += expert_weight[mask] * expert_out
        
        routed_output = routed_output.view(batch, seq_len, dim)
        
        # Combine shared and routed outputs
        return shared_output + routed_output

# ===================== ATTENTION COMPONENTS =====================
class MLA(nn.Module):
    """Multi-head Latent Attention with RoPE"""
    def __init__(self, config):
        super().__init__()
        self.d_model = config.n_embd
        self.n_heads = config.n_head
        self.dh = config.n_embd // config.n_head
        
        # Compression ratio
        self.compression_ratio = config.compression_ratio
        self.latent_dim = int(config.n_embd * self.compression_ratio)
        
        # Q projection (no compression)
        self.W_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # KV compression (shared across heads)
        self.W_kv_compress = nn.Linear(config.n_embd, 2 * self.latent_dim, bias=config.bias)
        
        # Head-specific KV reconstruction
        self.W_k_heads = nn.Parameter(torch.randn(config.n_head, self.latent_dim, self.dh))
        self.W_v_heads = nn.Parameter(torch.randn(config.n_head, self.latent_dim, self.dh))
        
        # Output projection
        self.W_o = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # RoPE setup
        self.max_seq_len = config.block_size
        self.rope_theta = 10000.0
        self._init_rope()
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
    def _init_rope(self):
        rope_dim_half = self.dh // 4
        freqs = 1.0 / (self.rope_theta ** (torch.arange(0, rope_dim_half).float() / rope_dim_half))
        emb = torch.outer(torch.arange(self.max_seq_len).float(), freqs)
        cos_cached = emb.cos()[None, None, :, :]
        sin_cached = emb.sin()[None, None, :, :]
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)
        
    def forward(self, x, kv_cache=None):
        B, S, D = x.size()
        
        # Q projection
        Q = self.W_q(x).view(B, S, self.n_heads, self.dh).transpose(1, 2)
        
        # KV compression
        kv_compressed = self.W_kv_compress(x)
        k_compressed, v_compressed = kv_compressed.chunk(2, dim=-1)
        
        # Reconstruct K, V per head
        K = torch.einsum('bsl,hld->bhsl', k_compressed, self.W_k_heads)
        V = torch.einsum('bsl,hld->bhsl', v_compressed, self.W_v_heads)
        
        # Apply RoPE
        cos = self.cos_cached[:, :, :S, :]
        sin = self.sin_cached[:, :, :S, :]
        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)
        
        # Attention
        attn = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True
        )
        
        # Reshape and project
        attn = attn.transpose(1, 2).contiguous().view(B, S, D)
        out = self.resid_dropout(self.W_o(attn))
        
        # Return compressed KV for caching during inference
        return out, (k_compressed, v_compressed)

# ===================== MODEL COMPONENTS =====================
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        self.attn = MLA(config)
        self.ln2 = LayerNorm(config.n_embd, config.bias)
        
        # Use MoE or standard FFN
        if config.moe_enabled:
            self.mlp = MoELayer(config)
        else:
            self.mlp = MLP(config)
            
    def forward(self, x, kv_cache=None):
        attn_out, new_kv = self.attn(self.ln1(x), kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_kv

class MLP(nn.Module):
    """Standard FFN for non-MoE models"""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x, _ = block(x)
            
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None
            
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
    
    def update_router_bias(self, gamma):
        """Update all router biases for load balancing"""
        if self.config.moe_enabled:
            for block in self.transformer.h:
                if hasattr(block.mlp, 'router'):
                    block.mlp.router.update_bias(gamma)

# ===================== TRAINING UTILITIES =====================
def get_data():
    """Load TinyStories dataset"""
    if not os.path.exists("train.bin"):
        print("Processing dataset...")
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories")
        enc = tiktoken.get_encoding("gpt2")
        
        def process(example):
            ids = enc.encode_ordinary(example['text'])
            return {'ids': ids, 'len': len(ids)}
        
        tokenized = ds.map(process, remove_columns=['text'], num_proc=8)
        
        for split, dset in tokenized.items():
            arr_len = np.sum(dset['len'], dtype=np.uint64)
            filename = f'{split}.bin'
            dtype = np.uint16
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
            
            idx = 0
            for batch_idx in tqdm(range(1024), desc=f'writing {filename}'):
                batch = dset.shard(num_shards=1024, index=batch_idx, contiguous=True).with_format('numpy')
                arr_batch = np.concatenate(batch['ids'])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

def get_batch(split, block_size, batch_size, device):
    data = np.memmap(f'{split}.bin', dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, eval_iters, block_size, batch_size, device, ctx):
    out = {}
    model.eval()
    for split in ['train', 'validation']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, block_size, batch_size, device)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ===================== EXPERIMENT UTILITIES =====================
def create_model_for_experiment(base_config, experiment_type, model_type):
    """Create model configurations for different experiments"""
    config = dataclass.replace(base_config)
    
    if experiment_type == "parameter_matched":
        if "moe" in model_type.lower():
            # Reduce hidden dim to account for extra expert params
            scale_factor = math.sqrt(8 / 64)  # sqrt(active_experts / total_experts)
            config.n_embd = int(config.n_embd * scale_factor)
            config.moe_enabled = True
            
    elif experiment_type == "flop_matched":
        if "moe" in model_type.lower():
            # Increase hidden dim for MoE (same FLOPs due to sparsity)
            scale_factor = math.sqrt(64 / 8)
            config.n_embd = int(config.n_embd * scale_factor)
            config.moe_enabled = True
            
    # Set compression ratio for MLA variants
    if "mla" in model_type.lower():
        config.compression_ratio = 0.5
    else:
        config.compression_ratio = 1.0
        
    return config

def measure_inference_metrics(model, device, batch_size=16, seq_len=512):
    """Measure latency and memory usage"""
    model.eval()
    
    # Warmup
    dummy_input = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Measure latency
    torch.cuda.synchronize()
    start_time = time.time()
    
    num_runs = 100
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(dummy_input)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / num_runs
    
    # Measure memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        _ = model(dummy_input)
        
    memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    return {
        'latency_ms': avg_latency * 1000,
        'memory_mb': memory_used,
        'throughput_tokens_per_sec': (batch_size * seq_len) / avg_latency
    }

def evaluate_generation_quality(model, prompts, device, max_length=100):
    """Generate text samples for quality evaluation"""
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    
    generations = []
    for prompt in prompts:
        prompt_ids = torch.tensor(enc.encode_ordinary(prompt)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(prompt_ids, max_new_tokens=max_length, temperature=0.8, top_k=40)
            
        generated_text = enc.decode(generated_ids[0].tolist())
        generations.append({
            'prompt': prompt,
            'completion': generated_text[len(prompt):]
        })
    
    return generations

# ===================== MAIN TRAINING FUNCTION =====================
def train_model(config_name, experiment_type, model_type, device='cuda'):
    """Train a model with specified configuration"""
    # Setup
    base_config = MODEL_CONFIGS[config_name]
    config = create_model_for_experiment(base_config, experiment_type, model_type)
    
    # Training hyperparameters
    learning_rate = 3e-4
    max_iters = 50000
    warmup_iters = 5000
    eval_interval = 500
    eval_iters = 25
    batch_size = 128
    block_size = config.block_size
    
    # Model
    model = GPT(config).to(device)
    print(f"Model type: {model_type}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Active parameters: {config.get_active_params():,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    
    # Scheduler
    def get_lr(it):
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        if it > max_iters:
            return learning_rate * 0.1
        decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return learning_rate * 0.1 + coeff * (learning_rate - learning_rate * 0.1)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
    
    for iter_num in tqdm(range(max_iters)):
        # Set learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = get_lr(iter_num)
        
        # Evaluate
        if iter_num % eval_interval == 0:
            losses = estimate_loss(model, eval_iters, block_size, batch_size, device, ctx)
            print(f"\nStep {iter_num}: train loss {losses['train']:.4f}, val loss {losses['validation']:.4f}")
            train_losses.append(losses['train'])
            val_losses.append(losses['validation'])
        
        # Get batch
        X, Y = get_batch('train', block_size, batch_size, device)
        
        # Forward pass
        with ctx:
            logits, loss = model(X, Y)
            
        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # Update router bias for load balancing
        if config.moe_enabled and iter_num % 100 == 0:
            model.update_router_bias(config.load_balancing_gamma)
    
    # Final evaluation
    final_losses = estimate_loss(model, 100, block_size, batch_size, device, ctx)
    inference_metrics = measure_inference_metrics(model, device)
    
    results = {
        'config': config.__dict__,
        'final_train_loss': final_losses['train'],
        'final_val_loss': final_losses['validation'],
        'train_history': train_losses,
        'val_history': val_losses,
        'inference_metrics': inference_metrics
    }
    
    return model, results

# ===================== RUN ALL EXPERIMENTS =====================
def run_all_experiments():
    """Run all experiments from the paper"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Ensure dataset is ready
    get_data()
    
    experiments = {
        'parameter_matched': {
            'MHA': 'mha',
            'MLA': 'mla',
            'MLA-RoPE': 'mla',  # Our base model already has RoPE
            'MoE-MHA': 'moe-mha',
            'MoE-MLA': 'moe-mla',
            'MoE-MLA-RoPE': 'moe-mla'
        },
        'flop_matched': {
            'MHA': 'mha',
            'MLA-RoPE': 'mla',
            'MoE-MHA': 'moe-mha',
            'MoE-MLA-RoPE': 'moe-mla'
        }
    }
    
    # Ablation: compression ratios
    compression_ratios = [1.0, 0.5, 0.25, 0.125]
    
    # Ablation: expert granularity
    expert_counts = [8, 16, 64, 128]
    
    all_results = {}
    
    # Main experiments
    for exp_type, models in experiments.items():
        print(f"\n{'='*50}")
        print(f"Running {exp_type} experiments")
        print(f"{'='*50}")
        
        for model_name, model_type in models.items():
            print(f"\nTraining {model_name}...")
            model, results = train_model('m', exp_type, model_type, device)
            all_results[f"{exp_type}_{model_name}"] = results
            
            # Save model
            torch.save(model.state_dict(), f"{exp_type}_{model_name}.pt")
    
    # Compression ratio ablation
    print(f"\n{'='*50}")
    print("Running compression ratio ablation")
    print(f"{'='*50}")
    
    for ratio in compression_ratios:
        config = MODEL_CONFIGS['m']
        config.compression_ratio = ratio
        config.moe_enabled = True
        
        print(f"\nTraining with compression ratio {ratio}...")
        model = GPT(config).to(device)
        # ... training code similar to above
    
    # Save all results
    with open('experiment_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results

# ===================== EVALUATION SCRIPT =====================
def evaluate_saved_models():
    """Evaluate all saved models and generate tables"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test prompts for generation
    test_prompts = [
        "Once upon a time, there was a little rabbit who lived in",
        "The brave knight fought the dragon",
        "In a magical forest, a young girl discovered",
        "A friendly robot wanted to help",
        "The wise old owl told the animals"
    ]
    
    # Load and evaluate each model
    for exp_type in ['parameter_matched', 'flop_matched']:
        print(f"\nEvaluating {exp_type} models:")
        
        for model_name in ['MHA', 'MLA-RoPE', 'MoE-MHA', 'MoE-MLA-RoPE']:
            model_path = f"{exp_type}_{model_name}.pt"
            if os.path.exists(model_path):
                # Load model
                config = MODEL_CONFIGS['m']  # Adjust as needed
                model = GPT(config).to(device)
                model.load_state_dict(torch.load(model_path))
                model.eval()
                
                # Evaluate
                val_loss = estimate_loss(model, 100, config.block_size, 128, device, nullcontext())
                metrics = measure_inference_metrics(model, device)
                generations = evaluate_generation_quality(model, test_prompts, device)
                
                print(f"\n{model_name}:")
                print(f"  Validation PPL: {math.exp(val_loss['validation']):.3f}")
                print(f"  Latency: {metrics['latency_ms']:.2f} ms")
                print(f"  Memory: {metrics['memory_mb']:.0f} MB")
                print(f"  Throughput: {metrics['throughput_tokens_per_sec']:.0f} tokens/sec")

if __name__ == "__main__":
    # Run all experiments
    print("Starting experiments...")
    results = run_all_experiments()
    
    # Evaluate models
    print("\nEvaluating models...")
    evaluate_saved_models()
    
    print("\nAll experiments completed!")
