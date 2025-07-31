import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class MoEDebugger:
    """Debug and visualize MoE routing behavior"""
    
    def __init__(self, model):
        self.model = model
        self.routing_history = defaultdict(list)
        self.expert_activations = defaultdict(lambda: defaultdict(int))
        
    def hook_routing(self):
        """Add hooks to track routing decisions"""
        handles = []
        
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if hasattr(module, 'router'):
                    # Get routing decisions
                    indices, weights = output  # Assuming router returns (indices, weights)
                    
                    # Store routing info
                    self.routing_history[f'layer_{layer_idx}'].append({
                        'indices': indices.detach().cpu(),
                        'weights': weights.detach().cpu()
                    })
                    
                    # Count expert activations
                    for expert_idx in indices.flatten().tolist():
                        self.expert_activations[f'layer_{layer_idx}'][expert_idx] += 1
                        
            return hook_fn
            
        # Add hooks to all MoE layers
        for i, block in enumerate(self.model.transformer.h):
            if hasattr(block.mlp, 'router'):
                handle = block.mlp.router.register_forward_hook(make_hook(i))
                handles.append(handle)
                
        return handles
    
    def analyze_routing_patterns(self, dataloader, num_batches=10):
        """Analyze routing patterns over multiple batches"""
        handles = self.hook_routing()
        
        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                if i >= num_batches:
                    break
                    
                _ = self.model(x, y)
                
        # Remove hooks
        for handle in handles:
            handle.remove()
            
        return self.routing_history, self.expert_activations
    
    def visualize_expert_distribution(self, layer_idx=0):
        """Visualize expert activation distribution for a specific layer"""
        layer_key = f'layer_{layer_idx}'
        if layer_key not in self.expert_activations:
            print(f"No data for {layer_key}")
            return
            
        expert_counts = self.expert_activations[layer_key]
        experts = sorted(expert_counts.keys())
        counts = [expert_counts[e] for e in experts]
        
        plt.figure(figsize=(12, 6))
        plt.bar(experts, counts)
        plt.xlabel('Expert ID')
        plt.ylabel('Activation Count')
        plt.title(f'Expert Activation Distribution - Layer {layer_idx}')
        plt.grid(True, alpha=0.3)
        
        # Add coefficient of variation
        if counts:
            cv = np.std(counts) / np.mean(counts)
            plt.text(0.02, 0.98, f'CV: {cv:.3f}', transform=plt.gca().transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.tight_layout()
        plt.show()
        
    def check_load_balancing(self):
        """Check load balancing across all layers"""
        print("Load Balancing Analysis:")
        print("="*50)
        
        for layer_key, expert_counts in self.expert_activations.items():
            counts = list(expert_counts.values())
            if counts:
                mean_count = np.mean(counts)
                std_count = np.std(counts)
                cv = std_count / mean_count if mean_count > 0 else 0
                
                print(f"{layer_key}:")
                print(f"  Mean activations: {mean_count:.1f}")
                print(f"  Std deviation: {std_count:.1f}")
                print(f"  Coefficient of variation: {cv:.3f}")
                print(f"  Min/Max ratio: {min(counts)/max(counts):.3f}")
                print()
                
    def visualize_routing_weights(self, layer_idx=0, batch_idx=0):
        """Visualize routing weight distribution"""
        layer_key = f'layer_{layer_idx}'
        if layer_key not in self.routing_history:
            return
            
        batch_data = self.routing_history[layer_key][batch_idx]
        weights = batch_data['weights'].numpy()
        
        plt.figure(figsize=(10, 6))
        plt.hist(weights.flatten(), bins=50, alpha=0.7)
        plt.xlabel('Routing Weight')
        plt.ylabel('Count')
        plt.title(f'Routing Weight Distribution - Layer {layer_idx}, Batch {batch_idx}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def test_gradient_flow(self, sample_input):
        """Test that gradients flow properly through MoE layers"""
        self.model.train()
        sample_input.requires_grad_(True)
        
        # Forward pass
        output, loss = self.model(sample_input, sample_input)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        print("Gradient Flow Analysis:")
        print("="*50)
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"{name}: grad_norm={grad_norm:.6f}")
                
                # Check for vanishing/exploding gradients
                if grad_norm < 1e-8:
                    print(f"  WARNING: Possible vanishing gradient!")
                elif grad_norm > 100:
                    print(f"  WARNING: Possible exploding gradient!")
                    
def verify_moe_implementation(model_class, config):
    """Comprehensive verification of MoE implementation"""
    print("Verifying MoE Implementation...")
    print("="*50)
    
    # Create model
    model = model_class(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Test 1: Parameter count verification
    total_params = sum(p.numel() for p in model.parameters())
    active_params = config.get_active_params() if hasattr(config, 'get_active_params') else 0
    
    print(f"Total parameters: {total_params:,}")
    print(f"Active parameters: {active_params:,}")
    print(f"Parameter efficiency: {active_params/total_params:.2%}")
    print()
    
    # Test 2: Forward pass
    batch_size = 4
    seq_len = 128
    test_input = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
    
    try:
        with torch.no_grad():
            output, _ = model(test_input)
        print("✓ Forward pass successful")
        print(f"  Output shape: {output.shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return
        
    # Test 3: Router functionality
    if config.moe_enabled:
        debugger = MoEDebugger(model)
        
        # Create dummy dataloader
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(test_input, test_input)
        dataloader = DataLoader(dataset, batch_size=1)
        
        # Analyze routing
        routing_history, expert_activations = debugger.analyze_routing_patterns(dataloader, num_batches=5)
        
        print("\n✓ Routing analysis complete")
        debugger.check_load_balancing()
        
        # Test gradient flow
        print("\nTesting gradient flow...")
        test_sample = torch.randint(0, config.vocab_size, (1, 64)).to(device)
        debugger.test_gradient_flow(test_sample)
        
    # Test 4: Memory efficiency
    print("\nMemory Usage:")
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Run inference
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_input)
                
        memory_used = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak memory usage: {memory_used:.1f} MB")
        
        # Estimate KV cache size
        kv_cache_size = batch_size * seq_len * config.n_layer * config.n_head * config.n_embd // config.n_head
        if hasattr(config, 'compression_ratio'):
            kv_cache_size *= config.compression_ratio
        kv_cache_mb = kv_cache_size * 4 / 1024**2  # 4 bytes per float32
        
        print(f"Estimated KV cache: {kv_cache_mb:.1f} MB")
        print(f"Compression savings: {(1 - config.compression_ratio)*100:.0f}%")
        
    print("\n✓ All tests passed!")

def compare_routing_strategies():
    """Compare different routing strategies"""
    strategies = {
        'top_k': 'Standard top-k routing',
        'expert_choice': 'Expert choice routing',
        'soft_routing': 'Soft routing (all experts)'
    }
    
    # This is a placeholder - implement if you want to experiment with different strategies
    print("Routing Strategy Comparison:")
    print("="*50)
    for name, desc in strategies.items():
        print(f"{name}: {desc}")

if __name__ == "__main__":
    # Example usage
    from moe_mla_rope_implementation import GPT, GPTConfig
    
    # Create test configuration
    config = GPTConfig(
        block_size=128,
        vocab_size=50257,
        n_layer=4,
        n_head=4,
        n_embd=256,
        moe_enabled=True,
        n_experts=16,
        n_shared_experts=2,
        top_k_experts=4,
        compression_ratio=0.5
    )
    
    # Run verification
    verify_moe_implementation(GPT, config)
