import torch

class NoiseHistoryBuffer:
    def __init__(self, K=10):
        self.K = K
        self.buffer_A = []
        self.buffer_B = []
    
    def update(self, n_A, n_B):
        """Add new noise predictions to buffer"""
        # Detach and clone to avoid memory issues
        # self.buffer_A.append(n_A.detach().clone()) #stores full tensors
        # self.buffer_B.append(n_B.detach().clone())
        # Store statistics instead of full tensors #changes 1
        self.buffer_A_mean.append(n_A.mean().item())
        self.buffer_A_std.append(n_A.std().item())
        self.buffer_B_mean.append(n_B.mean().item())
        self.buffer_B_std.append(n_B.std().item())
        
        # Remove oldest if buffer exceeds K
        # if len(self.buffer_A) > self.K:
        #     self.buffer_A.pop(0)
        #     self.buffer_B.pop(0)

        # Keep running buffers within K #changes 2
        if len(self.buffer_A_mean) > self.K:
           self.buffer_A_mean.pop(0)
           self.buffer_A_std.pop(0)
           self.buffer_B_mean.pop(0)
           self.buffer_B_std.pop(0)
    
    def get_stability(self):
        """Calculate stability from variance"""
        if len(self.buffer_A) < 2:
            # Not enough history, return ones (fully stable)
            return None, None
        
        # Stack buffers: [K, C, D, H, W]
        stack_A = torch.stack(self.buffer_A, dim=0)
        stack_B = torch.stack(self.buffer_B, dim=0)
        
        # Calculate variance across epochs (dim=0)
        var_A = torch.var(stack_A, dim=0)
        var_B = torch.var(stack_B, dim=0)
        
        # Calculate stability: 1 / (1 + variance)
        stability_A = 1.0 / (1.0 + var_A)
        stability_B = 1.0 / (1.0 + var_B)
        
        return stability_A, stability_B
    
    # OLD CODE - PROBLEMATIC: Multiplies raw noise values which can be negative or large
    # This causes alpha to be unbounded and potentially negative, leading to unstable training
    # def get_adaptive_alpha(self, n_A, n_B, alpha_max=0.7):
    #     """Get voxel-wise alpha based on stability and noise"""
    #     stability_A, stability_B = self.get_stability()
    #     
    #     if stability_A is None:
    #         # Not enough history, return global alpha
    #         return alpha_max, alpha_max
    #     
    #     # α(v) = α_max × stability(v) × n(v)  # PROBLEM: n(v) can be negative/large!
    #     alpha_A = alpha_max * stability_A * n_A
    #     alpha_B = alpha_max * stability_B * n_B
    #     
    #     return alpha_A, alpha_B
    
    # NEW CODE - IMPROVED: Normalizes noise magnitude and adds bounds for stability
    def get_adaptive_alpha(self, n_A, n_B, alpha_max=0.7, min_alpha=0.05):
        """
        Get voxel-wise alpha with proper scaling and bounds.
        
        Reason: Original multiplied raw noise values which could be negative or large,
        causing unstable training. Now normalize noise magnitude and add bounds.
        """
        stability_A, stability_B = self.get_stability()
        
        if stability_A is None:
            # Not enough history, return tensor of alpha_max for broadcasting
            return (alpha_max * torch.ones_like(n_A), 
                    alpha_max * torch.ones_like(n_B))
        
        # Normalize noise to [0, 1] range using absolute value + sigmoid
        # Reason: Noise magnitude should be positive, extreme values cause instability
        n_A_mag = torch.sigmoid(n_A.abs())  # Convert to [0, 1]
        n_B_mag = torch.sigmoid(n_B.abs())
        
        # Alpha = α_max × stability × normalized_noise_magnitude
        # Reason: Ensures alpha is proportional to noise strength but bounded
        alpha_A = alpha_max * stability_A * n_A_mag
        alpha_B = alpha_max * stability_B * n_B_mag
        
        # Clamp to reasonable range
        # Reason: Prevents vanishing (too small) or excessive (too large) correction
        alpha_A = torch.clamp(alpha_A, min=min_alpha, max=alpha_max)
        alpha_B = torch.clamp(alpha_B, min=min_alpha, max=alpha_max)
        
        return alpha_A, alpha_B
    
    # NEW METHOD ADDED: Monitor buffer statistics for debugging
    def get_buffer_stats(self):
        """Return buffer statistics for logging"""
        stats = {
            'size': len(self.buffer_A),
            'capacity': self.K,
            'filled_ratio': len(self.buffer_A) / self.K if self.K > 0 else 0
        }
        
        if len(self.buffer_A) > 0:
            stats['mean_magnitude_A'] = torch.mean(torch.abs(self.buffer_A[-1])).item()
            stats['mean_magnitude_B'] = torch.mean(torch.abs(self.buffer_B[-1])).item()
        
        return stats