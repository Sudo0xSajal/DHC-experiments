# noise_buffer_fixed.py - Fixed with better shape handling
import torch
from typing import Optional, Tuple, List

class NoiseHistoryBuffer:
    def __init__(self, K: int = 10, device: torch.device = None):
        """
        Initialize buffer with device management.
        """
        self.K = K
        self.buffer_A: List[torch.Tensor] = []
        self.buffer_B: List[torch.Tensor] = []
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def update(self, n_A: torch.Tensor, n_B: torch.Tensor) -> None:
        """Add new noise predictions to buffer with proper memory management."""
        try:
            # Move to buffer device and detach from computation graph
            n_A = n_A.to(self.device).detach().clone()
            n_B = n_B.to(self.device).detach().clone()
            
            # ----------------------------------------------------------------------
            # OLD CODE - Inconsistent handling of batch dimension
            # Remove batch dimension if it's size 1 for consistency
            # if n_A.dim() == 5 and n_A.size(0) == 1:
            #     n_A = n_A.squeeze(0)
            # if n_B.dim() == 5 and n_B.size(0) == 1:
            #     n_B = n_B.squeeze(0)
            # 
            # WHY NOT USING OLD CODE:
            # 1. Causes inconsistent shapes: some have batch dim, some don't
            # 2. Makes stacking impossible for stability calculation
            # 3. Should keep batch dimension for consistency
            # ----------------------------------------------------------------------
            
            # ----------------------------------------------------------------------
            # NEW IMPROVED CODE: Always keep batch dimension
            # ----------------------------------------------------------------------
            # Ensure both have batch dimension (dim=0)
            if n_A.dim() == 4:  # [C, D, H, W] format
                n_A = n_A.unsqueeze(0)  # Add batch dimension -> [1, C, D, H, W]
            if n_B.dim() == 4:  # [C, D, H, W] format
                n_B = n_B.unsqueeze(0)  # Add batch dimension -> [1, C, D, H, W]
            
            # If batch size > 1, we need to handle it (take mean or select first)
            if n_A.size(0) > 1:
                # Option 1: Take mean across batch (preserves information)
                n_A = n_A.mean(dim=0, keepdim=True)
                # Option 2: Take first element
                # n_A = n_A[0:1]
            if n_B.size(0) > 1:
                n_B = n_B.mean(dim=0, keepdim=True)
                # n_B = n_B[0:1]
            
            # Add to buffer
            self.buffer_A.append(n_A)
            self.buffer_B.append(n_B)
            
            # Enforce buffer size limit
            if len(self.buffer_A) > self.K:
                self.buffer_A.pop(0)
                self.buffer_B.pop(0)
                
        except Exception as e:
            raise RuntimeError(f"Failed to update noise buffer: {str(e)}")
    
    def _ensure_consistent_shape(self, tensor_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """Ensure all tensors in list have same shape."""
        if len(tensor_list) < 2:
            return tensor_list
        
        # ----------------------------------------------------------------------
        # OLD CODE - Complex shape checking that fails
        # first_tensor = tensor_list[0]
        # has_batch_dim = first_tensor.dim() == 5
        # 
        # consistent_list = []
        # for i, tensor in enumerate(tensor_list):
        #     if has_batch_dim and tensor.dim() == 4:
        #         tensor = tensor.unsqueeze(0)
        #     elif not has_batch_dim and tensor.dim() == 5:
        #         if tensor.size(0) == 1:
        #             tensor = tensor.squeeze(0)
        #         else:
        #             raise ValueError(f"Tensor {i} has batch size {tensor.size(0)}, expected 1 or 4D tensor")
        #     
        #     if tensor.shape[-4:] != first_tensor.shape[-4:]:
        #         raise ValueError(f"Tensor {i} shape {tensor.shape} doesn't match first tensor shape {first_tensor.shape}")
        #     
        #     consistent_list.append(tensor)
        # 
        # WHY NOT USING OLD CODE:
        # 1. Too complex with multiple conditions
        # 2. Inconsistent handling leads to errors
        # 3. Need simpler, more robust approach
        # ----------------------------------------------------------------------
        
        # ----------------------------------------------------------------------
        # NEW IMPROVED CODE: Simpler, more robust shape handling
        # ----------------------------------------------------------------------
        consistent_list = []
        
        for i, tensor in enumerate(tensor_list):
            # Ensure 5D tensor [B, C, D, H, W]
            if tensor.dim() == 4:
                tensor = tensor.unsqueeze(0)  # Add batch dimension
            
            # Ensure batch size is 1 (take mean if > 1)
            if tensor.size(0) > 1:
                tensor = tensor.mean(dim=0, keepdim=True)
            
            consistent_list.append(tensor)
        
        # Verify all tensors have same shape
        first_shape = consistent_list[0].shape
        for i, tensor in enumerate(consistent_list[1:], 1):
            if tensor.shape != first_shape:
                # Try to reshape if spatial dimensions differ (shouldn't happen)
                if tensor.shape[1:] != first_shape[1:]:
                    raise ValueError(
                        f"Tensor {i} shape {tensor.shape} doesn't match "
                        f"first tensor shape {first_shape}"
                    )
        
        return consistent_list
    
    def get_stability(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Calculate stability from variance with numerical stability."""
        if len(self.buffer_A) < 2:
            return None, None
        
        try:
            # Ensure consistent shapes before stacking
            buffer_A_consistent = self._ensure_consistent_shape(self.buffer_A)
            buffer_B_consistent = self._ensure_consistent_shape(self.buffer_B)
            
            # Stack buffers along new dimension [K, 1, C, D, H, W]
            stack_A = torch.stack(buffer_A_consistent, dim=0)
            stack_B = torch.stack(buffer_B_consistent, dim=0)
            
            # Calculate variance across time dimension (dim=0)
            var_A = torch.var(stack_A, dim=0, unbiased=False)
            var_B = torch.var(stack_B, dim=0, unbiased=False)
            
            # Add small epsilon to prevent numerical issues
            eps = 1e-8
            
            # Calculate stability: 1 / (1 + variance)
            stability_A = 1.0 / (1.0 + var_A + eps)
            stability_B = 1.0 / (1.0 + var_B + eps)
            
            # Clamp stability to [0, 1] for safety
            stability_A = torch.clamp(stability_A, 0.0, 1.0)
            stability_B = torch.clamp(stability_B, 0.0, 1.0)
            
            return stability_A, stability_B
            
        except Exception as e:
            # Instead of printing, just return None
            # print(f"Warning: Could not calculate stability: {str(e)}")
            return None, None
    
    def get_adaptive_alpha(
        self, 
        n_A: torch.Tensor, 
        n_B: torch.Tensor, 
        alpha_max: float = 0.7,
        normalize_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get voxel-wise alpha based on stability and noise."""
        stability_A, stability_B = self.get_stability()
        
        if stability_A is None:
            # Not enough history or calculation failed, return global alpha
            # Ensure correct shape with batch dimension
            if n_A.dim() == 4:
                shape = (1,) + n_A.shape  # Add batch dimension
            else:
                shape = n_A.shape
                
            alpha_A = torch.full(shape, alpha_max, device=n_A.device, dtype=n_A.dtype)
            alpha_B = torch.full(shape, alpha_max, device=n_B.device, dtype=n_B.dtype)
            return alpha_A, alpha_B
        
        try:
            current_n_A = n_A.to(self.device)
            current_n_B = n_B.to(self.device)
            
            # Ensure both have batch dimension
            if current_n_A.dim() == 4:
                current_n_A = current_n_A.unsqueeze(0)
            if current_n_B.dim() == 4:
                current_n_B = current_n_B.unsqueeze(0)
            
            # Ensure stability has batch dimension to match
            if stability_A.dim() == 4 and current_n_A.dim() == 5:
                stability_A = stability_A.unsqueeze(0)
                stability_B = stability_B.unsqueeze(0)
            
            # Handle batch size > 1 in current noise
            if current_n_A.size(0) > 1:
                current_n_A = current_n_A.mean(dim=0, keepdim=True)
            if current_n_B.size(0) > 1:
                current_n_B = current_n_B.mean(dim=0, keepdim=True)
            
            if normalize_noise:
                # Normalize noise predictions to [0, 1] for consistent scaling
                # Use min-max normalization
                n_A_min = current_n_A.amin(dim=(-3, -2, -1), keepdim=True)
                n_A_max = current_n_A.amax(dim=(-3, -2, -1), keepdim=True)
                n_A_norm = (current_n_A - n_A_min) / (n_A_max - n_A_min + 1e-8)
                
                n_B_min = current_n_B.amin(dim=(-3, -2, -1), keepdim=True)
                n_B_max = current_n_B.amax(dim=(-3, -2, -1), keepdim=True)
                n_B_norm = (current_n_B - n_B_min) / (n_B_max - n_B_min + 1e-8)
            else:
                n_A_norm = current_n_A
                n_B_norm = current_n_B
            
            # α(v) = α_max × stability(v) × n(v)
            alpha_A = alpha_max * stability_A * n_A_norm
            alpha_B = alpha_max * stability_B * n_B_norm
            
            # Clamp alphas to valid range [0, alpha_max]
            alpha_A = torch.clamp(alpha_A, 0.0, alpha_max)
            alpha_B = torch.clamp(alpha_B, 0.0, alpha_max)
            
            return alpha_A, alpha_B
            
        except Exception as e:
            # Fallback to global alpha
            if n_A.dim() == 4:
                shape = (1,) + n_A.shape
            else:
                shape = n_A.shape
                
            alpha_A = torch.full(shape, alpha_max, device=n_A.device, dtype=n_A.dtype)
            alpha_B = torch.full(shape, alpha_max, device=n_B.device, dtype=n_B.dtype)
            return alpha_A, alpha_B
    
    @property
    def is_ready(self) -> bool:
        """Check if buffer has enough data for stability calculation."""
        return len(self.buffer_A) >= 2
    
    @property
    def current_size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer_A)
    
    def clear(self) -> None:
        """Clear buffer and free memory."""
        self.buffer_A.clear()
        self.buffer_B.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def debug_info(self):
        """Print debug information about buffer contents."""
        if not self.buffer_A:
            print("Buffer is empty")
            return
        
        print(f"Buffer size: {len(self.buffer_A)}")
        print(f"Buffer device: {self.device}")
        
        # Check shapes
        shapes_A = [t.shape for t in self.buffer_A]
        shapes_B = [t.shape for t in self.buffer_B]
        
        print(f"Shapes in buffer A: {set(shapes_A)}")
        print(f"Shapes in buffer B: {set(shapes_B)}")
        
        # Check if all shapes are consistent
        consistent_A = all(s == shapes_A[0] for s in shapes_A)
        consistent_B = all(s == shapes_B[0] for s in shapes_B)
        
        print(f"Buffer A consistent: {consistent_A}")
        print(f"Buffer B consistent: {consistent_B}")
        
        if consistent_A and consistent_B:
            print(f"All tensors have shape: {shapes_A[0]}")