import torch

# OLD CODE - SIMPLE: Direct subtraction with no safety checks
# This can cause scale mismatches between logits and noise, leading to training instability
# def cross_correct_logits(logits_A, noise_B, alpha=0.7):
#     # subtract other model's noise; block gradient through that noise
#     return logits_A - alpha * noise_B.detach()

# NEW CODE - IMPROVED: Safe cross-correction with magnitude matching
def cross_correct_logits(logits_A, noise_B, alpha=0.7, safe_mode=True):
    """
    Safe cross-correction with magnitude matching.
    
    Reason: Direct subtraction can cause logits to explode if noise magnitude
    doesn't match logits scale. This version matches scales for stability.
    """
    # Detach noise to stop gradient flow through other model
    noise_detached = noise_B.detach()
    
    if safe_mode:
        # Match noise scale to logits scale
        # Reason: Prevent scale mismatch that causes training instability
        logits_std = logits_A.std() + 1e-8
        noise_std = noise_detached.std() + 1e-8
        
        # Scale noise to have similar std as logits
        scale_factor = logits_std / noise_std
        scale_factor = torch.clamp(scale_factor, 0.1, 10.0)  # Bound scaling
        
        noise_scaled = noise_detached * scale_factor
    else:
        noise_scaled = noise_detached
    
    # Apply correction
    corrected = logits_A - alpha * noise_scaled
    
    # Optional debug check
    if torch.isnan(corrected).any() or torch.isinf(corrected).any():
        raise ValueError("Cross correction produced NaN/Inf values!")
    
    return corrected

def refined_pseudo_from(logits, dim=1):
    probs = torch.softmax(logits, dim=dim)
    return torch.argmax(probs, dim=dim, keepdim=True).long()

# NEW FUNCTION ADDED: Monitor pseudo-label quality for debugging
def compute_pseudo_quality(pseudo_A, pseudo_B, mask=None):
    """
    Compute pseudo-label agreement and confidence.
    Reason: Monitor quality of generated pseudo-labels for debugging
    """
    agreement = (pseudo_A == pseudo_B).float()
    
    if mask is not None:
        agreement = agreement * mask
    
    agreement_rate = agreement.mean().item()
    
    return {
        'agreement_rate': agreement_rate,
        'total_voxels': pseudo_A.numel(),
        'agreeing_voxels': agreement.sum().item()
    }

# NEW FUNCTION ADDED: Batch processing for cleaner training loop
def batch_cross_correction(batch_data, noise_buffer=None, alpha=0.7):
    """
    Apply cross-correction to a batch of data.
    Reason: Encapsulates the correction logic for cleaner training code
    """
    result = {}
    
    # Extract data
    A_logits = batch_data.get('A_logits')
    B_logits = batch_data.get('B_logits')
    A_noise = batch_data.get('A_noise')
    B_noise = batch_data.get('B_noise')
    
    if A_logits is None or B_noise is None:
        return batch_data
    
    # Get adaptive alpha if buffer provided
    if noise_buffer is not None and A_noise is not None and B_noise is not None:
        alpha_A, alpha_B = noise_buffer.get_adaptive_alpha(A_noise, B_noise, alpha_max=alpha)
    else:
        alpha_A = alpha_B = alpha
    
    # Apply correction
    result['A_corrected'] = cross_correct_logits(A_logits, B_noise, alpha=alpha_B)
    result['B_corrected'] = cross_correct_logits(B_logits, A_noise, alpha=alpha_A)
    
    # Generate pseudo-labels from corrected logits
    result['pseudo_A'] = refined_pseudo_from(result['A_corrected'])
    result['pseudo_B'] = refined_pseudo_from(result['B_corrected'])
    
    return result
#new function #changes 1
def analyze_noise_correction(A_logits, B_logits, A_noise, B_noise, alpha_A, alpha_B):
    """
    Analyze noise correction effectiveness
    """
    with torch.no_grad():
        # Before correction
        pseudo_before_A = refined_pseudo_from(A_logits)
        pseudo_before_B = refined_pseudo_from(B_logits)
        agreement_before = (pseudo_before_A == pseudo_before_B).float().mean().item()
        
        # After correction
        A_corrected = cross_correct_logits(A_logits, B_noise, alpha=alpha_B)
        B_corrected = cross_correct_logits(B_logits, A_noise, alpha=alpha_A)
        pseudo_after_A = refined_pseudo_from(A_corrected)
        pseudo_after_B = refined_pseudo_from(B_corrected)
        agreement_after = (pseudo_after_A == pseudo_after_B).float().mean().item()
        
        # Correction magnitude
        correction_mag_A = (A_corrected - A_logits).abs().mean().item()
        correction_mag_B = (B_corrected - B_logits).abs().mean().item()
    
    return {
        'agreement_before': agreement_before,
        'agreement_after': agreement_after,
        'agreement_improvement': agreement_after - agreement_before,
        'correction_mag_A': correction_mag_A,
        'correction_mag_B': correction_mag_B
    }