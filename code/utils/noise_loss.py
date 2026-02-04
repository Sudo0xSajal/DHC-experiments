import torch
import torch.nn as nn

class NoiseLosses(nn.Module):
    """
    Provides:
    - L_sup_noise: supervise noise with |p - y_onehot| on labeled
    - L_u_disagree: on unlabeled, match noise to |pA - pB|
    - L_noise_cons: encourage nA ~ nB on the same input
    """
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='mean')
        self.mse = nn.MSELoss(reduction='mean')

    def labeled_target(self, p, y):
        # p: [B,C,...], y: [B,1,...] → |p - onehot(y)|
        tgt = torch.zeros_like(p)
        tgt.scatter_(1, y.long(), 1)
        return (p - tgt).abs()

    # OLD CODE - PROBLEMATIC: Only supervised nA_l, ignored nB_l even though it's provided
    # This caused Model B's noise to not be supervised on labeled data
    # def forward(self,
    #             pA_l=None, pB_l=None, y_l=None, nA_l=None, nB_l=None,
    #             pA_u=None, pB_u=None, nA_u=None, nB_u=None,
    #             w_l=1.0, w_u=1.0, w_cons=1.0):
    #     device = None
    #     for t in [pA_l, pA_u, nA_l, nA_u, pB_l, pB_u]:
    #         if t is not None:
    #             device = t.device; break
    #     zero = torch.tensor(0.0, device=device)
    #     L_sup, L_dis, L_cons = zero, zero, zero
    # 
    #     # PROBLEM: Only checks nA_l, ignores nB_l parameter
    #     if (pA_l is not None) and (y_l is not None) and (nA_l is not None):
    #         tgt = self.labeled_target(pA_l, y_l)
    #         L_sup = self.l1(nA_l, tgt) + self.l1(nB_l, tgt)  # nB_l might be None!
    #         L_sup = w_l * L_sup

    # NEW CODE - IMPROVED: Properly supervises both nA_l and nB_l for balanced training
    def forward(self,
                pA_l=None, pB_l=None, y_l=None, nA_l=None, nB_l=None,
                pA_u=None, pB_u=None, nA_u=None, nB_u=None,
                w_l=1.0, w_u=1.0, w_cons=1.0):
        device = None
        for t in [pA_l, pA_u, nA_l, nA_u, pB_l, pB_u]:
            if t is not None:
                device = t.device; break
        zero = torch.tensor(0.0, device=device)
        L_sup, L_dis, L_cons = zero, zero, zero

        # FIXED: Check for both nA_l AND nB_l before using them
        # Reason: Both models' noise should be supervised on labeled data
        if (pA_l is not None) and (y_l is not None) and (nA_l is not None) and (nB_l is not None):
            # Create separate targets for each model
            # Reason: Each model's noise should match its own prediction error
            tgt_A = self.labeled_target(pA_l, y_l)
            tgt_B = self.labeled_target(pB_l, y_l)
            
            # Supervise both models' noise predictions
            # Reason: Balanced training, both models learn noise estimation
            L_sup = self.l1(nA_l, tgt_A) + self.l1(nB_l, tgt_B)
            L_sup = w_l * L_sup

        if (pA_u is not None) and (pB_u is not None) and (nA_u is not None) and (nB_u is not None):
            disagree = (pA_u - pB_u).abs()  
            L_dis = self.mse(nA_u, disagree) + self.mse(nB_u, disagree)
            L_dis = w_u * L_dis
            L_cons = self.mse(nA_u, nB_u) * w_cons

        return L_sup, L_dis, L_cons

    # NEW METHOD ADDED: Structured forward pass for cleaner API
    def forward_structured(self, labeled_dict=None, unlabeled_dict=None, 
                          weights=None, num_classes=None):
        """
        Structured forward pass with dictionary inputs.
        Reason: Cleaner API, less error-prone than many positional arguments
        """
        if labeled_dict is None and unlabeled_dict is None:
            return self.forward()
        
        # Extract with defaults
        pA_l = labeled_dict.get('pA_l') if labeled_dict else None
        pB_l = labeled_dict.get('pB_l') if labeled_dict else None
        y_l = labeled_dict.get('y_l') if labeled_dict else None
        nA_l = labeled_dict.get('nA_l') if labeled_dict else None
        nB_l = labeled_dict.get('nB_l') if labeled_dict else None
        
        pA_u = unlabeled_dict.get('pA_u') if unlabeled_dict else None
        pB_u = unlabeled_dict.get('pB_u') if unlabeled_dict else None
        nA_u = unlabeled_dict.get('nA_u') if unlabeled_dict else None
        nB_u = unlabeled_dict.get('nB_u') if unlabeled_dict else None
        
        # Use default weights if not provided
        default_weights = {'w_l': 1.0, 'w_u': 1.0, 'w_cons': 1.0}
        if weights:
            default_weights.update(weights)
        
        return self.forward(
            pA_l=pA_l, pB_l=pB_l, y_l=y_l, nA_l=nA_l, nB_l=nB_l,
            pA_u=pA_u, pB_u=pB_u, nA_u=nA_u, nB_u=nB_u,
            w_l=default_weights['w_l'], 
            w_u=default_weights['w_u'], 
            w_cons=default_weights['w_cons']
        )