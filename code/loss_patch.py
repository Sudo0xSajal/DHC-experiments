# patch_loss.py - Apply fixes to loss.py
import torch

def patch_loss_file():
    """Patch utils/loss.py to handle GPU weight tensors."""
    
    with open('utils/loss.py', 'r') as f:
        content = f.read()
    
    # Patch RobustCrossEntropyLoss
    old_robust_init = '''    def __init__(self, weight=None):
        if weight is not None:
            weight = torch.FloatTensor(weight).cuda()
        super().__init__(weight=weight)'''
    
    new_robust_init = '''    def __init__(self, weight=None):
        if weight is not None:
            if torch.is_tensor(weight):
                # Weight is already a tensor
                weight = weight.float()
                if weight.device.type != 'cuda':
                    # Move to GPU if not already there
                    weight = weight.cuda()
            else:
                # Weight is numpy array or list
                # Create on CPU first, then move to GPU
                weight = torch.FloatTensor(weight).cuda()
        super().__init__(weight=weight)'''
    
    if old_robust_init in content:
        content = content.replace(old_robust_init, new_robust_init)
        print("✓ Patched RobustCrossEntropyLoss")
    else:
        print("✗ Could not find RobustCrossEntropyLoss.__init__ to patch")
    
    # Patch SoftDiceLoss
    old_dice_init = '''    def __init__(self, weight=None, apply_nonlin=None, batch_dice=True, do_bg=False, smooth=1.):
        """
        """
        super(SoftDiceLoss, self).__init__()
        if weight is not None:
            weight = torch.FloatTensor(weight).cuda()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.weight = weight'''
    
    new_dice_init = '''    def __init__(self, weight=None, apply_nonlin=None, batch_dice=True, do_bg=False, smooth=1.):
        """
        """
        super(SoftDiceLoss, self).__init__()
        if weight is not None:
            if torch.is_tensor(weight):
                # Weight is already a tensor
                weight = weight.float()
                if weight.device.type != 'cuda':
                    # Move to GPU if not already there
                    weight = weight.cuda()
            else:
                # Weight is numpy array or list
                # Create on CPU first, then move to GPU
                weight = torch.FloatTensor(weight).cuda()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.weight = weight'''
    
    if old_dice_init in content:
        content = content.replace(old_dice_init, new_dice_init)
        print("✓ Patched SoftDiceLoss")
    else:
        print("✗ Could not find SoftDiceLoss.__init__ to patch")
    
    # Save the patched file
    with open('utils/loss.py', 'w') as f:
        f.write(content)
    
    print("✓ Loss file patched successfully!")

if __name__ == '__main__':
    patch_loss_file()