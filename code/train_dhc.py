# train_dhc.py - Complete Enhanced Version
import os
import sys
import logging
from tqdm import tqdm
import argparse

# ============================================
# SECTION A: Enhanced Argument Parser
# ============================================

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='synapse')
parser.add_argument('--exp', type=str)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('-sl', '--split_labeled', type=str, default='labeled_20p')
parser.add_argument('-su', '--split_unlabeled', type=str, default='unlabeled_80p')
parser.add_argument('-se', '--split_eval', type=str, default='eval')
parser.add_argument('-m', '--mixed_precision', action='store_true', default=True)
parser.add_argument('-ep', '--max_epoch', type=int, default=500)
parser.add_argument('--cps_loss', type=str, default='wce')
parser.add_argument('--sup_loss', type=str, default='w_ce+dice')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--base_lr', type=float, default=0.001)
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('-w', '--cps_w', type=float, default=1)
parser.add_argument('-r', '--cps_rampup', action='store_true', default=True)
parser.add_argument('-cr', '--consistency_rampup', type=float, default=None)
parser.add_argument('--alpha_noise', type=float, default=0.7)
parser.add_argument('--alpha_rampup', type=int, default=100)
parser.add_argument('--w_sup_noise', type=float, default=0.3)
parser.add_argument('--w_u_noise', type=float, default=0.2)
parser.add_argument('--w_noise_cons', type=float, default=0.1)

# NEW ARGUMENTS ADDED: Training stability parameters
# Reason: Original code missing critical stability controls for noise training
parser.add_argument('--grad_clip', type=float, default=1.0,
                   help='Gradient clipping norm (default: 1.0)')
parser.add_argument('--min_alpha', type=float, default=0.05,
                   help='Minimum alpha for noise cancellation (default: 0.05)')
parser.add_argument('--warmup_epochs', type=int, default=50,
                   help='Warmup epochs before full noise training (default: 50)')
parser.add_argument('--noise_safe_mode', action='store_true', default=True,
                   help='Use safe noise scaling (default: True)')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from models.vnet import VNet
from utils import EMA, maybe_mkdir, get_lr, fetch_data, seed_worker, poly_lr, print_func, kaiming_normal_init_weight
from utils.loss import DC_and_CE_loss, RobustCrossEntropyLoss, SoftDiceLoss
from utils.noise_loss import NoiseLosses
from utils.noise_utils import cross_correct_logits, refined_pseudo_from
from data.transforms import RandomCrop, CenterCrop, ToTensor, RandomFlip_LR, RandomFlip_UD
from data.data_loaders import Synapse_AMOS
from utils.config import Config
from utils.noise_buffer import NoiseHistoryBuffer

config = Config(args.task)

# ============================================
# NEW FUNCTIONS ADDED: Validation and Gradient Clipping
# ============================================

# NEW FUNCTION ADDED: Tensor validation for early error detection
# Reason: Noise training can produce NaN/Inf values that break training
def validate_tensors(*tensors, name=""):
    """
    Validate tensors for NaN/Inf and extreme values.
    Reason: Noise training can produce invalid values that break training
    """
    for i, t in enumerate(tensors):
        if t is None:
            continue
        
        # Check for NaN/Inf
        if torch.isnan(t).any():
            raise ValueError(f"[{name}] Tensor {i} contains NaN values!")
        if torch.isinf(t).any():
            raise ValueError(f"[{name}] Tensor {i} contains Inf values!")
        
        # Check for extreme values
        if t.abs().max() > 100.0:
            logging.warning(f"[{name}] Tensor {i} has extreme values: "
                          f"max_abs={t.abs().max():.2f}")

# NEW FUNCTION ADDED: Gradient clipping to prevent explosion
# Reason: Unstable noise training can cause gradient explosion
def clip_gradients(model, scaler, optimizer, max_norm=1.0):
    """
    Clip gradients to prevent explosion.
    Reason: Unstable noise training can cause gradient explosion
    """
    # Unscale gradients first (important for mixed precision)
    scaler.unscale_(optimizer)
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

# ============================================
# Existing Helper Functions (Unchanged)
# ============================================

def sigmoid_rampup(current, rampup_length):
    '''Exponential rampup from https://arxiv.org/abs/1610.02242'''
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch):
    if args.cps_rampup:
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        if args.consistency_rampup is None:
            args.consistency_rampup = args.max_epoch
        return args.cps_w * sigmoid_rampup(epoch, args.consistency_rampup)
    else:
        return args.cps_w

# ENHANCED FUNCTION: Modified to use args.min_alpha
def get_alpha(epoch):
     if args.alpha_rampup and args.alpha_rampup > 0:
            # Use min_alpha as minimum value during rampup
            alpha_min = args.min_alpha if hasattr(args, 'min_alpha') else 0.05
            alpha_range = args.alpha_noise - alpha_min
            return alpha_min + alpha_range * sigmoid_rampup(epoch, args.alpha_rampup)
     return args.alpha_noise

def make_loss_function(name, weight=None):
    if name == 'ce':
        return RobustCrossEntropyLoss()
    elif name == 'wce':
        return RobustCrossEntropyLoss(weight=weight)
    elif name == 'ce+dice':
        return DC_and_CE_loss()
    elif name == 'wce+dice':
        return DC_and_CE_loss(w_ce=weight)
    elif name == 'w_ce+dice':
        return DC_and_CE_loss(w_dc=weight, w_ce=weight)
    else:
        raise ValueError(name)

def make_loader(split, dst_cls=Synapse_AMOS, repeat=None, is_training=True, unlabeled=False):
    if is_training:
        dst = dst_cls(
            task=args.task,
            split=split,
            repeat=repeat,
            unlabeled=unlabeled,
            num_cls=config.num_cls,
            transform=transforms.Compose([
                RandomCrop(config.patch_size, args.task),
                RandomFlip_LR(),
                RandomFlip_UD(),
                ToTensor()
            ])
        )
        return DataLoader(
            dst,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker
        )
    else:
        dst = dst_cls(
            task=args.task,
            split=split,
            is_val=True,
            num_cls=config.num_cls,
            transform=transforms.Compose([
                CenterCrop(config.patch_size, args.task),
                ToTensor()
            ])
        )
        return DataLoader(dst, pin_memory=True)

def make_model_all():
    model = VNet(
        n_channels=config.num_channels,
        n_classes=config.num_cls,
        n_filters=config.n_filters,
        normalization='batchnorm',
        has_dropout=True
    ).cuda()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=3e-5,
        nesterov=True
    )

    return model, optimizer

# ============================================
# Existing Classes (Unchanged)
# ============================================

class DistDW:
    def __init__(self, num_cls, do_bg=False, momentum=0.95):
        self.num_cls = num_cls
        self.do_bg = do_bg
        self.momentum = momentum

    def _cal_weights(self, num_each_class):
        num_each_class = torch.FloatTensor(num_each_class).cuda()
        P = (num_each_class.max()+1e-8) / (num_each_class+1e-8)
        P_log = torch.log(P)
        weight = P_log / P_log.max()
        return weight

    def init_weights(self, labeled_dataset):
        if labeled_dataset.unlabeled:
            raise ValueError
        num_each_class = np.zeros(self.num_cls)
        for data_id in labeled_dataset.ids_list:
            _, _, label = labeled_dataset._get_data(data_id)
            label = label.reshape(-1)
            tmp, _ = np.histogram(label, range(self.num_cls + 1))
            num_each_class += tmp
        weights = self._cal_weights(num_each_class)
        self.weights = weights * self.num_cls
        return self.weights.data.cpu().numpy()

    def get_ema_weights(self, pseudo_label):
        pseudo_label = torch.argmax(pseudo_label.detach(), dim=1, keepdim=True).long()
        label_numpy = pseudo_label.data.cpu().numpy()
        num_each_class = np.zeros(self.num_cls)
        for i in range(label_numpy.shape[0]):
            label = label_numpy[i].reshape(-1)
            tmp, _ = np.histogram(label, range(self.num_cls + 1))
            num_each_class += tmp

        cur_weights = self._cal_weights(num_each_class) * self.num_cls
        self.weights = EMA(cur_weights, self.weights, momentum=self.momentum)
        return self.weights

class DiffDW:
    def __init__(self, num_cls, accumulate_iters=20):
        self.last_dice = torch.zeros(num_cls).float().cuda() + 1e-8
        self.dice_func = SoftDiceLoss(smooth=1e-8, do_bg=True)
        self.cls_learn = torch.zeros(num_cls).float().cuda()
        self.cls_unlearn = torch.zeros(num_cls).float().cuda()
        self.num_cls = num_cls
        self.dice_weight = torch.zeros(num_cls).float().cuda()
        self.accumulate_iters = accumulate_iters

    def init_weights(self):
        weights = np.ones(config.num_cls) * self.num_cls
        self.weights = torch.FloatTensor(weights).cuda()
        return weights

    def cal_weights(self, pred,  label):
        x_onehot = torch.zeros(pred.shape).cuda()
        output = torch.argmax(pred, dim=1, keepdim=True).long()
        x_onehot.scatter_(1, output, 1)
        y_onehot = torch.zeros(pred.shape).cuda()
        y_onehot.scatter_(1, label, 1)
        cur_dice = self.dice_func(x_onehot, y_onehot, is_training=False)
        delta_dice = cur_dice - self.last_dice
        cur_cls_learn = torch.where(delta_dice>0, delta_dice, 0) * torch.log(cur_dice / self.last_dice)
        cur_cls_unlearn = torch.where(delta_dice<=0, delta_dice, 0) * torch.log(cur_dice / self.last_dice)
        self.last_dice = cur_dice
        self.cls_learn = EMA(cur_cls_learn, self.cls_learn, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        self.cls_unlearn = EMA(cur_cls_unlearn, self.cls_unlearn, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        cur_diff = (self.cls_unlearn + 1e-8) / (self.cls_learn + 1e-8)
        cur_diff = torch.pow(cur_diff, 1/5)
        self.dice_weight = EMA(1. - cur_dice, self.dice_weight, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        weights = cur_diff * self.dice_weight
        weights = weights / weights.max()
        return weights * self.num_cls

# ============================================
# Main Training Script
# ============================================

if __name__ == '__main__':
    import random
    SEED=args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    # make logger file
    snapshot_path = f'./logs/{args.exp}/'
    maybe_mkdir(snapshot_path)
    maybe_mkdir(os.path.join(snapshot_path, 'ckpts'))

    # make logger
    writer = SummaryWriter(os.path.join(snapshot_path, 'tensorboard'))
    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'train.log'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # make data loader
    unlabeled_loader = make_loader(args.split_unlabeled, unlabeled=True)
    labeled_loader = make_loader(args.split_labeled, repeat=len(unlabeled_loader.dataset))
    eval_loader = make_loader(args.split_eval, is_training=False)

    logging.info(f'{len(labeled_loader)} itertations per epoch (labeled)')
    logging.info(f'{len(unlabeled_loader)} itertations per epoch (unlabeled)')

    # make model, optimizer
    model_A, optimizer_A = make_model_all()
    model_B, optimizer_B  = make_model_all()
    model_A = kaiming_normal_init_weight(model_A)
    model_B = kaiming_normal_init_weight(model_B)

    # instantiate noise losses
    noise_losses = NoiseLosses()
    
    # Create noise buffer (using updated implementation)
    noise_buffer = NoiseHistoryBuffer(K=10)

    # make loss function
    diffdw = DiffDW(config.num_cls, accumulate_iters=50)
    distdw = DistDW(config.num_cls, momentum=0.99)

    weight_A = diffdw.init_weights()
    weight_B = distdw.init_weights(labeled_loader.dataset)

    loss_func_A     = make_loss_function(args.sup_loss, weight_A)
    loss_func_B     = make_loss_function(args.sup_loss, weight_B)
    cps_loss_func_A = make_loss_function(args.cps_loss, weight_A)
    cps_loss_func_B = make_loss_function(args.cps_loss, weight_B)

    if args.mixed_precision:
        amp_grad_scaler = GradScaler()

    cps_w = get_current_consistency_weight(0)
    best_eval = 0.0
    best_epoch = 0
    
    # NEW: Initialize monitoring variables
    noise_mean_A, noise_mean_B = 0.0, 0.0
    noise_std_A, noise_std_B = 0.0, 0.0
    alpha_mean, alpha_std = 0.0, 0.0
    agreement = 0.0
    
    for epoch_num in range(args.max_epoch + 1):
        loss_list = []
        loss_cps_list = []
        loss_sup_list = []
        loss_sup_noise_list = []
        loss_u_dis_list = []
        loss_cons_list = []

        model_A.train()
        model_B.train()
        
        for batch_l, batch_u in tqdm(zip(labeled_loader, unlabeled_loader)):
            optimizer_A.zero_grad()
            optimizer_B.zero_grad()

            image_l, label_l = fetch_data(batch_l)
            image_u = fetch_data(batch_u, labeled=False)
            image = torch.cat([image_l, image_u], dim=0)
            tmp_bs = image.shape[0] // 2

            if args.mixed_precision:
                with autocast():
                    # ============================================
                    # SECTION D: Modified Forward Pass with Validation
                    # ============================================
                    
                    outA = model_A(image)  # dict: seg_logits, seg_probs, noise, feat
                    outB = model_B(image)
                    del image

                    # split labeled / unlabeled with dict access
                    A_l = {k: v[:tmp_bs] for k, v in outA.items()}
                    A_u = {k: v[tmp_bs:] for k, v in outA.items()}
                    B_l = {k: v[:tmp_bs] for k, v in outB.items()}
                    B_u = {k: v[tmp_bs:] for k, v in outB.items()}

                    # NEW: Validate critical tensors before further processing
                    # Reason: Early detection of NaN/Inf values from noise training
                    validate_tensors(
                        A_u['seg_logits'], A_u['noise'],
                        B_u['seg_logits'], B_u['noise'],
                        name=f"Epoch{epoch_num}_Forward"
                    )

                    # Update noise buffer
                    noise_buffer.update(A_u['noise'], B_u['noise'])
                    
                    # NEW: Get adaptive alpha with min_alpha bound
                    # OLD CODE: Direct alpha usage without bounds
                    # alpha_A, alpha_B = noise_buffer.get_adaptive_alpha(
                    #     A_u['noise'], B_u['noise'], alpha_max=args.alpha_noise
                    # )
                    
                    # NEW CODE: With min_alpha parameter for stability
                    alpha_A, alpha_B = noise_buffer.get_adaptive_alpha(
                        A_u['noise'], B_u['noise'], 
                        alpha_max=args.alpha_noise,
                        min_alpha=args.min_alpha
                    )

                    # Keep global alpha for logging
                    alpha_global = get_alpha(epoch_num)

                    # supervised segmentation (unchanged)
                    loss_sup = loss_func_A(A_l['seg_logits'], label_l) + loss_func_B(B_l['seg_logits'], label_l)

                    # ============================================
                    # NEW: Structured noise loss call
                    # ============================================
                    
                    # OLD CODE: Direct call with many parameters (error-prone)
                    # L_sup_noise, L_u_dis, L_cons = noise_losses(
                    #     pA_l=A_l['seg_probs'], pB_l=B_l['seg_probs'], y_l=label_l,
                    #     nA_l=A_l['noise'], nB_l=B_l['noise'],
                    #     pA_u=A_u['seg_probs'], pB_u=B_u['seg_probs'],
                    #     nA_u=A_u['noise'], nB_u=B_u['noise'],
                    #     w_l=args.w_sup_noise, w_u=args.w_u_noise, w_cons=args.w_noise_cons
                    # )
                    
                    # NEW CODE: Structured call with dictionaries (cleaner, safer)
                    labeled_dict = {
                        'pA_l': A_l['seg_probs'],
                        'pB_l': B_l['seg_probs'],
                        'nA_l': A_l['noise'],
                        'nB_l': B_l['noise'],
                        'y_l': label_l
                    }
                    unlabeled_dict = {
                        'pA_u': A_u['seg_probs'],
                        'pB_u': B_u['seg_probs'],
                        'nA_u': A_u['noise'],
                        'nB_u': B_u['noise']
                    }
                    
                    L_sup_noise, L_u_dis, L_cons = noise_losses.forward_structured(
                        labeled_dict=labeled_dict,
                        unlabeled_dict=unlabeled_dict,
                        weights={
                            'w_l': args.w_sup_noise,
                            'w_u': args.w_u_noise,
                            'w_cons': args.w_noise_cons
                        }
                    )

                    # ============================================
                    # NEW: Safe cross-correction with scaling
                    # ============================================
                    
                    # OLD CODE: Direct subtraction without scaling
                    # A_logits_ref_u = A_u['seg_logits'] - alpha_B * B_u['noise']
                    # B_logits_ref_u = B_u['seg_logits'] - alpha_A * A_u['noise']
                    
                    # NEW CODE: Safe cross-correction with noise_safe_mode
                    A_logits_ref_u = cross_correct_logits(
                        A_u['seg_logits'], B_u['noise'], 
                        alpha=alpha_B, safe_mode=args.noise_safe_mode
                    )
                    B_logits_ref_u = cross_correct_logits(
                        B_u['seg_logits'], A_u['noise'],
                        alpha=alpha_A, safe_mode=args.noise_safe_mode
                    )

                    # build refined logits for full batch
                    A_logits_ref = torch.cat([A_l['seg_logits'], A_logits_ref_u], dim=0)
                    B_logits_ref = torch.cat([B_l['seg_logits'], B_logits_ref_u], dim=0)

                    # refined pseudo-labels
                    max_A = refined_pseudo_from(B_logits_ref)  # target for A
                    max_B = refined_pseudo_from(A_logits_ref)  # target for B

                    weight_A = diffdw.cal_weights(A_l['seg_logits'].detach(), label_l.detach())
                    weight_B = distdw.get_ema_weights(B_u['seg_probs'].detach())

                    loss_func_A.update_weight(weight_A)
                    loss_func_B.update_weight(weight_B)
                    cps_loss_func_A.update_weight(weight_A)
                    cps_loss_func_B.update_weight(weight_B)

                    # CPS with refined pseudo-labels
                    loss_cps = cps_loss_func_A(A_logits_ref, max_A) + cps_loss_func_B(B_logits_ref, max_B)
                    
                    # total loss with noise losses
                    loss = loss_sup + cps_w * loss_cps + (L_sup_noise + L_u_dis + L_cons)

                # ============================================
                # SECTION E: Modified Backward Pass with Gradient Clipping
                # ============================================
                
                # OLD CODE: No gradient clipping
                # amp_grad_scaler.scale(loss).backward()
                # amp_grad_scaler.step(optimizer_A)
                # amp_grad_scaler.step(optimizer_B)
                # amp_grad_scaler.update()
                
                # NEW CODE: With gradient clipping for stability
                amp_grad_scaler.scale(loss).backward()

                # Apply gradient clipping before optimizer step
                clip_gradients(model_A, amp_grad_scaler, optimizer_A, max_norm=args.grad_clip)
                clip_gradients(model_B, amp_grad_scaler, optimizer_B, max_norm=args.grad_clip)

                amp_grad_scaler.step(optimizer_A)
                amp_grad_scaler.step(optimizer_B)
                amp_grad_scaler.update()

            else:
                raise NotImplementedError

            loss_list.append(loss.item())
            loss_sup_list.append(loss_sup.item())
            loss_cps_list.append(loss_cps.item())
            loss_sup_noise_list.append(L_sup_noise.item())
            loss_u_dis_list.append(L_u_dis.item())
            loss_cons_list.append(L_cons.item())

        # ============================================
        # SECTION F: Enhanced Logging
        # ============================================
        
        # Collect monitoring metrics every epoch
        if epoch_num % 5 == 0:  # Log more frequently for noise metrics
            with torch.no_grad():
                # Monitor noise statistics
                noise_mean_A = A_u['noise'].mean().item()
                noise_mean_B = B_u['noise'].mean().item()
                noise_std_A = A_u['noise'].std().item()
                noise_std_B = B_u['noise'].std().item()
                
                # Monitor actual alpha values
                if isinstance(alpha_A, torch.Tensor):
                    alpha_mean = alpha_A.mean().item()
                    alpha_std = alpha_A.std().item()
                else:
                    alpha_mean = alpha_A
                    alpha_std = 0.0
                
                # Monitor pseudo-label agreement
                pseudo_A = refined_pseudo_from(B_logits_ref)
                pseudo_B = refined_pseudo_from(A_logits_ref)
                agreement = (pseudo_A == pseudo_B).float().mean().item()
                
                # Monitor buffer stats
                buffer_stats = noise_buffer.get_buffer_stats()

        # Basic tensorboard logging (unchanged)
        writer.add_scalar('lr', get_lr(optimizer_A), epoch_num)
        writer.add_scalar('cps_w', cps_w, epoch_num)
        writer.add_scalar('loss/loss', np.mean(loss_list), epoch_num)
        writer.add_scalar('loss/sup', np.mean(loss_sup_list), epoch_num)
        writer.add_scalar('loss/cps', np.mean(loss_cps_list), epoch_num)
        writer.add_scalar('loss/sup_noise', np.mean(loss_sup_noise_list), epoch_num)
        writer.add_scalar('loss/u_disagree', np.mean(loss_u_dis_list), epoch_num)
        writer.add_scalar('loss/noise_cons', np.mean(loss_cons_list), epoch_num)
        writer.add_scalar('noise/alpha_global', alpha_global, epoch_num)
        
        # NEW: Enhanced tensorboard logging
        writer.add_scalar('noise/mean_A', noise_mean_A, epoch_num)
        writer.add_scalar('noise/mean_B', noise_mean_B, epoch_num)
        writer.add_scalar('noise/std_A', noise_std_A, epoch_num)
        writer.add_scalar('noise/std_B', noise_std_B, epoch_num)
        writer.add_scalar('alpha/mean', alpha_mean, epoch_num)
        writer.add_scalar('alpha/std', alpha_std, epoch_num)
        writer.add_scalar('pseudo/agreement', agreement, epoch_num)
        writer.add_scalar('buffer/filled_ratio', buffer_stats.get('filled_ratio', 0), epoch_num)
        
        # OLD LOGGING: Basic loss only
        # logging.info(f'epoch {epoch_num} : loss : {np.mean(loss_list)}')
        # logging.info(f'     noise_losses: sup{np.mean(loss_sup_noise_list):.4f}, dis{np.mean(loss_u_dis_list):.4f}, cons{np.mean(loss_cons_list):.4f}')
        
        # NEW LOGGING: Comprehensive training metrics
        logging.info(f'epoch {epoch_num} : loss : {np.mean(loss_list):.4f}')
        logging.info(f'  noise stats - A(μ={noise_mean_A:.3f},σ={noise_std_A:.3f}), '
                     f'B(μ={noise_mean_B:.3f},σ={noise_std_B:.3f})')
        logging.info(f'  alpha: μ={alpha_mean:.3f}, σ={alpha_std:.3f}, '
                     f'pseudo_agree={agreement:.3f}')
        logging.info(f'  noise_losses: sup={np.mean(loss_sup_noise_list):.4f}, '
                     f'dis={np.mean(loss_u_dis_list):.4f}, cons={np.mean(loss_cons_list):.4f}')
        
        # Log adaptive alpha information
        if isinstance(alpha_A, torch.Tensor):
            logging.info(f'     adaptive alpha: A={alpha_A.mean().item():.4f}, B={alpha_B.mean().item():.4f}')
        else:
            logging.info(f'     adaptive alpha: A={alpha_A:.4f}, B={alpha_B:.4f}')
            
        logging.info(f"     Class Weights A: {print_func(weight_A)}, lr: {get_lr(optimizer_A)}")
        logging.info(f"     Class Weights B: {print_func(weight_B)}")
        
        # Update learning rate
        optimizer_A.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        optimizer_B.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        
        cps_w = get_current_consistency_weight(epoch_num)

        if epoch_num % 10 == 0:
            # Evaluation
            dice_list = [[] for _ in range(config.num_cls-1)]
            model_A.eval()
            model_B.eval()
            dice_func = SoftDiceLoss(smooth=1e-8)
            for batch in tqdm(eval_loader):
                with torch.no_grad():
                    image, gt = fetch_data(batch)
                    # evaluation uses return_dict=False
                    output_A = model_A(image, return_dict=False)
                    output_B = model_B(image, return_dict=False)
                    output = (output_A + output_B) / 2.0
                    del image

                    shp = output.shape
                    gt = gt.long()
                    y_onehot = torch.zeros(shp).cuda()
                    y_onehot.scatter_(1, gt, 1)

                    x_onehot = torch.zeros(shp).cuda()
                    output = torch.argmax(output, dim=1, keepdim=True).long()
                    x_onehot.scatter_(1, output, 1)

                    dice = dice_func(x_onehot, y_onehot, is_training=False)
                    dice = dice.data.cpu().numpy()
                    for i, d in enumerate(dice):
                        dice_list[i].append(d)

            dice_mean = []
            for dice in dice_list:
                dice_mean.append(np.mean(dice))
            logging.info(f'evaluation epoch {epoch_num}, dice: {np.mean(dice_mean)}, {dice_mean}')
            
            if np.mean(dice_mean) > best_eval:
                best_eval = np.mean(dice_mean)
                best_epoch = epoch_num
                save_path = os.path.join(snapshot_path, f'ckpts/best_model.pth')
                torch.save({
                    'A': model_A.state_dict(),
                    'B': model_B.state_dict()
                }, save_path)
                logging.info(f'saving best model to {save_path}')
            logging.info(f'\t best eval dice is {best_eval} in epoch {best_epoch}')
            # if epoch_num - best_epoch == config.early_stop_patience:
            #     logging.info(f'Early stop.')
            #     break
            
    writer.close()
    logging.info(f'Training completed. Best dice: {best_eval} at epoch {best_epoch}')