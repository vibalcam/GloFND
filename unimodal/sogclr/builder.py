# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union
from sogclr.fnd import LambdaThreshold
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
from sogclr.utils import concat_all_gather, all_gather_layer


# Function to create and log histograms as high-quality images
def log_hist_as_image(data, title, bins=64, dpi=300):
    # Create a figure with high DPI for better quality
    fig = plt.figure(figsize=(6, 4), dpi=dpi)
    plt.hist(data, bins=bins, density=True)
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot to an image and log with wandb
    wandb.log({title: wandb.Image(fig)}, commit=False)
    plt.close(fig)


class SimCLR(nn.Module):
    """
    Build a SimCLR-based model with a base encoder, and two MLPs
   
    """
    def __init__(self,
        base_encoder,
        dim=256,
        mlp_dim=2048,
        T=0.1,
        loss_type='dcl',
        glofnd_type = 'none',
        N=50000,
        num_proj_layers=2,
        device=None,
        alpha:List[float]=[0.0],
        lr_lda=1.0,
        lda_start=20,
        init_quantile=True,
        clip_grad_mult=None,
        distributed=False,
        args=None,
        **kwargs
    ):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(SimCLR, self).__init__()
        self.T = T
        self.N = N
        self.loss_type = loss_type
        self.counter = 0
        
        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        
        # build non-linear projection heads
        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        # sogclr 
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device 
        
        # for DCL
        self.u = torch.zeros(len(alpha), N, 1) #.to(self.device) 
        self.LARGE_NUM = 1e9

        # for GloFND
        print(f"Using lambda threshold with {loss_type} and glofnd type {glofnd_type}")
        self.lambda_threshold = [LambdaThreshold(distributed=distributed, N=N, alpha=a, lr_lda=lr_lda, lda_start=lda_start, init_quantile=init_quantile, clip_grad_mult=clip_grad_mult, u_warmup=args.u_warmup, glofnd_type=glofnd_type, start_update=args.start_update, **kwargs) for a in alpha]
        self.lambda_threshold = nn.ModuleList(self.lambda_threshold)

    def train(self, mode = True):
        r = super().train(mode)
        return r

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass
    
    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def dynamic_contrastive_loss(
        self,
        i,
        u,
        hidden1,
        hidden2,
        index=None,
        gamma=0.9,
        distributed=True,
        **kwargs
    ):
        # Get (normalized) hidden1 and hidden2.
        hidden1, hidden2 = F.normalize(hidden1, p=2, dim=1), F.normalize(hidden2, p=2, dim=1)
        batch_size = hidden1.shape[0]
        
        # Gather hidden1/hidden2 across replicas and create local labels.
        if distributed:  
           hidden1_large = torch.cat(all_gather_layer.apply(hidden1), dim=0) # why concat_all_gather()
           hidden2_large =  torch.cat(all_gather_layer.apply(hidden2), dim=0)
           enlarged_batch_size = hidden1_large.shape[0]

           labels_idx = (torch.arange(batch_size, dtype=torch.long) + batch_size  * torch.distributed.get_rank()).to(self.device) 
           labels = F.one_hot(labels_idx, enlarged_batch_size*2).to(self.device) 
           masks  = F.one_hot(labels_idx, enlarged_batch_size).to(self.device) 
           batch_size = enlarged_batch_size
        else:
           hidden1_large = hidden1
           hidden2_large = hidden2
           labels = F.one_hot(torch.arange(batch_size, dtype=torch.long), batch_size * 2).to(self.device) 
           masks  = F.one_hot(torch.arange(batch_size, dtype=torch.long), batch_size).to(self.device) 

        logits_aa = torch.matmul(hidden1, hidden1_large.T)
        logits_aa = logits_aa - masks * self.LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.T)
        logits_bb = logits_bb - masks * self.LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.T)
        logits_ba = torch.matmul(hidden2, hidden1_large.T)

        #  SogCLR
        neg_mask = 1-labels
        logits_ab_aa = torch.cat([logits_ab, logits_aa ], 1)
        logits_ba_bb = torch.cat([logits_ba, logits_bb ], 1)
      
        neg_logits1 = torch.exp(logits_ab_aa /self.T)*neg_mask   #(B, 2B)
        neg_logits2 = torch.exp(logits_ba_bb /self.T)*neg_mask

        # u init    
        if u[index.cpu()].sum() == 0:
            gamma = 1
            
        u1 = (1 - gamma) * u[index.cpu()].cuda() + gamma * torch.sum(neg_logits1, dim=1, keepdim=True)/(2*(batch_size-1))
        u2 = (1 - gamma) * u[index.cpu()].cuda() + gamma * torch.sum(neg_logits2, dim=1, keepdim=True)/(2*(batch_size-1))

        # this sync on all devices (since "hidden" are gathering from all devices)
        if distributed:
           u1_large = concat_all_gather(u1)
           u2_large = concat_all_gather(u2)
           index_large = concat_all_gather(index)
           u[index_large.cpu()] = (u1_large.detach().cpu() + u2_large.detach().cpu())/2 
        else:
           u[index.cpu()] = (u1.detach().cpu() + u2.detach().cpu())/2 

        p_neg_weights1 = (neg_logits1/u1).detach()
        p_neg_weights2 = (neg_logits2/u2).detach()

        def softmax_cross_entropy_with_logits(labels, logits, weights):
            expsum_neg_logits = torch.sum(weights*logits, dim=1, keepdim=True)/(2*(batch_size-1))
            normalized_logits = logits - expsum_neg_logits
            return -torch.sum(labels * normalized_logits, dim=1)

        loss_a = softmax_cross_entropy_with_logits(labels, logits_ab_aa, p_neg_weights1)
        loss_b = softmax_cross_entropy_with_logits(labels, logits_ba_bb, p_neg_weights2)
        loss = (loss_a + loss_b).mean()

        return loss, {}
    
    def glofnd_dynamic_contrastive_loss(
        self,
        i,
        u,
        lda,
        hidden1,
        hidden2,
        index=None,
        gamma=0.9,
        distributed=True,
        epoch=None,
        class_labels=None,
        **kwargs
    ):
        # Get (normalized) hidden1 and hidden2.
        hidden1, hidden2 = F.normalize(hidden1, p=2, dim=1), F.normalize(hidden2, p=2, dim=1)
        batch_size = hidden1.shape[0]
        
        # Gather hidden1/hidden2 across replicas and create local labels
        if distributed:  
            hidden1_large = torch.cat(all_gather_layer.apply(hidden1), dim=0) # why concat_all_gather()
            hidden2_large =  torch.cat(all_gather_layer.apply(hidden2), dim=0)
            enlarged_batch_size = hidden1_large.shape[0]

            labels_idx = (torch.arange(batch_size, dtype=torch.long) + batch_size  * torch.distributed.get_rank()).to(self.device) 
            labels = F.one_hot(labels_idx, enlarged_batch_size*2).to(self.device) 
            masks  = F.one_hot(labels_idx, enlarged_batch_size).to(self.device) 
            batch_size = enlarged_batch_size
        else:
            hidden1_large = hidden1
            hidden2_large = hidden2
            labels = F.one_hot(torch.arange(batch_size, dtype=torch.long), batch_size * 2).to(self.device) 
            masks  = F.one_hot(torch.arange(batch_size, dtype=torch.long), batch_size).to(self.device) 
        neg_self_mask = 1 - labels
        neg_self_mask[:, batch_size:] -= masks

        logits_aa = torch.matmul(hidden1, hidden1_large.T)
        logits_aa = logits_aa - masks * self.LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.T)
        logits_bb = logits_bb - masks * self.LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.T)
        logits_ba = torch.matmul(hidden2, hidden1_large.T)

        #  SogCLR
        neg_mask = 1-labels
        # neg_mask = neg_self_mask
        logits_ab_aa = torch.cat([logits_ab, logits_aa ], 1)
        logits_ba_bb = torch.cat([logits_ba, logits_bb ], 1)
    
        neg_logits1 = torch.exp(logits_ab_aa /self.T)*neg_mask   #(B, 2B)
        neg_logits2 = torch.exp(logits_ba_bb /self.T)*neg_mask

        t_logits_ab_aa = logits_ab_aa.detach()
        t_logits_ba_bb = logits_ba_bb.detach()

        is_init = u[index.cpu()].sum() == 0

        # Compute GloFND mask
        log_update = lda.update(sim=[t_logits_ab_aa, t_logits_ba_bb], idx=index, neg_self_mask=neg_self_mask, epoch=epoch)
        mask_ab_aa, num_negatives_ab_aa, log_mask_ab_aa = lda.get_mask(t_logits_ab_aa, index, neg_self_mask, epoch=epoch, labels=class_labels, mask=mask_ab_aa)
        mask_ba_bb, num_negatives_ba_bb, log_mask_ba_bb = lda.get_mask(t_logits_ba_bb, index, neg_self_mask, epoch=epoch, labels=class_labels, mask=mask_ba_bb)

        # Apply GloFND mask
        neg_logits1 = neg_logits1 * mask_ab_aa
        neg_logits2 = neg_logits2 * mask_ba_bb

        # u init    
        if is_init:
            gamma = 1
            
        u1 = (1 - gamma) * u[index.cpu()].cuda() + gamma * torch.sum(neg_logits1, dim=1, keepdim=True)/num_negatives_ab_aa
        u2 = (1 - gamma) * u[index.cpu()].cuda() + gamma * torch.sum(neg_logits2, dim=1, keepdim=True)/num_negatives_ba_bb

        # this sync on all devices (since "hidden" are gathering from all devices)
        if distributed:
           u1_large = concat_all_gather(u1)
           u2_large = concat_all_gather(u2)
           index_large = concat_all_gather(index)
           u[index_large.cpu()] = (u1_large.detach().cpu() + u2_large.detach().cpu())/2 
        else:
           u[index.cpu()] = (u1.detach().cpu() + u2.detach().cpu())/2 

        p_neg_weights1 = (neg_logits1/u1).detach()
        p_neg_weights2 = (neg_logits2/u2).detach()

        def softmax_cross_entropy_with_logits(labels, logits, weights, num_negatives):
            expsum_neg_logits = torch.sum(weights*logits, dim=1, keepdim=True)/num_negatives
            normalized_logits = logits - expsum_neg_logits
            return -torch.sum(labels * normalized_logits, dim=1)

        loss_a = softmax_cross_entropy_with_logits(labels, logits_ab_aa, p_neg_weights1, num_negatives_ab_aa)
        loss_b = softmax_cross_entropy_with_logits(labels, logits_ba_bb, p_neg_weights2, num_negatives_ba_bb)
        loss = (loss_a + loss_b).mean()

        # compute logging info
        self.log(i, lda, log_update, log_mask_ab_aa, log_mask_ba_bb, neg_self_mask, index)
        log_update = {f"{k}/{i}":v for k,v in log_update.items()}
        log_update[f"loss/{i}"] = loss.item()

        return loss, log_update

    @torch.inference_mode()
    def log(self, i, lda, log_update, log_mask_ab_aa, log_mask_ba_bb, neg_self_mask, index):
        if 'num_negatives_perc' in log_mask_ab_aa:
            num_neg = (log_mask_ab_aa['num_negatives_perc'] + log_mask_ba_bb['num_negatives_perc']).sum(1) / (2 * neg_self_mask.sum(1))
            log_update['num_negatives_perc'] = num_neg.mean().item()
        if 'num_filtered_perc' in log_mask_ab_aa:
            num_fil = (log_mask_ab_aa['num_filtered_perc'] + log_mask_ba_bb['num_filtered_perc']).sum(1) / (2 * neg_self_mask.sum(1))
            log_update['num_filtered_perc'] = num_fil.mean().item()
        if 'mean_weights' in log_mask_ab_aa:
            mean_weight = (log_mask_ab_aa['mean_weights'] + log_mask_ba_bb['mean_weights']) / 2
            log_update['mean_weights'] = mean_weight.mean().item()
        if 'mask_inv' in log_mask_ab_aa:
            concat_mask_inv = torch.cat([log_mask_ab_aa['mask_inv'], log_mask_ba_bb['mask_inv']], 1)
            concat_true_mask = torch.cat([log_mask_ab_aa['true_mask'], log_mask_ba_bb['true_mask']], 1)
            true_positives = (concat_mask_inv * concat_true_mask).sum(1).float()
            filtered_recall = (true_positives + 1e-9) / (concat_true_mask.sum(1).float() + 1e-9)
            filtered_precision = (true_positives + 1e-9) / (concat_mask_inv.sum(1).float() + 1e-9)
            log_update['lda_recall'] = filtered_recall.mean().item()
            log_update['lda_precision'] = filtered_precision.mean().item()

        self.counter += 1
        if self.counter % 1000 == 0:
            self.counter = 0
            log_hist_as_image(lda.lda.detach().cpu().numpy(), "distributions/lda")
            if 'num_negatives_perc' in log_mask_ab_aa:
                log_hist_as_image(num_neg.detach().cpu().numpy(), "distributions/numNegatives")
            if 'num_filtered_perc' in log_mask_ab_aa:
                log_hist_as_image(num_fil.detach().cpu().numpy(), "distributions/numFiltered")
            if 'mask_inv' in log_mask_ab_aa:
                log_hist_as_image(filtered_recall.detach().cpu().numpy(), "distributions/filteredRecall")
                log_hist_as_image(filtered_precision.detach().cpu().numpy(), "distributions/filteredPrecision")

        lda_i = lda.lda
        v_grad = lda.v_grad
        if lda_i.shape[0] > 1:
            lda_i = lda_i[index]
            v_grad = v_grad[index]
        log_update.update({
            "lambda": lda_i.mean().item(),
            "lda_v_grad": v_grad.mean().item(),
            "lda_lr": lda.lr_lda.item(),
        })
    
    def dist_update_params(self):
        for lda in self.lambda_threshold:
            lda.dist_update_params()
    
    def forward(self, x1, x2, index, get_hidden=False, class_labels=None, **kwargs):
        """
        Input:
            x1: first views of images
            x2: second views of images
            index: index of image
            gamma: moving average of sogclr 
        Output:
            loss
        """
        # compute features
        h1 = self.base_encoder(x1)
        h2 = self.base_encoder(x2)
        losses = []
        logs_d = {}
        for i, lda in enumerate(self.lambda_threshold):
            u = self.u[i]
            if self.loss_type == 'dcl':
                if lda.glofnd_type is None:
                    loss, log_d = self.dynamic_contrastive_loss(i, u, h1, h2, index, **kwargs)
                else:
                    loss, log_d = self.glofnd_dynamic_contrastive_loss(i, u, lda, h1, h2, index, class_labels=class_labels, **kwargs)
            else:
                raise ValueError(f"Invalid loss type: {self.loss_type}")
            
            losses.append(loss)
            logs_d.update(log_d)

        loss = torch.mean(torch.stack(losses))
        
        if get_hidden:
            raise NotImplementedError
            # to get hidden would need to get from layer previous to fc
            return loss, log_d, [h1, h2]
        else:
            return loss, logs_d


class SimCLR_ResNet(SimCLR):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim, num_proj_layers=2):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc  # remove original fc layer
            
        # projectors
        # TODO: increase number of mlp layers 
        self.base_encoder.fc = self._build_mlp(num_proj_layers, hidden_dim, mlp_dim, dim)

        # todo remove fc and separate so we can get both outputs
