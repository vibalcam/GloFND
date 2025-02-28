# Copyright (c) 2025 Vicente Balmaseda
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from typing import List, Optional, Union
import torch


class LambdaThreshold(torch.nn.Module):
    def __init__(
        self,
        device=None,
        glofnd_type='glofnd',
        N:int=50000,
        alpha:float=0.0,
        lr_lda:float=0.05,
        lda_beta1:float=0.9, 
        lda_beta2:float=0.98,
        normalize_grad:bool = False,
        opt_type="adam",
        eps:float=1e-8,
        start_update:int=15,
        lda_start:int=15, 
        init_quantile:bool = False,
        clip_grad_mult:Optional[float] = None,
        distributed = False,
        **kwargs,
    ) -> None:
        super(LambdaThreshold, self).__init__()

        if glofnd_type == 'none':
            glofnd_type = None
        self.glofnd_type = glofnd_type
        assert self.glofnd_type in ['glofnd', 'fnd', 'oracle', None], "Invalid glofnd type"

        self.learned = self.glofnd_type == 'glofnd'
        
        self.init_quantile = init_quantile
        self.normalize_grad = normalize_grad
        self.eps = eps

        if clip_grad_mult is not None and clip_grad_mult > 0:
            self.clip_lda_grad = lambda x, alpha: torch.clamp(x, min=-alpha * clip_grad_mult, max=alpha * clip_grad_mult)
        else:
            self.clip_lda_grad = lambda x, alpha: x

        # parameters
        assert start_update <= lda_start, "Start update must be less than lambda start"

        self.register_buffer("start_update", torch.tensor(start_update, requires_grad=False, device=device))
        self.register_buffer("lda_start", torch.tensor(lda_start, requires_grad=False, device=device))
        self.register_buffer("lda", torch.ones(N, requires_grad=False, device=device).reshape(-1, 1))
        self.register_buffer("v_grad", torch.zeros(N, requires_grad=False, device=device).reshape(-1, 1))
        self.register_buffer("lr_lda", torch.tensor(lr_lda, requires_grad=False, device=device))
        self.register_buffer("alpha", torch.tensor(alpha, requires_grad=False, device=device))
        
        self.opt_type = opt_type
        self.register_buffer("lda_beta1", torch.tensor(lda_beta1, requires_grad=False, device=device))
        if opt_type == "adam":
            self.register_buffer("m_grad", torch.zeros(N, requires_grad=False, device=device).reshape(-1, 1))
            self.register_buffer("lda_beta2", torch.tensor(lda_beta2, requires_grad=False, device=device))
        else:
            raise ValueError(f"Invalid optimizer type: {opt_type}")
        
        self.distributed = distributed
        if distributed:
            self.register_buffer("indices", torch.zeros(N, requires_grad=False, dtype=torch.bool, device=device))

    @torch.no_grad()
    def update(self, sim: Union[torch.Tensor, List[torch.Tensor]], idx: torch.Tensor, neg_self_mask: torch.Tensor, epoch: int):
        assert epoch is not None and epoch >= 0, "Epoch must be provided and >= 0"
        if self.start_update > epoch:
            return

        if isinstance(sim, torch.Tensor):
            sim = sim.unsqueeze(-1).detach()
        else:
            sim = torch.stack(sim, dim=-1).detach()

        batch_size = sim.shape[0]
        num_negative_pairs = neg_self_mask[0].sum().int()
        neg_self_mask = neg_self_mask.int()

        # get lr_lda from lr scheduler if any
        lr_lda = self.lr_lda #/ self.total_epochs * (self.total_epochs - epoch)

        # get alpha from scheduler if any
        alpha = self.alpha
        
        with torch.inference_mode():
            # for logging purposes
            lda_quantile = [s.float()[neg_self_mask.bool()].view(batch_size,-1).quantile(1-alpha, dim=1, keepdim=True).type_as(self.lda) for s in sim.unbind(-1)]
            lda_quantile = torch.stack(lda_quantile, dim=-1).mean(-1)

            if not self.learned:
                return {
                    'lda_quantiles': lda_quantile.mean().detach()
                }

            if epoch == self.start_update and self.init_quantile:
                lda = lda_quantile
            else:
                # update lambdas
                lda = self.lda[idx]

                # compute gradients
                g_mask = (sim > lda.unsqueeze(-1)).int() * neg_self_mask.unsqueeze(-1)
                g_mask = g_mask.sum(-1)
                lda_grad = self.alpha - g_mask.sum(1, keepdim=True) / (sim.shape[-1] * num_negative_pairs)

                # normalize gradients by max value
                if self.normalize_grad:
                    # 0.01 max lda, same as 0.01 lr
                    lda_grad = lda_grad / self.alpha

                # clip gradients
                lda_grad = self.clip_lda_grad(lda_grad, alpha)

                if self.opt_type == "adam":
                    current_epoch = epoch

                    # adam update
                    m_grad = self.m_grad[idx]
                    v_grad = self.v_grad[idx]
                    m_grad = self.lda_beta1 * m_grad + (1 - self.lda_beta1) * lda_grad
                    v_grad = self.lda_beta2 * v_grad + (1 - self.lda_beta2) * (lda_grad ** 2)
                    self.m_grad[idx] = m_grad.detach()
                    self.v_grad[idx] = v_grad.detach()

                    # bias correction
                    m_grad_hat = m_grad / (1 - self.lda_beta1 ** (current_epoch + 1))
                    v_grad_hat = v_grad / (1 - self.lda_beta2 ** (current_epoch + 1))

                    # update parameters
                    lda = (lda - lr_lda * m_grad_hat / (v_grad_hat ** 0.5 + self.eps)).clamp(min=-1, max=1)
                else:
                    raise ValueError(f"Invalid optimizer type: {self.opt_type}")

            self.lda[idx] = lda.detach()

            if self.distributed:
                self.indices[idx] = True

        return {
            'lda_quantiles': lda_quantile.mean().detach()
        }

    @torch.no_grad()
    def get_mask(self, sim: torch.Tensor, idx: Optional[torch.Tensor], neg_self_mask: torch.Tensor, epoch:int, labels:Optional[torch.Tensor]=None):
        """
        Returns the mask to filter self similarities and similarities above the learned threshold.
        It also returns the number of non-filtered negatives.
        """
        neg_self_mask_sum = neg_self_mask.sum(1, keepdim=True)
        if self.glofnd_type is None:
            return neg_self_mask, neg_self_mask_sum, {}
        
        if self.learned: # use learned threshold
            if idx is None:
                lda = self.lda
            else:
                lda = self.lda[idx]
        elif self.glofnd_type == 'fnd': # use minibatch quantile
            lda = sim.float()[neg_self_mask.bool()].view(sim.shape[0],-1).quantile(1-self.alpha, dim=1, keepdim=True).type_as(self.lda)
        elif self.glofnd_type == 'oracle': # use true false negatives from labels
            assert labels is not None, "Oracle requires labels"
        else:
            raise ValueError(f"Invalid glofnd type: {self.glofnd_type}")
        
        # compute mask
        neg_self_mask = neg_self_mask.float().detach()
        
        if labels is not None:
            # 1 if same class (false negative), 0 if different
            if labels.dim() == 1:
                true_mask = (labels[None, :] == labels[:, None]).float().repeat(1,2)
            else:
                true_mask = labels.float() 
            true_mask *= neg_self_mask

        # get masks for filtering, 1 keep, 0 filter
        if self.glofnd_type == 'oracle':
            mask = 1-true_mask
        else:
            mask = (sim <= lda).float()
        mask *= neg_self_mask
        # sum kept
        log_sum_mask = mask.sum(1, keepdim=True)

        # if no alpha or before start, return full negative mask
        if self.lda_start > epoch:
            return neg_self_mask, neg_self_mask_sum, {
                'num_negatives_perc': log_sum_mask / neg_self_mask_sum,
                'num_filtered_perc': neg_self_mask_sum / neg_self_mask_sum,
            }

        mask_weighted = mask.float().detach().clone()

        log_dict = {}
        # compute precision and recall
        if labels is not None:
            with torch.inference_mode():
                # 1 if false negative
                mask_inv = (1 - mask) * neg_self_mask
                log_dict['mask_inv'] = mask_inv
                log_dict['true_mask'] = true_mask

        mask_sum = mask.sum(1, keepdim=True).detach()
        mask_weighted_sum = mask_weighted.sum(1, keepdim=True)
        log_dict.update({
            'num_negatives_perc': log_sum_mask / neg_self_mask_sum,
            'num_filtered_perc': mask_sum / neg_self_mask_sum,
        })
        return mask_weighted, mask_weighted_sum, log_dict
