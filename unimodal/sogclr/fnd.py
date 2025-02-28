from typing import List, Optional, Union
import torch
import torch.nn as nn
from sogclr.utils import concat_all_gather


class LambdaThreshold(nn.Module):
    def __init__(
        self,
        glofnd_type:str,
        N:int,
        alpha:float=0.0,
        lr_lda:float=0.01,
        lda_beta1:float=0.9, 
        lda_beta2:float=0.98,
        normalize_grad:bool = False,
        opt_type="adam",
        eps:float=1e-8,
        start_update:int=15,
        lda_start:int=15, 
        u_warmup:int=0,
        init_quantile:bool = False,
        clip_grad_mult:Optional[float] = None,
        distributed = False,
    ) -> None:
        super(LambdaThreshold, self).__init__()

        if glofnd_type == 'none':
            glofnd_type = None
        self.glofnd_type = glofnd_type
        assert self.glofnd_type in ['glofnd', 'uglofnd', 'fnd', 'oracle', 'disabled', None], "Invalid glofnd type"

        self.learned = self.glofnd_type in ['glofnd', 'uglofnd']
        
        self.init_quantile = init_quantile
        self.normalize_grad = normalize_grad
        self.eps = eps

        if clip_grad_mult is not None:
            self.clip_lda_grad = lambda x, alpha: torch.clamp(x, min=-alpha * clip_grad_mult, max=alpha * clip_grad_mult)
        else:
            self.clip_lda_grad = lambda x, alpha: x
        
        self.opt_type = opt_type
        self.distributed = distributed
        if distributed:
            self.register_buffer("indices", torch.zeros(N, requires_grad=False, dtype=torch.bool))

        assert not (self.distributed and torch.distributed.get_world_size() > 1 and (self.glofnd_type == 'uglofnd' or u_warmup > 0)), "Cannot use uglofnd with distributed"
        assert start_update <= lda_start, "Start update must be less than lambda start"
        assert self.opt_type == "adam", "Only adam optimizer is supported"

        # parameters
        self.register_buffer("start_update", torch.tensor(start_update, requires_grad=False))
        self.register_buffer("u_warmup", torch.tensor(u_warmup, requires_grad=False))
        self.register_buffer("lda_start", torch.tensor(lda_start, requires_grad=False))
        self.register_buffer("lr_lda", torch.tensor(lr_lda, requires_grad=False))
        self.register_buffer("alpha", torch.tensor(alpha, requires_grad=False))
        self.register_buffer("lda_beta1", torch.tensor(lda_beta1, requires_grad=False))
        self.register_buffer("lda_beta2", torch.tensor(lda_beta2, requires_grad=False))
        if self.glofnd_type in ['uglofnd']:
            self.register_buffer("lda", torch.ones(1, requires_grad=False).reshape(-1, 1))
            self.register_buffer("v_grad", torch.zeros(1, requires_grad=False).reshape(-1, 1))
            self.register_buffer("m_grad", torch.zeros(1, requires_grad=False).reshape(-1, 1))
        else:
            self.register_buffer("lda", torch.ones(N, requires_grad=False).reshape(-1, 1))
            self.register_buffer("v_grad", torch.zeros(N, requires_grad=False).reshape(-1, 1))
            self.register_buffer("m_grad", torch.zeros(N, requires_grad=False).reshape(-1, 1))

    @torch.no_grad()
    def update(self, sim: Union[torch.Tensor, List[torch.Tensor]], idx: torch.Tensor, neg_self_mask: torch.Tensor, epoch: int):
        assert epoch is not None and epoch >= 0, "Epoch must be provided and >= 0"
        if self.start_update > epoch:
            return {}

        if isinstance(sim, torch.Tensor):
            sim = sim.unsqueeze(-1).detach()
        else:
            sim = torch.stack(sim, dim=-1).detach()

        # device = sim1.device
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
                assert self.glofnd_type not in ['uglofnd'], "Cannot use glofnd type uglofnd with init quantile"
                lda = lda_quantile
            else:
                # update lambdas
                if self.glofnd_type == 'uglofnd' or epoch < self.start_update + self.u_warmup:
                    idx = [0]
                lda = self.lda[idx]

                # compute gradients
                if self.glofnd_type == 'uglofnd' or epoch < self.start_update + self.u_warmup:
                    g_mask = (sim > lda.unsqueeze(-1)).int() * neg_self_mask.unsqueeze(-1)
                    g_mask = g_mask.sum(-1)
                    lda_grad = self.alpha - g_mask.sum([0,1], keepdim=True) / (sim.shape[-1] * num_negative_pairs * batch_size)
                else:
                    g_mask = (sim > lda.unsqueeze(-1)).int() * neg_self_mask.unsqueeze(-1)
                    g_mask = g_mask.sum(-1)
                    lda_grad = self.alpha - g_mask.sum(1, keepdim=True) / (sim.shape[-1] * num_negative_pairs)

                # normalize gradients by max value
                if self.normalize_grad:
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
                    if epoch < self.start_update + self.u_warmup:
                        self.m_grad[:] = m_grad.detach()
                        self.v_grad[:] = v_grad.detach()
                    else:
                        self.m_grad[idx] = m_grad.detach()
                        self.v_grad[idx] = v_grad.detach()

                    # bias correction
                    m_grad_hat = m_grad / (1 - self.lda_beta1 ** (current_epoch + 1))
                    v_grad_hat = v_grad / (1 - self.lda_beta2 ** (current_epoch + 1))

                    # update parameters
                    lda = (lda - lr_lda * m_grad_hat / (v_grad_hat ** 0.5 + self.eps)).clamp(min=-1, max=1)
                else:
                    raise ValueError(f"Invalid optimizer type: {self.opt_type}")

            if epoch < self.start_update + self.u_warmup:
                self.lda[:] = lda.detach()
            else:
                self.lda[idx] = lda.detach()

            if self.distributed:
                self.indices[idx] = True

        return {
            'lda_quantiles': lda_quantile.mean().detach()
        }
    
    @torch.no_grad()
    def dist_update_params(self):
        if not self.learned or not self.distributed or self.glofnd_type == 'uglofnd' or self.u_warmup > 0:
            return

        index_large = concat_all_gather(self.indices.nonzero().squeeze())
        v_grad_large = concat_all_gather(self.v_grad[self.indices])
        self.v_grad[index_large] = v_grad_large.detach()
        if self.opt_type == "adam":
            m_grad_large = concat_all_gather(self.m_grad[self.indices])
            self.m_grad[index_large] = m_grad_large.detach()
        lda_large = concat_all_gather(self.lda[self.indices])
        self.lda[index_large] = lda_large.detach()

        self.indices[:] = False

        print("Updated params")

    @torch.no_grad()
    def get_mask(self, sim: torch.Tensor, idx: torch.Tensor, neg_self_mask: torch.Tensor, epoch:int, labels:Optional[torch.Tensor]=None, mask:Optional[torch.Tensor]=None):
        """
        Returns the mask to filter self similarities and similarities above the learned threshold.
        It also returns the number of non-filtered negatives.
        """
        if self.glofnd_type is None:
            neg_self_mask_sum = neg_self_mask.sum(1, keepdim=True)
            return neg_self_mask, neg_self_mask_sum, {}
        
        if self.learned: # use learned threshold
            if self.glofnd_type == 'uglofnd' or epoch < self.start_update + self.u_warmup:
                lda = self.lda[[0]]
            else:
                lda = self.lda[idx]
        elif self.glofnd_type == 'fnd': # use minibatch quantile
            lda = sim.float()[neg_self_mask.bool()].view(sim.shape[0],-1).quantile(1-self.alpha, dim=1, keepdim=True).type_as(self.lda)
        elif self.glofnd_type == 'oracle': # use true false negatives from labels
            assert labels is not None, "Oracle requires labels"
        elif self.glofnd_type == 'disabled':
            assert mask is not None, "Disabled requires pseudo labels"
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
        if mask is not None:
            mask = mask.float()
        elif self.glofnd_type == 'oracle':
            mask = 1-true_mask
        else:
            mask = (sim <= lda).float()
        mask *= neg_self_mask
        # sum kept
        log_sum_mask = mask.sum(1, keepdim=True)

        # if no alpha or before start, return full negative mask
        if self.lda_start > epoch:
            neg_self_mask_sum = neg_self_mask.sum(1, keepdim=True)
            return neg_self_mask, neg_self_mask_sum, {
                'num_negatives_perc': log_sum_mask,
                'num_filtered_perc': neg_self_mask_sum,
            }

        mask_weighted = mask.float().detach().clone()
        log_dict = {}
        if labels is not None:
            with torch.inference_mode():
                # 1 if false negative
                mask_inv = (1 - mask) * neg_self_mask
                log_dict['mask_inv'] = mask_inv
                log_dict['true_mask'] = true_mask

        mask_sum = mask.sum(1, keepdim=True).detach()
        mask_weighted_sum = mask_weighted.sum(1, keepdim=True)
        log_dict.update({
            'num_negatives_perc': log_sum_mask,
            'num_filtered_perc': mask_sum,
        })
        return mask_weighted, mask_weighted_sum, log_dict
