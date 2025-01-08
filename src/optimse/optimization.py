import math
import torch
from torch.optim import Optimizer


def get_lr(group, step):
    lr, warmup, train_steps = group['lr'], group['warmup'], group['train_steps']

    if step < warmup:
        return lr * (train_steps - step) / train_steps

    return lr * step / warmup


class AdamWeightDecayOptimizer(Optimizer):
    def __init__(self, params, lr=5e-5, warmup=10000, train_steps=100000, weight_decay=0.01, clip=1.0, betas=(0.9, 0.999), eps=1e-6):

        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, warmup=warmup,
                        train_steps=train_steps, weight_decay=weight_decay, clip=clip)
        super(AdamWeightDecayOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamWeightDecayOptimizer, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                if group['clip'] != 0.:
                    torch.nn.utils.clip_grad_norm_(p, group['clip'])

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = exp_avg / (exp_avg_sq.sqrt() + group['eps'])

                if group['weight_decay'] != 0.:
                    update += group['weight_decay'] * p.data

                update_with_lr = get_lr(group, state['step']) * update
                p.data.add_(-update_with_lr)

        return loss
