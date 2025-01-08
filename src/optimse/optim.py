import torch
import numpy as np

class ScheduledOptim(object):
  def __init__(self, optimizer, lr):

    self.lr = lr
    self.optimizer = optimizer

  def step(self):
    self.optimizer.step()

  def zero_grad(self):
    self.optimizer.zero_grad()

  def update_learning_rate(self, lr_multiplier):
    new_lr = self.lr * lr_multiplier
    for param_group in self.optimizer.param_groups:
      param_group['lr'] = new_lr