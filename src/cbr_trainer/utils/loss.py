
import torch
from torch import Tensor, BoolTensor
from torch.nn import Module

class TXent(Module):
    """
    Temperature-scaled Cross-entropy Loss
    The NT-Xent loss is minimized by a neural network that learns to embed data points 
    in a way that minimizes the distance between similar data points and maximizes the 
    distance between dissimilar data points. 
    """
    def __init__(self, temperature, **kwargs):
        super(TXent, self).__init__()
        self.temperature = temperature

    def forward(self, dist_arr: Tensor, labels: BoolTensor) -> Tensor:
        """
        :param dist_arr: Array of shape [n_nodes] containing distances (lower is better as opposed to score)
        :param labels: Labels array of shape [n_nodes] with 1 for positive class and 0 otherwise
        :return: Tensor loss
        """
        sim = - dist_arr / self.temperature
        assert sim.dim() == 1
        
        denom_log_sum_exp = torch.logsumexp(sim, dim=0)
        loss = - torch.logsumexp(sim[labels], dim=0) + denom_log_sum_exp
        
        return loss