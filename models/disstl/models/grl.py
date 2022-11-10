import torch
import torch.nn as nn


class GradientReversalFunction(torch.autograd.Function):
    """
    Simple gradient reversal function that permits a grad_weight that is adjustable during training
    """

    @staticmethod
    def forward(ctx, x, grad_weight: float):
        ctx.grad_weight = grad_weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # see here for discussion on setting breakpoints
        # in this method: https://discuss.pytorch.org/t/custom-backward-breakpoint-doesnt-get-hit/6473/7
        return -ctx.grad_weight * grad_output, None


class GradientReversalModule(nn.Module):
    """
    Convenience module for adding a GradientReversalFunction to an existing nn.Sequential module

    Usage: grl_mlp = nn.Sequential(GradientReversalModule(), nn.Linear(), ...)

        In practice this could look like this:
        embedding_network = <some embedding network>
        grl_mlp = nn.Sequential(GradientReversalModule(grad_weight=grad_weight), nn.Linear(), ...)

        output = grl_mlp(embedding_network(x))
        loss = Loss(output, mlp_target)
        loss.backward()

        The effect is this:
        * with grad_weight = 0.0:
             the weights of the embedding_network will not be updated in response to the classification loss
        * With grad_weight = 1.0:
             the weights of the embedding_network will be updated to cause the embeddings to be bad for
             classification
        * with grad_weight = -1.0:
             the weights of the embedding_network will be updated to help minimize the classification loss
    """

    def __init__(self, grad_weight: float = 1.0):
        super().__init__()
        self.grad_weight = grad_weight

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.grad_weight)
