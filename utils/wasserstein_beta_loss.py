import torch
import torch.nn as nn
import torch.nn.functional as F


class WassersteinBetaLoss(nn.Module):
    """
    Wasserstein-2 distance loss for beta distributions with support for class weighting.
    
    Computes the analytical Wasserstein-2 distance between predicted and target
    beta distributions. This is more efficient and stable than sampling-based
    approaches for beta distribution learning.
    
    Args:
        reduction (str): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'
        eps (float): Small epsilon value to prevent numerical instability.
            Default: 1e-8
        alpha_min (float): Minimum value for alpha parameter to ensure valid beta.
            Default: 1e-3
        beta_min (float): Minimum value for beta parameter to ensure valid beta.
            Default: 1e-3
        weight (torch.Tensor, optional): A tensor of weights for each class (negative, positive).
            Shape: (2,) with [weight_for_class_0, weight_for_class_1].
            If None, all classes are weighted equally.
    
    Shape:
        - pred_alpha: (N, *) where N is batch size
        - pred_beta: (N, *) where N is batch size  
        - target_alpha: (N, *) where N is batch size
        - target_beta: (N, *) where N is batch size
        - labels (optional): (N,) class labels (0 for negative, 1 for positive). Required if weight is provided.
        - Output: scalar if reduction != 'none', otherwise (N, *)
    
    Example:
        >>> # Basic usage without class weights
        >>> loss_fn = WassersteinBetaLoss()
        >>> pred_alpha = torch.tensor([2.0, 3.0])
        >>> pred_beta = torch.tensor([1.0, 2.0])
        >>> target_alpha = torch.tensor([1.5, 2.5])
        >>> target_beta = torch.tensor([1.2, 1.8])
        >>> loss = loss_fn(pred_alpha, pred_beta, target_alpha, target_beta)
        
        >>> # Usage with class weights
        >>> class_weights = torch.tensor([1.0, 3.0])  # Higher weight for positive class
        >>> loss_fn = WassersteinBetaLoss(weight=class_weights)
        >>> labels = torch.tensor([0, 1])  # 0=no crash, 1=crash
        >>> loss = loss_fn(pred_alpha, pred_beta, target_alpha, target_beta, labels)
    """
    
    def __init__(self, reduction='mean', eps=1e-8, alpha_min=1e-3, beta_min=1e-3, weight=None):
        super(WassersteinBetaLoss, self).__init__()
        self.reduction = reduction
        self.eps = eps
        self.alpha_min = alpha_min
        self.beta_min = beta_min
        self.register_buffer('weight', weight)  # Use register_buffer to handle device movement
        
        # Validate reduction parameter
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got {reduction}")
    
    def forward(self, pred_alpha, pred_beta, target_alpha, target_beta, labels=None):
        """
        Compute Wasserstein-2 distance between predicted and target beta distributions.
        
        Args:
            pred_alpha (Tensor): Predicted alpha parameters of beta distribution
            pred_beta (Tensor): Predicted beta parameters of beta distribution
            target_alpha (Tensor): Target alpha parameters of beta distribution
            target_beta (Tensor): Target beta parameters of beta distribution
            labels (Tensor, optional): The class labels (0 for negative, 1 for positive).
                                      Shape: (batch_size,). Required if weight is provided.
            
        Returns:
            Tensor: Wasserstein-2 distance loss
        """
        # Check if weight is provided but labels is None
        if self.weight is not None and labels is None:
            raise ValueError("Labels must be provided when using class weights. "
                           "Either provide labels or set weight=None.")
        
        # Ensure parameters are valid (positive)
        pred_alpha = torch.clamp(pred_alpha, min=self.alpha_min)
        pred_beta = torch.clamp(pred_beta, min=self.beta_min)
        target_alpha = torch.clamp(target_alpha, min=self.alpha_min)
        target_beta = torch.clamp(target_beta, min=self.beta_min)
        
        # Compute means of beta distributions
        pred_mean = pred_alpha / (pred_alpha + pred_beta)
        target_mean = target_alpha / (target_alpha + target_beta)
        
        # Compute variances of beta distributions
        pred_sum = pred_alpha + pred_beta
        target_sum = target_alpha + target_beta
        
        pred_var = (pred_alpha * pred_beta) / (pred_sum**2 * (pred_sum + 1) + self.eps)
        target_var = (target_alpha * target_beta) / (target_sum**2 * (target_sum + 1) + self.eps)
        
        # Wasserstein-2 distance components
        mean_diff_squared = (pred_mean - target_mean)**2
        var_diff_squared = (torch.sqrt(pred_var + self.eps) - torch.sqrt(target_var + self.eps))**2
        
        # Total Wasserstein-2 distance
        wasserstein_dist = mean_diff_squared + var_diff_squared
        
        # Apply weighting only if both weight and labels are provided
        if self.weight is not None and labels is not None:
            # Use class-specific weights: weight[0] for class 0, weight[1] for class 1
            sample_weights = self.weight[labels.long()]
            wasserstein_dist = wasserstein_dist * sample_weights
        
        # Apply reduction
        if self.reduction == 'none':
            return wasserstein_dist
        elif self.reduction == 'mean':
            return torch.mean(wasserstein_dist)
        elif self.reduction == 'sum':
            return torch.sum(wasserstein_dist)
    
    def extra_repr(self):
        """String representation of the loss function parameters."""
        weight_str = f', weight={self.weight}' if self.weight is not None else ''
        return f'reduction={self.reduction}, eps={self.eps}, alpha_min={self.alpha_min}, beta_min={self.beta_min}{weight_str}'


class WassersteinBetaLossWithRegularization(WassersteinBetaLoss):
    """
    Extended Wasserstein-2 loss with regularization terms for beta distribution learning.
    
    Adds regularization to encourage well-behaved beta parameters and prevent
    extreme distributions that might be unrealistic for crash risk modeling.
    
    Args:
        reduction (str): Reduction method. Default: 'mean'
        eps (float): Numerical stability epsilon. Default: 1e-8
        alpha_min (float): Minimum alpha value. Default: 1e-3
        beta_min (float): Minimum beta value. Default: 1e-3
        weight (torch.Tensor, optional): A tensor of weights for each class (negative, positive).
            Shape: (2,) with [weight_for_class_0, weight_for_class_1].
            If None, all classes are weighted equally.
        reg_weight (float): Weight for regularization term. Default: 0.01
        target_concentration (float): Target concentration (alpha + beta) for regularization.
            Higher values encourage more confident predictions. Default: 2.0
    """
    
    def __init__(self, reduction='mean', eps=1e-8, alpha_min=1e-3, beta_min=1e-3, 
                 weight=None, reg_weight=0.01, target_concentration=2.0):
        super().__init__(reduction, eps, alpha_min, beta_min, weight)
        self.reg_weight = reg_weight
        self.target_concentration = target_concentration
    
    def forward(self, pred_alpha, pred_beta, target_alpha, target_beta, labels=None):
        """
        Compute Wasserstein-2 distance with regularization.
        
        Args:
            pred_alpha (Tensor): Predicted alpha parameters of beta distribution
            pred_beta (Tensor): Predicted beta parameters of beta distribution
            target_alpha (Tensor): Target alpha parameters of beta distribution
            target_beta (Tensor): Target beta parameters of beta distribution
            labels (Tensor, optional): The class labels (0 for negative, 1 for positive).
                                      Shape: (batch_size,). Required if weight is provided.
        
        Returns:
            Tensor: Total loss (Wasserstein distance + regularization)
        """
        # Base Wasserstein loss
        wasserstein_loss = super().forward(pred_alpha, pred_beta, target_alpha, target_beta, labels)
        
        # Regularization: encourage reasonable concentration
        pred_concentration = pred_alpha + pred_beta
        concentration_penalty = torch.mean((pred_concentration - self.target_concentration)**2)
        
        # Total loss
        total_loss = wasserstein_loss + self.reg_weight * concentration_penalty
        
        return total_loss
    
    def extra_repr(self):
        """String representation including regularization parameters."""
        base_repr = super().extra_repr()
        return f'{base_repr}, reg_weight={self.reg_weight}, target_concentration={self.target_concentration}'
