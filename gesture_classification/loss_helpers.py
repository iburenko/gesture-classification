from torch.nn import BCEWithLogitsLoss
# from focal_loss.focal_loss import FocalLoss

class LossFunction:
    def __init__(self, loss_function_name):
        self.loss_function_name = loss_function_name
    
    @property
    def loss_function_name(self):
        return self._loss_function_name
    
    @loss_function_name.setter
    def loss_function_name(self, loss_function_params: tuple) -> None:
        loss_function_name, *params = loss_function_params
        if not params:
            params = [0]
        self._loss_function_name = loss_function_name
        self._loss_function = self._set_loss_function(params)

    @property
    def loss_function(self):
        return self._loss_function
    
    def _set_loss_function(self, params):
        gamma, *_ = params
        if self.loss_function_name == "bce":
            return BCEWithLogitsLoss() 
        elif self.loss_function_name == "focal":
            return FocalLoss(gamma=gamma)
