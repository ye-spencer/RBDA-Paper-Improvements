from abc import ABC, abstractmethod

import numpy as np

""" Abstract Base Class for Coreset Selection """
class CoresetSelection(ABC):

    def __init__(self, coreset_fraction):
        self.coreset_fraction = coreset_fraction

    """ Abstract method to be implemented by subclasses 
    
    Args:
        dataset: torch.utils.data.Dataset
    
    Returns:
        Coreset: Selected coreset (list of indices)
    """
    @abstractmethod
    def select_coreset(self, dataset) -> list[int]:
        pass

""" Full Dataset Selection """
class FullDatasetSelection(CoresetSelection):
    def __init__(self, coreset_fraction=1.0):
        super().__init__(coreset_fraction)

    def select_coreset(self, dataset):
        return list(range(len(dataset)))

""" Random Coreset Selection """
class RandomCoresetSelection(CoresetSelection):
    def __init__(self, coreset_fraction):
        super().__init__(coreset_fraction)

    def select_coreset(self, dataset):
        np.random.seed(0)
        return np.random.choice(len(dataset), int(self.coreset_fraction * len(dataset)), replace=False).tolist()

""" MRMC Original Coreset Selection """
class MRMCOriginalCoresetSelection(CoresetSelection):
    def __init__(self, coreset_fraction, R, rho, gamma, model_fn, device):
        super().__init__(coreset_fraction)
        self.R = R
        self.rho = rho
        self.gamma = gamma
        self.model_fn = model_fn
        self.device = device

    def select_coreset(self, dataset):
        raise NotImplementedError("MRMC Original Coreset Selection not implemented yet")
        

""" MRMC Adaptive Hyperparameter Coreset Selection """
class MRMCAdaptiveCoresetSelection(CoresetSelection):
    def __init__(self, coreset_fraction, R, rho, gamma, model_fn, device):
        super().__init__(coreset_fraction)
        self.R = R
        self.rho = rho
        self.gamma = gamma
        self.model_fn = model_fn
        self.device = device

    def select_coreset(self, dataset):
        raise NotImplementedError("MRMC Adaptive Hyperparameter Coreset Selection not implemented yet")