from abc import ABC, abstractmethod


class BaseLayer(ABC):

    @abstractmethod
    @staticmethod
    def create(opt):
        pass
