from .network import DensityNetwork
from .Lineformer import Lineformer
from .dif import DIF_Net


def get_network(type):
    if type == "mlp":
        return DensityNetwork
    elif type == "Lineformer":
        return Lineformer
    elif type == "dif":
        return
    else:
        raise NotImplementedError("Unknown network type!")
