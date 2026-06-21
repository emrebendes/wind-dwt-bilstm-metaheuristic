# -*- coding: utf-8 -*-
"""
optimizers/ - Generic metaheuristic optimizer framework.

Tum algoritmalar BaseOptimizer'dan turer ve string ID ile erisilir:
    from optimizers import get_optimizer
    cls = get_optimizer('abc')

Mevcut algoritmalar: abc, ga, pso, gwo, ho, fno, raindrop, toa
"""

_ALGORITHM_REGISTRY = {}


def register_algorithm(name):
    """Decorator: yeni algoritma sinifini registry'e kaydeder."""
    def decorator(cls):
        _ALGORITHM_REGISTRY[name.lower()] = cls
        cls.ALGORITHM_NAME = name.lower()
        return cls
    return decorator


def get_optimizer(name):
    """String isim ile optimizer sinifini dondurur."""
    _ensure_registered()
    name = name.lower()
    if name not in _ALGORITHM_REGISTRY:
        available = ', '.join(sorted(_ALGORITHM_REGISTRY.keys()))
        raise KeyError(
            f"Bilinmeyen algoritma: '{name}'. Mevcut: {available}"
        )
    return _ALGORITHM_REGISTRY[name]


def list_algorithms():
    """Kayitli tum algoritmalarin listesini dondurur."""
    _ensure_registered()
    return sorted(_ALGORITHM_REGISTRY.keys())


def _ensure_registered():
    """Mevcut algoritma siniflarini import ederek registry'i doldurur."""
    from optimizers import abc_optimizer
    from optimizers import ga_optimizer
    from optimizers import pso_optimizer
    from optimizers import gwo_optimizer
    from optimizers import ho_optimizer
    from optimizers import fno_optimizer
    from optimizers import raindrop_optimizer
    from optimizers import toa_optimizer


__all__ = [
    'register_algorithm',
    'get_optimizer',
    'list_algorithms',
]
