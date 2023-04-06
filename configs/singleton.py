import functools


def Singleton(cls):
    """
    单例decorator
    """
    _instance = {}

    @functools.wraps(cls)
    def _singleton(*args, **kargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
        return _instance[cls]

    return _singleton
