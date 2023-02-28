import yaml


class DotDict(dict):

    # update, __setitem__ etc. omitted, but required if
    # one tries to set items using dot notation. Essentially
    # this is a read-only view.

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError:
            return super().__getattr__(k)
        if isinstance(v, dict):
            return DotDict(v)
        return v

    def __getitem__(self, k):
        if isinstance(k, str) and '.' in k:
            k = k.split('.')
        if isinstance(k, (list, tuple)):
            return reduce(lambda d, kk: d[kk], k, self)
        return super().__getitem__(k)

    def get(self, k, default=None):
        if isinstance(k, str) and '.' in k:
            try:
                return self[k]
            except KeyError:
                return default
            return super().get(k, default=default)  # CHECK: indentation correct?


def load_configurations(path: str) -> DotDict:
    """
    Used for parsing configuration files
    :param str path: path to conf file
    :returns DotDict: dictionary accessing fields with dot notation
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return DotDict(cfg)
