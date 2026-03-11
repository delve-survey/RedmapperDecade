from __future__ import division, absolute_import, print_function

def setdefaults(kwargs, defaults):
    for k, v in defaults.items():
        kwargs.setdefault(k, v)
    return kwargs

