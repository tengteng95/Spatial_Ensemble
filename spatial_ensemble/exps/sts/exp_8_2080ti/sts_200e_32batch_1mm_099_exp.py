# encoding: utf-8
"""
@author: zeming li
@contact: zengarden2009@gmail.com
"""

import os
from spatial_ensemble.exps.sts.sts_exp import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.max_epoch = 200
        self.param_momentum = 1.0 - 0.01
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
