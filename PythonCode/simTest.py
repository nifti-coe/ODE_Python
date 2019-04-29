from scipy.integrate import odeint
import numpy as np
import pandas as pd
import os
import math
from numba import jit


def multP_dict(simNum, dictOfParams):
    retVal = dictOfParams["listToExp"][simNum] ** dictOfParams["exponentNum"]
    return(retVal)
