import numpy as np
from numpy.typing import NDArray
import math

class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        tot = 0
        m = max(z)
        for i in range(len(z)):
            z[i] -= m
        
        denom = sum(np.exp(z))
        for i in range(len(z)):
            z[i] = math.exp(z[i])/denom

        return np.round(z,4)
    
