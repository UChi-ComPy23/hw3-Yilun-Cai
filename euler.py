"""
Defintions for problem 0
"""

import numpy as np
import scipy.integrate
from scipy.integrate import DenseOutput
from scipy.interpolate import interp1d
from warnings import warn

class ForwardEulerOutput(DenseOutput):
    """Dense output."""
    
    def __init__(self, t_old, t, y_old, y):
        
        super().__init__(t_old, t)
        self.y_old = y_old
        self.y = y
        self.h = t - t_old
        
    def _call_impl(self, t):
        
        # Linear interpolation between y_old and y
        if self.h == 0:
            return self.y.copy()

        theta = (t - self.t_old) / self.h
        return self.y_old + theta * (self.y - self.y_old)

class ForwardEuler(scipy.integrate.OdeSolver):
    
    def __init__(self, fun, t0, y0, t_bound, vectorized=False, **extraneous):
       
        # Set default step size if not provided
        self.h = float(extraneous.pop("h", (t_bound - t0) / 100.0))
       
        # Initialize
        super().__init__(fun, t0, y0, t_bound, vectorized, **extraneous)
        
        self.direction = 1  # Forward integration

        self.t_old = self.t
        self.y_old = self.y.copy()

        self.nfev = 0
        
        self.njev = 0
        self.nlu = 0
        
        self.nsteps = 0
    
    def _step_impl(self):
        """Perform one step of Forward Euler method."""

        f = self.fun(self.t, self.y)
        self.nfev += 1

        h_step = self.h * self.direction
        t_new = self.t + h_step
        
        if self.direction * (t_new - self.t_bound) > 0:
            t_new = self.t_bound
            h_step = t_new - self.t
        
        y_new = self.y + h_step * f

        self.t_old, self.y_old = self.t, self.y
        self.t, self.y = t_new, y_new
        
        if self.t == self.t_bound:
            self.status = 'finished'
        
        self.nsteps += 1

        return True, None
    
    def _dense_output_impl(self):
        """Return dense output for the last step."""
        return ForwardEulerOutput(self.t_old, self.t, self.y_old, self.y)