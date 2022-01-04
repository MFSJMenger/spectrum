"""Simple toolset to broaden a given dataset to get
a more contiguous set of datapoints, e.g. computing an
broadend spectrum in quantum chemistry with a set of given
excitation energies and oscillator strengths
"""
from functools import lru_cache
import numpy as np
from scipy.integrate import quad


__all__ = ['register_broadening_scheme', 'broaden', 'Dataset', 'SpectralDataset']


def broadening_gaussian(x, x0, intensity, sigma):
    """Apply gaussian broadening"""
    return intensity/sigma * np.exp(- ((x-x0)/sigma)**2)


def broadening_lorenzian(x, x0, intensity, sigma):
    """Apply lorenzian broadening"""
    return intensity * 1.0/(1 + ((x - x0)*2/sigma)**2)


class _BroadeningCalculator:

    """Base class to compute the broadening point utilizing lru_cache
    to speed up calculations
    """

    _broadening_schemes = {
            'gaussian': broadening_gaussian,
            'lorenzian': broadening_lorenzian,
    }

    @classmethod
    def register_scheme(cls, name, function):
        """register broadening scheme

        Args
        ----

        name, str:
            name of the broadening schme

        function, callable, function(x, x0, weights, sigma):

            x: numpy array of reference points
            x0: point the broadend value should be computed
            weights: float or numpy array of floats,
                     representing the height/intensity of the reference point
            sigma: broadening constant

        """
        if name not in cls._broadening_schemes:
            raise SystemExit('broadening scheme already defined, please use a different name')

        if not callable(function):
            raise SystemExit('new broadening function need to be callable')
        cls._broadening_schemes[name] = function

    def __init__(self, values, weights, sigma, broadening, *, nsteps=None):
        self._values = values
        self._weights = weights
        self._sigma = sigma
        self._broadening = self._select_broadening(broadening)
        self._call = lru_cache(maxsize=nsteps)(
                     self._get_call_function(values, weights, sigma, self._broadening))

    @staticmethod
    def _get_call_function(values, weights, sigma, broadening):
        """return function `call`"""
        if weights is None:
            def _call(x):
                nonlocal values, sigma
                return np.sum(broadening(x, values, 1.0, sigma))
        else:
            def _call(x):
                nonlocal values, sigma, weights
                return np.sum(broadening(x, values, weights, sigma))
        return _call

    @classmethod
    def _select_broadening(cls, selection):
        """select the broadening scheme"""
        sel = cls._broadening_schemes.get(selection, None)
        if sel is not None:
            return sel
        raise ValueError("Selection has to be among "
                         f"{', '.join(cls._broadening_schemes)} and not '{selection}'")

    def __call__(self, x0):
        """Compute the broadend value"""
        return self._call(x0)


def integrate(x, y, dx, calc, typ='quad', *, bounds=None):
    """numerically integrate the function `calc`
    at the points `x` with value `y`, seperated by `dx`

    Args:
    -----

    x, np.array(double):
        points at which the function is evaluated

    y, np.array(double):
        function results at the points x

    dx, double:
        distance between points in x

    calc, callable, f(x):
        function to integrate between `bounds[0]` and `bounds[-1]`
        x, double: point to compute the value

    typ, str:
        method to use for integration, currently used:
        trapez: using np.trapz
        quad: using scipy.integrate.quad

    bounds, (float, float):
        bounds = (start, stop) for the integration
        if None:
            bounds = (x[0], x[-1])
        else:
            integration bounds
    """
    if typ == 'trapez':
        return np.trapz(y, x, dx)
    if typ == 'quad':
        if bounds is None:
            bounds = (x[0], x[-1])
        return quad(calc, *bounds)[0]
    raise ValueError(f"Unknown integration typ '{typ}'")


class Dataset:
    """Easy repesentation of a 2D dataset with `x` and corresponding `y` values
    computed using a `calculator` function (used to integrate the dataset)
    """

    def __init__(self, x, y, calculator):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        self.x = x
        self.y = y
        self.calc = calculator

    def as_tuple(self):
        """return dataset as a tuple"""
        return (self.x, self.y)

    @property
    def dx(self):
        """compute the distance between two datapoints"""
        return abs(self.x[0] - self.x[1])

    def scale(self, const, typ='x'):
        """scale either x or y with a constant"""
        if typ == 'x':
            self.x *= const
        elif typ == 'y':
            self.y *= const
        raise ValueError(f"Could not understand scaling type '{typ}', can only be 'x' or 'y'")

    def apply(self, func, typ='x'):
        """apply a function to all values of either the x or y"""
        if typ == 'x':
            self.x = func(self.x)
        elif typ == 'y':
            self.y = func(self.y)
        raise ValueError(f"Could not understand convert type '{typ}', can only be 'x' or 'y'")

    def normalize(self, typ, bounds=None):
        """normalize the dataset, either by the highest value ('height', or the integral 'density')
        """
        if typ == 'height':
            scale = np.max(self.y)
        elif typ == 'density':
            scale = integrate(self.x, self.y, self.dx, self.calc, typ='quad', bounds=bounds)
        else:
            raise ValueError("Unknown normalization scheme")

        if scale != 0.0:
            self.y /= scale
        else:
            print("Warning: no normalization done, as normalization factor is 0.0")


def broaden(sigma, values, weights=None, *,
            bounds=None, nsteps=100, broadening='gaussian', normalize=None):
    """Broaden a set of `values` with optional `weights` between the given
    `bounds` at `nsteps` equal distant points (using np.linspace),
    using a given broadening scheme, with a fixed broadening `sigma`

    sigma, float:
        Broadening used

    values, Sequence:
        known values which should be broadend

    weights, Sequence, optional:
        weights of the values, e.g. their intensity
        if None: no weights are applied

    bounds, (float, float), optional:
        bounds between the spectrum is computed
        (xstart, xstop)
        if None, values are computed automatically according to:
            xstart = np.min(values)
            xstop = np.max(values)
    nsteps, int:
        number of points to compute

    broadening, str:
        broadening scheme to choose

    normalize, str, optional:
        If to normalize the spectrum, if None it will be not normalized
        normalize can be currently: height -> set the maximum y value to 1.0
                                    density -> numerically integrate the spectrum within
                                               the bounds and set the integral to 1.0

    Returns:
    --------
        Dataset

    """
    if weights is not None:
        if len(values) != len(weights):
            raise ValueError("Values and weights need to have same size!")
        weights = np.array(weights)
        values = np.array(values)
    x = _select_range(bounds, values, nsteps)
    calc = _BroadeningCalculator(values, weights, sigma, broadening, nsteps=nsteps)
    y = np.fromiter((calc(x0) for x0 in x), np.double)
    result = Dataset(x, y, calc)
    if normalize is not None:
        result.normalize(normalize)
    return result


def _select_range(bounds, values, nsteps):
    if bounds is not None:
        try:
            xstart, xstop = bounds
        except Exception:
            raise ValueError("Bounds need to be tuple/list of size 2 and not "
                             f"'{type(bounds)}'") from None
    else:
        xstart, xstop = None, None
    if xstart is None:
        xstart = np.min(values)
    if xstop is None:
        xstop = np.max(values)
    return np.linspace(xstart, xstop, nsteps)


class SpectralDataset:

    """Dataset to be broadend"""

    def __init__(self, xvalues, weights=None):
        if weights is not None:
            if len(xvalues) != len(weights):
                raise ValueError("Values and weights need to have same size!")
            weights = np.array(weights)
        self.xvalues = np.array(xvalues)
        self.weights = weights

    def broaden(self, sigma, nsteps=100, *,
                without_weights=False, bounds=None, broadening='gaussian', normalize=None):
        """broaden dataset"""
        if without_weights is True:
            weights = None
        else:
            weights = self.weights
        return broaden(sigma, self.xvalues, weights=weights, bounds=bounds,
                       nsteps=nsteps, broadening=broadening, normalize=normalize)


register_broadening_scheme = _BroadeningCalculator.register_scheme
