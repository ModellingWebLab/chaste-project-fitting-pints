import scipy.stats as stats
import numpy
import numpy.random as rand


class Distribution:
    """
    Base class for univariate distributions over single model parameters
    """
    def draw(self):
        """ Returns value randomly drawn from the distribution """
        raise NotImplementedError

    def pdf(self, value):
        """ Takes float value and returns float probability """
        raise NotImplementedError

    def is_positive(self):
        """
        Returns True if this distribution allows only positive values.
        """
        return False


class ParameterDistribution:
    """
    Base class for distributions over model parameter vectors
    """

    def draw(self):
        """
        Returns a dict of {name,value} pairs representing a random draw from
        the distribution
        """
        raise NotImplementedError

    # NOTE: despite naming convention, can support discrete distributions
    def pdf(self, values):
        """
        Accepts a dict of {name,value} pairs and returns the probability under
        the distribution
        """
        raise NotImplementedError

    def is_positive(self):
        """
        Returns True if this distribution allows only positive values.
        """
        return False


class Uniform(Distribution):
    """
    Initializes a uniform distribution between 'low' and 'high'.
    """
    def __init__(self, low=0, high=1):
        self.low = float(min(low, high))
        self.high = float(max(low, high))

    def draw(self):
        return rand.uniform(self.low, self.high)

    def pdf(self, value):
        u = stats.uniform(loc=self.low, scale=self.high - self.low)
        return u.pdf(value)

    def mean(self):
        return (self.high - self.low) / 2

    def std(self):
        return 1.0 / numpy.sqrt(12) * (self.high - self.low)

    def is_positive(self):
        return self.low > 0


class IndependentParameterDistribution(ParameterDistribution):
    """
    Handles parameter vector distribution as a dictionary of Distribution
    objects distributions: {name,Distribution}
    constraints: {'eq'|'ineq',<callable>}, or sequence of said dicts. The
    callable object/function accepts a parameter list and returns 0 to pass an
    equality constraint or a non-negative value to pass an inequality
    constraint.
    """
    def __init__(self, distributions, constraints=None):
        self.distributions = distributions
        # NOTE: Should we check constraints for valid formatting?
        self.constraints = constraints

    def checkConstraints(self, values):
        if self.constraints is None:
            return True
        # NOTE: because of specification, if 'type' is incorrectly provided,
        # will default to the more general inequality constraint.
        if isinstance(self.constraints, list):
            for const in self.constraints:
                if not const(values):
                    return False
            return True
        else:
            return self.constraints(values)

    def draw(self):
        ret = dict([(k, d.draw()) for k, d in self.distributions.iteritems()])
        if self.checkConstraints(ret):
            return ret
        else:
            # NOTE: should we put an iteration limit (or a warning) for when
            # constraints might be impossible/too strict? Is there a way to
            # test for this a priori?
            while True:
                # Mutates existing dictionary to save allocation time
                for key, dist in self.distributions.iteritems():
                    ret[key] = dist.draw()
                if self.checkConstraints(ret):
                    return ret

    def pdf(self, values):
        prob = 1.0
        if not self.checkConstraints(values):
            return 0.0
        else:
            # NOTE: does not enforce bijection between keys.
            #       'values' dictionary could contain irrelevant keys.
            for key, dist in self.distributions.iteritems():
                prob = prob * dist.pdf(values[key])
            return prob

    def std(self):
        for name, d in self.distributions.iteritems():
            if not hasattr(d, 'std'):
                raise ValueError(
                    'Not all component distributions have defined standard'
                    ' deviations')
        return dict(
            [(name, d.std()) for name, d in self.distributions.iteritems()]
        )

    def mean(self):
        for name, d in self.distributions.iteritems():
            if not hasattr(d, 'mean'):
                raise ValueError(
                    'Not all component distributions have defined means')
        return dict(
            [(name, d.mean()) for name, d in self.distributions.iteritems()]
        )

    def is_positive(self):
        """
        Returns True if all parameters must be positive.
        """
        for name, d in self.distributions.items():
            if not d.is_positive():
                return False
        return True


class DiscreteParameterDistribution(ParameterDistribution):
    """
    Handles (potentially) dependent parameter distributions, such as those
    generated by posterior estimating parameterization techniques.
    particles: list<{name,float}>
    weights: list<float>
    """
    def __init__(self, particles):
        # Ensure that all particles are identically structured dicts
        err_msg = "DiscreteParameterDistribution: invalid instantiation"
        for p in particles:
            assert isinstance(p, dict), err_msg
            assert len(set(p.keys()) - set(particles[0].keys())) == 0, err_msg
        self.particles = particles

        # Normalize weights (may require float casting for ints)
        self.weights = numpy.ones(len(particles))
        self.weights = self.weights / sum(self.weights)

    def _marginals(self):
        marginals = {}

        for i, particle in enumerate(self.particles):
            for pname in particle:
                if pname not in marginals:
                    marginals[pname] = [particle[pname]]
                else:
                    marginals[pname].append(particle[pname])
        return marginals

    # Returns a dicitonary of mean values of marginal distributions for each
    # parameter.
    def mean(self):
        marginals = self._marginals()
        means = dict([(pname, 0.0) for pname in marginals])
        try:
            for pname in marginals:
                for i, val in enumerate(marginals[pname]):
                    means[pname] += self.weights[i] * val
            return means
        except:
            raise ValueError(
                'DiscreteParameterDistribution - parameter structure '
                'incompatible with mean')

    # Returns a dictionary of standard deviations of marginal distributions for
    # each parameter.
    def std(self):
        marginals = self._marginals()
        means = self.mean()
        stds = dict([(pname, 0.0) for pname in marginals])
        try:
            for pname in marginals:
                for i, val in enumerate(marginals[pname]):
                    stds[pname] += self.weights[i] * (val-means[pname])**2
            for pname in stds:
                stds[pname] = stds[pname]**(0.5)
            return stds
        except:
            raise ValueError(
                'DiscreteParameterDistribution - parameter structure'
                ' incompatible with mean')
