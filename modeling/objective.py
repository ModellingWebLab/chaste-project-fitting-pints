import numpy


class LogLikGauss(object):
    """
    Objective metric between any given simulated/experimental data (nd-arrays)
    """

    @classmethod
    def __call__(cls, data1, data2, args={}):
        """
        Accepts two dictionaries of data in <name, value> format and returns a
        float. We assume the two data sets are of identical format before
        reaching the Objective.

        Accepts two nd-arrays or two dictionaries with the same <name,nd-array>
        structure
        Accepts a 'weighting' argument which may be valued as:
            - A dictionary of <outputname, weight> format
            - A callable object operating on an nd-array (data2)
            - A string describing a common weighting scheme:
              'mean'|'meanSquare'|'sdev'
        Unless separate arguments for the objectiveMetric are passed as
        'objectiveMetricArgs', the arguments to __call__ will be forwarded.
        """

        # If the data are both nd-arrays, skip the weighting step
        if isinstance(data1,list) or isinstance(data1,numpy.ndarray):
            if not (isinstance(data2,list) or isinstance(data2,numpy.ndarray)):
                raise ValueError(
                    'Both data sets must be nd-array of identical shape or'
                    ' parallel dictionaries thereof')

            omargs = args
            if 'objectiveMetricArgs' in args:
                omargs = args['objectiveMetricArgs']
            return cls.objectiveMetric(
                numpy.array(data1),
                numpy.array(data2),omargs
            )

        # Otherwise, data expected in named dictionary format
        else:
            assert isinstance(data1,dict) and isinstance(data2,dict), "Both data sets must be nd-array of identical shape or parallel dictionaries thereof"
        obj = 0.0
        for key in data1.keys():
            assert key in data2.keys(), "Both data sets must be nd-array of identical shape or parallel dictionaries thereof"
            assert numpy.shape(data1[key]) == numpy.shape(data2[key]), "Both data sets must be nd-array of identical shape or parallel dictionaries thereof"

            # Defaults to uniform weighting in the absence of specification
            wt = 1.0
            omargs = args.copy()

            # Allows passing of objective function argumetns that
            # differentially apply to different parameters
            for argname,argval in args.iteritems():
                if isinstance(argval,dict):
                    if key in argval:
                        omargs[argname] = argval[key]
            #print "Arguments for output "+key+": "+str(omargs)

            if 'weighting' in args:
                # Dictionary of custom weights
                # NOTE: should we allow for dictionary of callable?
                if isinstance(args['weighting'],dict):
                    assert key in args['weighting'].keys(), "Weighting scheme incompletely specified"
                    assert isinstance(args['weighting'][key],float), "Invalid weighting specified for output"
                    wt = args['weighting'][key]
                # Function of experimental data
                elif hasattr(args['weighting'],'__call__'):
                    wt = args['weighting'](data2[key])
                # Predefined weighting scheme
                elif args['weighting'] == 'mean':
                    wt = 1.0/pow(numpy.mean(data2[key]),2)
                elif args['weighting'] == 'meanSquare':
                    wt = 1.0/numpy.mean(numpy.square(data2[key]))
                elif args['weighting'] == 'sdev':
                    wt = 1.0/numpy.std(data2[key])
                else:
                    assert 0, "Invalid formatting of weighting scheme for Objective"
            if 'objectiveMetricArgs' in args:
                omargs = args['objectiveMetricArgs']

            # Uses computed/provided weight to calculate total error
            obj = obj + wt * cls._objectiveMetric(data1[key],data2[key],omargs)
        return obj

    @staticmethod
    def _objectiveMetric(array1,array2,args={}):
        if not isinstance(array1,numpy.ndarray):
            if not isinstance(array1,list):
                # Data is single number; cast it to 1d numpy array to give it a 'len'
                array1 = numpy.array([array1])
            else:
                # Data is a list; cast it to a 1d numpy array
                array1 = numpy.array(array1)
        else:
            if not hasattr(array1,'len'):
                # Data is a 0d array
                array1 = numpy.array([array1])

        if not isinstance(array2,numpy.ndarray):
            if not isinstance(array2,list):
                # Data is single number; cast it to 1d numpy array to give it a 'len'
                array2 = numpy.array([array2])
            else:
                # Data is a list; cast it to a 1d numpy array
                array2 = numpy.array(array2)
        else:
            if not hasattr(array2,'len'):
                # Data is a 0d array
                array2 = numpy.array([array2])

        mean, std = 0.0, 1.0
        if 'mean' in args:
            mean = args['mean']
        if 'std' in args:
            std = args['std']

        if isinstance(std,numpy.ndarray) or isinstance(std,list):
            reg = -len(array1)*numpy.log(numpy.sqrt(2*numpy.pi)) + numpy.sum(std)
        else:
            reg = -len(array1)*numpy.log(numpy.sqrt(2*numpy.pi)*std)
        lik = -0.5*(1.0/numpy.power(std,2))*numpy.sum(numpy.power(((array1-array2)-mean),2))
        return reg+lik

