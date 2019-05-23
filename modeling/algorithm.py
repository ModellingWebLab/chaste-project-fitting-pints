import numpy as np


class ParameterFittingTask(object):
    """
    Wrapper class for the following essential parameter fitting arguments:
        - Ordered list of parameter names
        - ParameterDistribution prior
        - Experiment experiment
        - dict expData
        - Objective objFun
    And the following optional arguments:
        - dict outputMapping: maps names of simulated output to keys in
          expData
        - dict inputs: maps names of protocol inputs to desired values
        - dict objArgs: arguments to the objective function

    Used as simplified input for ParameterFittingAlgorithm class.
    """

    def __init__(
            self, parameters, prior, experiment, expData, outputMapping=None,
            objArgs={}):
        self.parameters = parameters
        self.prior = prior
        self.experiment = experiment
        self.expData = expData
        self.outputMapping = outputMapping
        self.objArgs = objArgs
        self.objFun = None

    def calculateObjective(self,parameters):
        """
        Handles interaction between components to produce Objective output from
        parameter values.
        Primary method utilized by ParameterFittingAlgorithm.
        """

        data1, data2 = {}, {}

        # If parameters are specified with the reserved namespace 'obj',
        # pass them to the objective function.
        # If objective args are specified with the same name, they will be
        # overwritten on calls to calculateObjective.
        simParams = {}

        for key,val in parameters.iteritems():
            tokens = key.split(':')
            if len(tokens)>1 and tokens[0]=='obj':
                if self.objArgs == None:
                    self.objArgs = {}
                self.objArgs[tokens[1]] = val
            else:
                simParams[key] = val

        try:
            simData = self.experiment.simulate(simParams)
        except:
            return -np.inf


        # Match experimental/simulated data for input to objective function
        if self.outputMapping != None:
            for simName,expName in self.outputMapping.iteritems():
                data1[simName] = simData[simName]
                data2[simName] = self.expData[expName]
        else:
            data1 = simData
            data2 = self.expData

        return self.objFun(data1,data2,self.objArgs)

