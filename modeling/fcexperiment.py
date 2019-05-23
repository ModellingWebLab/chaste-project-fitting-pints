import numpy as np

import fc
import fc.utility.test_support as TestSupport
import fc.simulations.model as Model
from fc.simulations.solvers import CvodeSolver
from fc.language import values as V


class FunctionalCurationExperiment(object):
    """
    Accepts paths (absolute or relative to Chaste installation location) for
    functional curation protocol and CellML model files.
    """
    def __init__(
            self, protoFile, modelFile, outputFolder=None, solver=CvodeSolver()
            ):
        self.protocol = fc.Protocol(protoFile)
        # The exposeAnnotatedVariables flag allows direct modification of model
        # parameters
        self.protocol.SetModel(modelFile,exposeNamedParameters=True)
        self.protocol.model.SetSolver(solver)
        # Ideally, protocols will not write output in order to avoid I/O calls
        if outputFolder != None:
            self.protocol.SetOutputFolder(outputFolder)

    def simulate(self,parameters=None):
        """
        Returns dict of experimental outputs when simulated under supplied
        parameters.

        Parameters MUST be of the form 'namespace:parameter' with relation to
        the model.
        Two reserved namespaces exist:
            - 'obj' - parameters to an objective function (ignored here).
            - 'proto' - parameters controlling protocol input.
        When no parameters supplied, simulates under default parameterization.
        """

        self.protocol.model.ResetState()
        if parameters != None:
            for key,val in parameters.iteritems():
                # NOTE: May want to check existence of parameters/namespaces
                # for informative failure.
                tokens = key.split(':')
                assert len(tokens) == 2, "Parameter name not of the form namespace:parameter"

                if tokens[0] == 'proto':
                    self.protocol.SetInput(tokens[1],V.Simple(val))
                else:
                    envmap = self.protocol.model.GetEnvironmentMap()[tokens[0]]
                    envmap.OverwriteDefinition(tokens[1],V.Simple(val))
            # Must adjust initial state AFTER setting parameters so that nested
            # simulations do not reset to default parameter values between
            # loops
            self.protocol.model.initialState = self.protocol.model.state.copy()

        self.protocol.Run(verbose=False,writeOut=False)

        # Return dict with 'Value' representation unwrapped to underlying Python representation
        return dict([(key,val.unwrapped) for key,val in self.protocol.outputEnv.bindings.iteritems()])


    def setInputs(self,inputs):
        """
        Accepts a dictionary of name,value pairs and updates experimental
        inputs for next run. Required only for experimental design analyses
        (not yet supported from Algorithm).

        """
        for name,value in inputs.iteritems():
            # Value.Array expects a np array in its constructor
            if isinstance(value,list):
                self.protocol.SetInput(name,V.Array(np.array(value)))
            elif isinstance(value,np.ndarray):
                self.protocol.SetInput(name,V.Array(value))
            # If not iterable, assumes a single value (Value.Simple constructor
            # will trigger error otherwise)
            else:
                self.protocol.SetInput(name,V.Simple(value))
