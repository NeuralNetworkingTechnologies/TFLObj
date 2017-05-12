
from .core import CoreLayer

'''
    This layer applies a function to every timestep of the input tensor. The
    custom function first argument must be the input tensor at every timestep.
    Additional parameters for the custom function may be specified in 'args'
    argument (as a list).
'''
class TimeDistributed(CoreLayer):

    '''
        Constructs the Time Distributed Layer.

        incoming: Tensor. The incoming tensor.
        fn: function. A function to apply at every timestep. This function first
            parameter must be the input tensor per timestep. Additional
            parameters may be specified in 'args' argument.
        args: list. A list of parameters to use with the provided function.
        scope: str. A scope to give to each timestep tensor. Useful when sharing
               weights. Each timestep tensor scope will be generated as
               'scope'-'i' where i represents the timestep id. Note that your
               custom function will be required to have a 'scope' parameter.
    '''
    def __init__(self, incoming, fn, args=None, scope=None):

        # Invoke the super class constructor
        super(TimeDistributed, self).__init__()

        # Save off the incoming parameters
        self.incoming = incoming
        self.fn = fn
        self.args = args
        self.scope = scope

        self.initialize_timedistributed_layer()

    '''
        Initializes the time distributed layer

        Args:
            None

        Returns:
            Nothing
    '''
    def initialize_timedistributed_layer(self):
        from tflearn.layers.core import time_distributed
        self.layer = time_distributed(self.incoming,
                                      self.fn,
                                      self.args,
                                      self.scope)
