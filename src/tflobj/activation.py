
from .core import CoreLayer

'''
    Apply given activation to incoming tensor.
'''
class Activation(CoreLayer):

    '''
        Constructs the Activation Layer.

        Params:
            incoming: A Tensor. The incoming tensor.
            activation: str (name) or function (returning a Tensor). Activation
                        applied to this layer. Default: 'linear'.
            name: A name for this layer (optional).
    '''
    def __init__(self, incoming, activation='linear', name='activation'):

        # Invoke the super class constructor
        super(Activation, self).__init__()

        # Save off the incoming parameters
        self.incoming = incoming
        self.activation = activation
        self.name = name

        self.initialize_activation_layer()

    '''
        Initializes the activation layer

        Args:
            None

        Returns:
            Nothing
    '''

    def initialize_activation_layer(self):
        from tflearn.layers.core import activation
        self.layer = activation(self.incoming, self.activation, self.name)
