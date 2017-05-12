
from .core import CoreLayer

'''
    Flatten the incoming Tensor.
'''
class Flatten(CoreLayer):

    '''
        Constructs the Flatten Layer.

        incoming: A Tensor. The incoming tensor.
        name: A name for this layer (optional).
    '''
    def __init__(self, incoming, name='Flatten'):

        # Invoke the super class constructor
        super(Flatten, self).__init__()

        # Save off the incoming parameters
        self.incoming = incoming
        self.name = name

        self.initialize_flatten_layer()

    '''
        Initializes the flatten layer

        Args:
            None

        Returns:
            Nothing
    '''
    def initialize_flatten_layer(self):
        from tflearn.layers.core import flatten
        self.layer = flatten(self.incoming, self.name)
