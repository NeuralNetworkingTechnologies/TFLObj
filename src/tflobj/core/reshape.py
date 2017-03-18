
'''
    A layer that reshape the incoming layer tensor output to the desired shape.
'''
class Reshape(CoreLayer):

    '''
        Constructs the Reshape Layer.

        incoming: A Tensor. The incoming tensor.
        new_shape: A list of int. The desired shape.
        name: A name for this layer (optional).
    '''
    def __init__(self, incoming, new_shape, name='Reshape'):

        # Invoke the super class constructor
        super(Reshape, self).__init__()

        # Save off the incoming parameters
        self.incoming = incoming
        self.new_shape = new_shape
        self.name = name

        self.initialize_reshape_layer()

    '''
        Initializes the re-shape layer

        Args:
            None

        Returns:
            Nothing
    '''
    def initialize_reshape_layer(self):
        from tflearn.layers.core import reshape
        self.layer = reshape(self.incoming,
                             self.new_shape,
                             self.name)
