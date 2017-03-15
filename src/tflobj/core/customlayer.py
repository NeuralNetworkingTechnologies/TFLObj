
'''
    A custom layer that can apply any operations to the incoming Tensor or list
    of Tensor. The custom function can be pass as a parameter along with its
    parameters.
'''
class CustomLayer(CoreLayer):

    '''
        Constructs the Custom Layer.

        incoming : A Tensor or list of Tensor. Incoming tensor.
        custom_fn : A custom function, to apply some ops on incoming tensor.
        **kwargs: Some custom parameters that custom function might need.
    '''
    def __init__(self, incoming, custom_fn, **kwargs):

        # Invoke the super class constructor
        super(CustomLayer, self).__init__()

        # Save off the incoming parameters
        self.incoming = incoming
        self.custom_fn = custom_fn
        self.kwargs = kwargs

        self.initialize_custom_layer()

    '''
        Initializes the custom layer

        Args:
            None

        Returns:
            Nothing
    '''

    def initialize_custom_layer(self):
        from tflearn.layers.core import custom_layer
        self.layer = custom_layer(self.incoming,
                                  self.custom_fn,
                                  self.kwargs)
