
'''
    Outputs the input element scaled up by 1 / keep_prob. The scaling is so that
    the expected sum is unchanged.

    By default, each element is kept or dropped independently. If noise_shape is
    specified, it must be broadcastable to the shape of x, and only dimensions
    with noise_shape[i] == shape(x)[i] will make independent decisions. For
    example, if shape(x) = [k, l, m, n] and noise_shape = [k, 1, 1, n], each
    batch and channel component will be kept independently and each row and
    column will be kept or not kept together.
'''
class Dropout(CoreLayer):

    '''
        Constructs the Dropout Layer.

        incoming : A Tensor. The incoming tensor.
        keep_prob : A float representing the probability that each element is
                    kept.
        noise_shape : A 1-D Tensor of type int32, representing the shape for
                      randomly generated keep/drop flags.
        name : A name for this layer (optional).
    '''
    def __init__(self, incoming, keep_prob, noise_shape=None, name='Dropout'):

        # Invoke the super class constructor
        super(Dropout, self).__init__()

        # Save off the incoming parameters
        self.incoming = incoming
        self.keep_prob = keep_prob
        self.noise_shape = noise_shape
        self.name = name

        self.initialize_dropout_layer()

    '''
        Initializes the dropout layer

        Args:
            None

        Returns:
            Nothing
    '''
    def initialize_dropout_layer(self):
        from tflearn.layers.core import dropout
        self.layer = dropout(incoming, keep_prob, noise_shape, name)
