
'''
    A fully connected highway network layer
'''
class FullyConnectedHighway(CoreLayer):

    '''
        Constructs the Fully Connected Highway Layer.

        incoming: Tensor. Incoming (2+)D Tensor.
        n_units: int, number of units for this layer.
        activation: str (name) or function (returning a Tensor).
                    Default: 'linear'.
        transform_dropout: float: Keep probability on the highway transform gate.
        weights_init: str (name) or Tensor. Weights initialization.
                      Default: 'truncated_normal'.
        bias_init: str (name) or Tensor. Bias initialization. Default: 'zeros'.
        regularizer: str (name) or Tensor. Add a regularizer to this layer
                     weights (see tflearn.regularizers). Default: None.
        weight_decay: float. Regularizer decay parameter. Default: 0.001.
        trainable: bool. If True, weights will be trainable.
        restore: bool. If True, this layer weights will be restored when
                 loading a model
        reuse: bool. If True and 'scope' is provided, this layer variables
                     will be reused (shared).
        scope: str. Define this layer scope (optional). A scope can be used to
               share variables between layers. Note that scope will override name.
        name: A name for this layer (optional). Default: 'FullyConnectedHighway'.
    '''

    def __init__(self,
                 incoming,
                 n_units,
                 activation='linear',
                 transform_dropout=None,
                 weights_init='truncated_normal',
                 bias_init='zeros',
                 regularizer=None,
                 weight_decay=0.001,
                 trainable=True,
                 restore=True,
                 reuse=False,
                 scope=None,
                 name='FullyConnectedHighway'):

        # Invoke the super class constructor
        super(FullyConnectedHighway, self).__init__()

        # Save off incoming parameters
        self.incoming = incoming
        self.n_units = n_units
        self.activation = activation
        self.transform_dropout = transform_dropout
        self.weights_init = weights_init
        self.bias_init = bias_init
        self.regularizer = regularizer
        self.weight_decay = weight_decay
        self.trainable = trainable
        self.restore = restore
        self.reuse = reuse
        self.scope = scope
        self.name = name

        self.initialize_fullyconnectedhighway_layer()

    '''
        Initializes the fully connected highway layer

        Args:
            None

        Returns:
            Nothing
    '''
    def initialize_fullyconnectedhighway_layer(self):
        from tflearn.layers.core import highway
        self.layer = highway(self.incoming,
                             self.n_units,
                             self.activation,
                             self.transform_dropout,
                             self.weights_init,
                             self.bias_init,
                             self.regularizer,
                             self.weight_decay,
                             self.trainable,
                             self.restore,
                             self.reuse,
                             self.scope,
                             self.name)
