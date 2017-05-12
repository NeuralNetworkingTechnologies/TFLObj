
from .core import CoreLayer

'''
    A fully connected layer.
'''
class FullyConnected(CoreLayer):

    '''
        Initializes the fully connected layer

        incoming: Tensor. Incoming (2+)D Tensor.
        n_units: int, number of units for this layer.
        activation: str (name) or function (returning a Tensor). Activation
                    applied to this layer (see tflearn.activations).
                    Default: 'linear'.
        bias: bool. If True, a bias is used.
        weights_init: str (name) or Tensor. Weights initialization.
                      Default: 'truncated_normal'.
        bias_init: str (name) or Tensor. Bias initialization. Default: 'zeros'.
        regularizer: str (name) or Tensor. Add a regularizer to this layer
                     weights. Default: None.
        weight_decay: float. Regularizer decay parameter. Default: 0.001.
        trainable: bool. If True, weights will be trainable.
        restore: bool. If True, this layer weights will be restored when
                 loading a model.
        reuse: bool. If True and 'scope' is provided, this layer variables
                     will be reused (shared).
        scope: str. Define this layer scope (optional). A scope can be used to
               share variables between layers. Note that scope will override
               name.
        name: A name for this layer (optional). Default: 'FullyConnected'.
    '''
    def __init__(self,
                 incoming,
                 n_units,
                 activation='linear',
                 bias=True,
                 weights_init='truncated_normal',
                 bias_init='zeros',
                 regularizer=None,
                 weight_decay=0.001,
                 trainable=True,
                 restore=True,
                 reuse=False,
                 scope=None,
                 name='FullyConnected'):

        # Invoke the super class constructor
        super(FullyConnected, self).__init__()

        # Save off the incoming paramters
        self.incoming = incoming
        self.n_units = n_units
        self.activation = activation
        self.bias = bias
        self.weights_init = weights_init
        self.bias_init = bias_init
        self.regularizer = regularizer
        self.weight_decay = weight_decay
        self.trainable = trainable
        self.restore = restore
        self.reuse = reuse
        self.scope = scope
        self.name = name

        self.initialize_fullyconnected_layer()

    '''
        Initializes the fully-connected layer

        Args:
            None

        Returns:
            Nothing
    '''
    def initialize_fullyconnected_layer(self):
        from tflearn.layers.core import fully_connected
        self.layer = fully_connected(self.incoming,
                                     self.n_units,
                                     self.activation,
                                     self.bias,
                                     self.weights_init,
                                     self.bias_init,
                                     self.regularizer,
                                     self.weight_decay,
                                     self.trainable,
                                     self.restore,
                                     self.reuse,
                                     self.scope,
                                     self.name)
