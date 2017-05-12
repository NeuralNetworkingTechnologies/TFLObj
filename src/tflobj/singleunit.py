
from .core import CoreLayer

'''
    A single unit (Linear) Layer.
'''
class SingleUnit(CoreLayer):

    '''
        Constructs the Single Unit Layer.

        incoming: Tensor. Incoming Tensor.
        activation: str (name) or function. Activation applied to this layer.
                    Default: 'linear'.
        bias: bool. If True, a bias is used.
        trainable: bool. If True, weights will be trainable.
        restore: bool. If True, this layer weights will be restored when
                 loading a model.
        reuse: bool. If True and 'scope' is provided, this layer variables will
               be reused (shared).
        scope: str. Define this layer scope (optional). A scope can be used to
               share variables between layers. Note that scope will override
               name.
        name: A name for this layer (optional). Default: 'Linear'.
    '''
    def __init__(self,
                 incoming,
                 activation='linear',
                 bias=True,
                 trainable=True,
                 restore=True,
                 reuse=False,
                 scope=None,
                 name='Linear'):

        # Invoke the super class constructor
        super(SingleUnit, self).__init__()

        # Save off the incoming parameters
        self.incoming = incoming
        self.activation = activation
        self.bias = bias
        self.trainable = trainable
        self.restore = restore
        self.reuse = reuse
        self.scope = scope
        self.name = name

        self.initialize_singleunit_layer()

    '''
        Initializes the single-unit layer

        Args:
            None

        Returns:
            Nothing
    '''
    def initialize_singleunit_layer(self):
        from tflearn.layers.core import single_unit
        self.layer = single_unit(self.incoming,
                                 self.activation,
                                 self.bias,
                                 self.trainable,
                                 self.restore,
                                 self.reuse,
                                 self.scope,
                                 self.name)
