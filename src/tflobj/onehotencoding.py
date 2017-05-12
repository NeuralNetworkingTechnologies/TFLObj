
from .core import CoreLayer

'''
    Transform numeric labels into a binary vector.
'''
class OneHotEncoding(CoreLayer):

    '''
        Constructs the One-Hot Encoding Layer.

        target: Placeholder. The labels placeholder.
        n_classes: int. Total number of classes.
        on_value: scalar. A scalar defining the on-value.
        off_value: scalar. A scalar defining the off-value.
        name: A name for this layer (optional). Default: 'OneHotEncoding'.
    '''
    def __init__(self,
                 target,
                 n_classes,
                 on_value=1.0,
                 off_value=0.0,
                 name='OneHotEncoding'):

        # Invoke the super class constructor
        super(OneHotEncoding, self).__init__()

        # Save off the incoming parameters
        self.target = target
        self.n_classes = n_classes
        self.on_value = on_value
        self.off_value = off_value
        self.name = name

        self.initialize_onehotencoding_layer()

    '''
        Initializes the one hot encoding layer

        Args:
            None

        Returns:
            Nothing
    '''
    def initialize_onehotencoding_layer(self):
        from tflearn.layers.core import one_hot_encoding
        self.layer = one_hot_encoding(self.target,
                                      self.n_classes,
                                      self.on_value,
                                      self.off_value,
                                      self.name)
