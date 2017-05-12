
from .core import CoreLayer

'''
    This layer is used for inputting (aka. feeding) data to a network. A
    TensorFlow placeholder will be used if it is supplied, otherwise a new
    placeholder will be created with the given shape.
'''
class Input(CoreLayer):

    '''
        Constructs the Input Layer.

        Args:

            shape:  list of int. An array or tuple representing input data shape.
                    It is required if no placeholder is provided. First element
                    should be 'None' (representing batch size), if not provided,
                    it will be added automatically.
            placeholder: A Placeholder to use for feeding this layer (optional).
                         If not specified, a placeholder will be automatically
                         created. You can retrieve that placeholder through graph
                         key: 'INPUTS', or the 'placeholder' attribute of this
                         function's returned tensor.
            dtype: Placeholder data type (optional). Default: float32.
            data_preprocessing: A DataPreprocessing subclass object to manage
                                real-time data pre-processing when training and
                                predicting (such as zero center data, std
                                normalization...).
            data_augmentation: DataAugmentation. A DataAugmentation subclass
                               object to manage real-time data augmentation
                               while training (such as random image crop, random
                               image flip, random sequence reverse...).
            name: str. A name for this layer (optional).
    '''
    def __init__(self,
                 shape=None,
                 placeholder=None,
                 dtype=tf.float32,
                 data_preprocessing=None,
                 data_augmentation=None,
                 name='InputData'):

        # Invoke the super class constructor
        super(Input, self).__init__()

        # Save off the parameters
        self.shape = shape
        self.placeholder = placeholder
        self.dtype = dtype
        self.data_preprocessing = data_preprocessing
        self.data_augmentation = data_augmentation
        self.name = name

        self.initialize_input_layer()

    '''
        Initializes the input layer

        Args:
            None

        Returns:
            Nothing
    '''
    def initialize_input_layer(self):
        from tflearn.layers.core import input_data
        self.layer = input_data(self.shape,
                                self.placeholder,
                                self.dtype,
                                self.data_preprocessing,
                                self.data_augmentation,
                                self.name)
