
'''
    The root of the TF-learn Object Model
'''
class TfLearnObj(object):
    '''
        Constructs the Root TF-Learn Object.
    '''
    def __init__(self):
        self.layer = None

    '''
        Returns the internal TF-learn layer.
    '''
    def get_layer(self):
        return self.layer
