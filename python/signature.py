class type(object):

    def __init__(self, name, dims=[]):
        self.name = name
        self.dims = dims

    def __mul__(self, s):
        return type(self.name, self.dims + [s])

float = type('float')
int = type('int')
