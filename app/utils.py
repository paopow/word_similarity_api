from numpy import dot
from numpy.linalg import norm

def cossim(a, b):
    return dot(a, b) / (norm(a) * norm(b))
