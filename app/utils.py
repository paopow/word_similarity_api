from numpy import dot
from numpy.linalg import norm
import csv

def cossim(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def read_csv(filename, has_header = True):
    items = []
    with open(filename, 'rU') as f:
        reader = csv.reader(f)
        if has_header:
            reader.next()

        for row in reader:
            items.append(row)
    return items
