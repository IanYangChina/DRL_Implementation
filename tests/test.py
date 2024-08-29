from collections import namedtuple

t = namedtuple('t', ['a', 'b'])

print('b' in t._fields)