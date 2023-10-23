from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat')

PRIMITIVES = [
    'none',
    'noise',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

######## S1-S4 Space ########
#### cifar10 s1 - s4
arch_c10 = Genotype(normal=[('shear_xy', 0)], normal_concat=range(-1, 3))

