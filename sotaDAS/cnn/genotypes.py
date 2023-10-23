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
arch = Genotype(normal=[('rotate', 0)], normal_concat=range(-1, 3))

arch_pool =  Genotype(normal=[('pool', 0)], normal_concat=range(-1, 3))

arch_ALL =  Genotype(normal=[('TraslZoomRot', 0)], normal_concat=range(-1, 3))

arch_trasl = Genotype(normal=[('vhshift', 0)], normal_concat=range(-1, 3))