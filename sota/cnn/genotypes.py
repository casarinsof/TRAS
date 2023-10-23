from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat')

PRIMITIVES = [
    'none',
    'noise',
    'vhshift',
    'rotate',
    'pool',
    'TraslZoomRot'
]

######## S1-S4 Space ########
#### cifar10 s1 - s4
arch = Genotype(normal=[('rotate', 0)], normal_concat=range(-1, 3))

arch_pool =  Genotype(normal=[('pool', 0)], normal_concat=range(-1, 3))

arch_ALL =  Genotype(normal=[('TraslZoomRot', 0)], normal_concat=range(-1, 3))

arch_trasl = Genotype(normal=[('vhshift', 0)], normal_concat=range(-1, 3))

arch_rotate = Genotype(normal=[('rotate', 0)], normal_concat=range(-1, 3))

arch_vzoom = Genotype(normal=[('vzoom', 0)], normal_concat=range(-1, 3))

arch_rotzoom = Genotype(normal=[('rotzoom', 0)], normal_concat=range(-1, 3))