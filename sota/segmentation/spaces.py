from collections import OrderedDict

PRIMITIVES = [
    'identity',
    'vshift',
    'hshift',
    'zoom',
    'rotate',
]

primitives = OrderedDict([('primitives_normal', 14 * [PRIMITIVES])])


spaces_dict = {
    's5': primitives,

}
