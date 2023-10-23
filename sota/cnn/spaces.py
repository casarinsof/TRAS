from collections import OrderedDict





# primitives_2 = OrderedDict([('primitives_normal', 14 * [['skip_connect',
#                                                          'sep_conv_3x3']]),
#                             ('primitives_reduct', 14 * [['skip_connect',
#                                                          'sep_conv_3x3']])])
#
# primitives_3 = OrderedDict([('primitives_normal', 14 * [['none',
#                                                          'skip_connect',
#                                                          'sep_conv_3x3']]),
#                             ('primitives_reduct', 14 * [['none',
#                                                          'skip_connect',
#                                                          'sep_conv_3x3']])])
#
# primitives_4 = OrderedDict([('primitives_normal', 14 * [['noise',
#                                                          'sep_conv_3x3']]),
#                             ('primitives_reduct', 14 * [['noise',
#                                                          'sep_conv_3x3']])])

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
