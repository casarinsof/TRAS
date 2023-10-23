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
    'vshift',
    'hshift',
    'zoom',
    'rotate',
    'vhshift',
    'vzoom',
    'rotzoom',
    'pool',
    'tralRot',
    'TraslZoomRot'
]

PRIMITIVESs6 = [
    'vshift',
    'hshift',
    'zoom',
    'rotate',
    'vhshift',
    'vzoom',
    'rotzoom',
    'tralRot',
    'TraslZoomRot'
]

primitives_5 = OrderedDict([('primitives_normal', 1 * [PRIMITIVES])])

primitives_6 = OrderedDict([('primitives_normal', 1 * [PRIMITIVESs6])])

spaces_dict = {
    's5': primitives_5,
    's6': primitives_6,
}
