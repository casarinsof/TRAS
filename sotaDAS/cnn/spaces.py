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
    'noise',
    'translate_x',
    'translate_y',
    'pool',
    'rotate',
    'autocontrast',
    'invert',
    'equalize',
    'solarize',
    'posterize',
    'contrast',
    'brightness',
    'sharpness',
    'color',
    'shear_xy',
  #  'cutout',
]


primitives_5 = OrderedDict([('primitives_normal', 1 * [PRIMITIVES])])


spaces_dict = {
    's5': primitives_5,

}
