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
  #  'none',
  #  'noise',
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
    # 'max_pool_3x3', # 0
    # 'avg_pool_3x3', # 1
    # 'skip_connect', # 2
    # 'sep_conv_3x3', # 3
    # 'sep_conv_5x5', # 4
    # 'dil_conv_3x3', # 5
    # 'dil_conv_5x5'  # 6
]

primitives_5 = OrderedDict([('primitives_normal', 1 * [PRIMITIVES])])

spaces_dict = {
    's5': primitives_5, # DARTS Space
}
