# -----------------------------------------------------------------------------------
# Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose
# Copyright (c) 2018 Daniil Osokin
#
# @inproceedings{osokin2018lightweight_openpose,
#     author={Osokin, Daniil},
#     title={Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose},
#     booktitle = {arXiv preprint arXiv:1811.12004},
#     year = {2018}
# }
# -----------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# Modified for BodyFlow Version: 2.0
# Modifications Copyright (c) 2024 Instituto Tecnologico de Aragon
# (www.ita.es) (Spain)
# Date: March 2024
# Authors: Ana Caren Hernandez Ruiz                      ahernandez@ita.es
#          Angel Gimeno Valero                              agimeno@ita.es
#          Carlos Maranes Nueno                            cmaranes@ita.es
#          Irene Lopez Bosque                                ilopez@ita.es
#          Jose Ignacio Calvo Callejo                       jicalvo@ita.es
#          Maria de la Vega Rodrigalvarez Chamarro   vrodrigalvarez@ita.es
#          Pilar Salvo Ibanez                                psalvo@ita.es
#          Rafael del Hoyo Alonso                          rdelhoyo@ita.es
#          Rocio Aznar Gimeno                                raznar@ita.es
#          Pablo Perez Lazaro                               plazaro@ita.es
#          Marcos Marina Castello                           mmarina@ita.es
# All rights reserved
# --------------------------------------------------------------------------------


import collections
import logging


def load_state(net, checkpoint):
    source_state = checkpoint['state_dict']
    target_state = net.state_dict()
    new_target_state = collections.OrderedDict()
    for target_key, target_value in target_state.items():
        if target_key in source_state and source_state[target_key].size() == target_state[target_key].size():
            new_target_state[target_key] = source_state[target_key]
        else:
            new_target_state[target_key] = target_state[target_key]
            logging.warning('Not found pre-trained parameters for {}'.format(target_key))

    net.load_state_dict(new_target_state)
