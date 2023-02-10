# --------------------------------------------------------------------------------
# BodyFlow
# Version: 1.0
# Copyright (c) 2023 Instituto Tecnológico de Aragón (www.itainnova.es) (Spain)
# Date: February 2023
# Authors: Ana Caren Hernández Ruiz                      ahernandez@itainnova.es
#          Ángel Gimeno Valero                              agimeno@itainnova.es
#          Carlos Marañes Nueno                            cmaranes@itainnova.es
#          Irene López Bosque                                ilopez@itainnova.es
#          María de la Vega Rodrigálvarez Chamarro   vrodrigalvarez@itainnova.es
#          Pilar Salvo Ibáñez                                psalvo@itainnova.es
#          Rafael del Hoyo Alonso                          rdelhoyo@itainnova.es
#          Rocío Aznar Gimeno                                raznar@itainnova.es
# All rights reserved 
# --------------------------------------------------------------------------------

from models.HPE2D import HPE2D
from models.HPE3D import HPE3D
from common_pose.BodyLandmarks import BodyLandmarks3d

class HPE:
    """
    End-to-end human pose estimator. It needs a 2D and a 3D pose estimator which
    lifts to 3D the 2D position.
    """
    def __init__(self, hpe_2d: HPE2D, hpe_3d: HPE3D):
        self._hpe_2d = hpe_2d
        self._hpe_3d = hpe_3d

    def init_buffers(self, frame, timestamp) -> bool:
        """
        It returns False while the buffers are not still full. Once they are full,
        it returns True.
        """
        keypoints_2d = self._hpe_2d.get_frame_keypoints(frame)
        converted_keypoints = self._hpe_3d.translate_keypoints_2d(keypoints_2d)
        return self._hpe_3d.init_buffers(frame, converted_keypoints, timestamp, keypoints_2d)

    def destroy_buffer(self) -> BodyLandmarks3d:
        """
        It returns body landmarks until de buffer is destroyed. Once destruyed,
        it returns None
        """
        return self._hpe_3d.destroy_buffer()

    def predict_pose(self) -> BodyLandmarks3d:
        """
        It predicts the corresponding pose.
        """
        keypoints_3d = self._hpe_3d.get_3d_keypoints()
        return keypoints_3d

    def add_frame(self, frame, timestamp) -> bool:
        """
        It adds a new frame into the buffer, returning true if the buffer is complete,
        meaning that it overrides the oldest frame. If false, it means the that buffer is
        not still full.
        """
        keypoints_2d = self._hpe_2d.get_frame_keypoints(frame)
        converted_keypoints = self._hpe_3d.translate_keypoints_2d(keypoints_2d)
        self._hpe_3d.add_frame(frame, converted_keypoints, timestamp, keypoints_2d)