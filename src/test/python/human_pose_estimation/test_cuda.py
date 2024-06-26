# --------------------------------------------------------------------------------
# BodyFlow
# Version: 2.0
# Copyright (c) 2024 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
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

import unittest
import os
import sys
sys.path.append(os.path.join('src', 'main', 'python', 'human_pose_estimation'))  # To avoid importing problems
sys.path.append(os.path.join('src', 'main', 'python', 'human_pose_estimation', 'models', 'predictors_3d', 'mixste'))
sys.path.append(os.path.join('src', 'main', 'python', 'human_pose_estimation', 'models', 'predictors_3d', 'motionbert'))


# Pose Predictors from RGB to 2D
from models.predictors_2d.CPN import CPN
from models.predictors_2d.Lightweight2D import Lightweight2D

# Pose Predictors from 2D to 3D
from models.predictors_3d.MHFormer import MHFormer
from models.predictors_3d.VideoPose3D import VideoPose3D
from models.predictors_3d.MixSTE import MixSTE
from models.predictors_3d.MotionBert import MotionBert

class TestCuda(unittest.TestCase):
    """
    Test to check that the models are loaded in the GPU
    """
    def setUp(self) -> None:
        self._gpu = '0' # Id of the GPU to check
        os.environ["CUDA_VISIBLE_DEVICES"] = self._gpu


class TestCudaPredictorTwoDimensions(TestCuda):
    def setUp(self) -> None:
        super().setUp()

    def test_cpn_gpu(self):
        cpn = CPN()
        pose_model = cpn._pose_model
        self.assertTrue(next(pose_model.parameters()).is_cuda)

        human_model = cpn._human_model
        self.assertTrue(next(human_model.parameters()).is_cuda)
    
    def test_lightweight_gpu(self):
        lightweight = Lightweight2D()
        lightweight_model = lightweight._lightweight_net
        self.assertTrue(next(lightweight_model.parameters()).is_cuda)


class TestCudaPredictorThreeDimensions(TestCuda):
    def setUp(self) -> None:
        super().setUp()

    def test_mhformer_gpu(self):
        mhformer = MHFormer()
        model = mhformer.model
        self.assertTrue(next(model.parameters()).is_cuda)
    
    def test_videopose_gpu(self):
        videopose = VideoPose3D()
        model = videopose._model_pos
        self.assertTrue(next(model.parameters()).is_cuda)
        
    def test_mixste_gpu(self):
        mixste = MixSTE()
        model = mixste._model_pos_
        self.assertTrue(next(model.parameters()).is_cuda)
        
    def test_mixste_gpu(self):
        motionbert = MotionBert()
        model = motionbert.model
        self.assertTrue(next(model.parameters()).is_cuda)
        
        
if __name__ == '__main__':
    unittest.main()
