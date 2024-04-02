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

"""
Mediapipe3D. It uses the Mediapipe library which given an RGB image, it is able to return the 3D joints.
NOTE: THIS DOES NOT NEED 2D JOINTS, THIS IS AN END-TO-END MODEL (use dummy2d to leverage full potential)
"""
from human_pose_estimation.models.HPE3D import HPE3D
import cv2
import mediapipe as mp
from human_pose_estimation.common_pose.BodyLandmarks import BodyLandmarks3d, Landmark

class Mediapipe3D(HPE3D):
    def __init__(self, window_length=1):
        super().__init__(window_length)
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    def translate_keypoints_2d(self, keypoints_2d):
        return None
    

    def bodyLandmarks3d_None(self, timestamp, bodyLandmarks2d):
        body_landmarks = BodyLandmarks3d(
                                        raw=None,
                                        timestamp=timestamp,
                                        nose=Landmark("nose", "3d", None, None),
                                        left_eye_inner=Landmark("left_eye_inner", "3d", None, None),
                                        left_eye=Landmark("left_eye", "3d", None, None),
                                        left_eye_outer=Landmark("left_eye_outer", "3d", None, None),
                                        right_eye_inner=Landmark("right_eye_inner", "3d", None, None),
                                        right_eye=Landmark("right_eye", "3d", None, None),
                                        right_eye_outer=Landmark("right_eye_outer", "3d", None, None),
                                        left_ear=Landmark("left_ear", "3d", None, None),
                                        right_ear=Landmark("right_ear", "3d", None, None),
                                        mouth_left=Landmark("mouth_left", "3d", None, None),
                                        mouth_right=Landmark("mouth_right", "3d", None, None),
                                        left_shoulder=Landmark("left_shoulder", "3d", None, None),
                                        right_shoulder=Landmark("right_shoulder", "3d", None, None),
                                        left_elbow=Landmark("left_elbow", "3d", None, None),
                                        right_elbow=Landmark("right_elbow", "3d", None, None),
                                        left_wrist=Landmark("left_wrist", "3d", None, None),
                                        right_wrist=Landmark("right_wrist", "3d", None, None),
                                        left_pinky=Landmark("left_pinky", "3d", None, None),
                                        right_pinky=Landmark("right_pinky", "3d", None, None),
                                        left_index=Landmark("left_index", "3d", None, None),
                                        right_index=Landmark("right_index", "3d", None, None),
                                        left_thumb=Landmark("left_thumb", "3d", None, None),
                                        right_thumb=Landmark("right_thumb", "3d", None, None),
                                        left_hip=Landmark("left_hip", "3d", None, None),
                                        right_hip=Landmark("right_hip", "3d", None, None),
                                        left_knee=Landmark("left_knee", "3d", None, None),
                                        right_knee=Landmark("right_knee", "3d", None, None),
                                        left_ankle=Landmark("left_ankle", "3d", None, None),
                                        right_ankle=Landmark("right_ankle", "3d", None, None),
                                        left_heel=Landmark("left_heel", "3d", None, None),
                                        right_heel=Landmark("right_heel", "3d", None, None),
                                        left_foot_index=Landmark("left_foot_index", "3d", None, None),
                                        right_foot_index=Landmark("right_foot_index", "3d", None, None),
                                        jaw=Landmark("jaw", "3d", None, None),
                                        chest=Landmark("chest", "3d", None, None),
                                        spine=Landmark("spine", "3d", None, None),
                                        hips=Landmark("hips", "3d", None, None),
                                        # 2D Keypoints
                                        bodyLandmarks2d=bodyLandmarks2d
        )
        return body_landmarks


    def get_3d_keypoints(self, img_w, img_h, bodyLandmarks2d, timestamp, input_2D_no, frame, bboxes):
        assert len(bboxes) == 1, f"Mediapipe only works with window time = 1"
        if bboxes[0] is None:
            return self.bodyLandmarks3d_None(timestamp, bodyLandmarks2d)
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = bboxes[0]
        cropped_frame = RGB[int(y1):int(y2), int(x1):int(x2)]
        # process the RGB frame to get the result
        results = self._pose.process(cropped_frame)
        if results.pose_world_landmarks is None:
            return self.bodyLandmarks3d_None(timestamp, bodyLandmarks2d)
        landmarks = results.pose_world_landmarks.landmark

        frame_keypoints_3d = {
            'nose': [landmarks[self._mp_pose.PoseLandmark.NOSE].x,
                    landmarks[self._mp_pose.PoseLandmark.NOSE].y,
                    landmarks[self._mp_pose.PoseLandmark.NOSE].z],
            'left_eye_inner': [landmarks[self._mp_pose.PoseLandmark.LEFT_EYE_INNER].x,
                            landmarks[self._mp_pose.PoseLandmark.LEFT_EYE_INNER].y,
                            landmarks[self._mp_pose.PoseLandmark.LEFT_EYE_INNER].z],
            'left_eye': [landmarks[self._mp_pose.PoseLandmark.LEFT_EYE].x,
                        landmarks[self._mp_pose.PoseLandmark.LEFT_EYE].y,
                        landmarks[self._mp_pose.PoseLandmark.LEFT_EYE].z],
            'left_eye_outer': [landmarks[self._mp_pose.PoseLandmark.LEFT_EYE_OUTER].x,
                            landmarks[self._mp_pose.PoseLandmark.LEFT_EYE_OUTER].y,
                            landmarks[self._mp_pose.PoseLandmark.LEFT_EYE_OUTER].z],
            'right_eye_inner': [landmarks[self._mp_pose.PoseLandmark.RIGHT_EYE_INNER].x,
                                landmarks[self._mp_pose.PoseLandmark.RIGHT_EYE_INNER].y,
                                landmarks[self._mp_pose.PoseLandmark.RIGHT_EYE_INNER].z],
            'right_eye': [landmarks[self._mp_pose.PoseLandmark.RIGHT_EYE].x,
                        landmarks[self._mp_pose.PoseLandmark.RIGHT_EYE].y,
                        landmarks[self._mp_pose.PoseLandmark.RIGHT_EYE].z],
            'right_eye_outer': [landmarks[self._mp_pose.PoseLandmark.RIGHT_EYE_OUTER].x,
                                landmarks[self._mp_pose.PoseLandmark.RIGHT_EYE_OUTER].y,
                                landmarks[self._mp_pose.PoseLandmark.RIGHT_EYE_OUTER].z],
            'left_ear': [landmarks[self._mp_pose.PoseLandmark.LEFT_EAR].x,
                        landmarks[self._mp_pose.PoseLandmark.LEFT_EAR].y,
                        landmarks[self._mp_pose.PoseLandmark.LEFT_EAR].z],
            'right_ear': [landmarks[self._mp_pose.PoseLandmark.RIGHT_EAR].x,
                        landmarks[self._mp_pose.PoseLandmark.RIGHT_EAR].y,
                        landmarks[self._mp_pose.PoseLandmark.RIGHT_EAR].z],
            'mouth_left': [landmarks[self._mp_pose.PoseLandmark.MOUTH_LEFT].x,
                        landmarks[self._mp_pose.PoseLandmark.MOUTH_LEFT].y,
                        landmarks[self._mp_pose.PoseLandmark.MOUTH_LEFT].z],
            'mouth_right': [landmarks[self._mp_pose.PoseLandmark.MOUTH_RIGHT].x,
                            landmarks[self._mp_pose.PoseLandmark.MOUTH_RIGHT].y,
                            landmarks[self._mp_pose.PoseLandmark.MOUTH_RIGHT].z],
            'left_shoulder': [landmarks[self._mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                            landmarks[self._mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                            landmarks[self._mp_pose.PoseLandmark.LEFT_SHOULDER].z],
            'right_shoulder': [landmarks[self._mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                            landmarks[self._mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                            landmarks[self._mp_pose.PoseLandmark.RIGHT_SHOULDER].z],
            'left_elbow': [landmarks[self._mp_pose.PoseLandmark.LEFT_ELBOW].x,
                        landmarks[self._mp_pose.PoseLandmark.LEFT_ELBOW].y,
                        landmarks[self._mp_pose.PoseLandmark.LEFT_ELBOW].z],
            'right_elbow': [landmarks[self._mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                            landmarks[self._mp_pose.PoseLandmark.RIGHT_ELBOW].y,
                            landmarks[self._mp_pose.PoseLandmark.RIGHT_ELBOW].z],
            'left_wrist': [landmarks[self._mp_pose.PoseLandmark.LEFT_WRIST].x,
                        landmarks[self._mp_pose.PoseLandmark.LEFT_WRIST].y,
                        landmarks[self._mp_pose.PoseLandmark.LEFT_WRIST].z],
            'right_wrist': [landmarks[self._mp_pose.PoseLandmark.RIGHT_WRIST].x,
                            landmarks[self._mp_pose.PoseLandmark.RIGHT_WRIST].y,
                            landmarks[self._mp_pose.PoseLandmark.RIGHT_WRIST].z],
            'left_pinky': [landmarks[self._mp_pose.PoseLandmark.LEFT_PINKY].x,
                        landmarks[self._mp_pose.PoseLandmark.LEFT_PINKY].y,
                        landmarks[self._mp_pose.PoseLandmark.LEFT_PINKY].z],
            'right_pinky': [landmarks[self._mp_pose.PoseLandmark.RIGHT_PINKY].x,
                            landmarks[self._mp_pose.PoseLandmark.RIGHT_PINKY].y,
                            landmarks[self._mp_pose.PoseLandmark.RIGHT_PINKY].z],
            'left_index': [landmarks[self._mp_pose.PoseLandmark.LEFT_INDEX].x,
                        landmarks[self._mp_pose.PoseLandmark.LEFT_INDEX].y,
                        landmarks[self._mp_pose.PoseLandmark.LEFT_INDEX].z],
            'right_index': [landmarks[self._mp_pose.PoseLandmark.RIGHT_INDEX].x,
                            landmarks[self._mp_pose.PoseLandmark.RIGHT_INDEX].y,
                            landmarks[self._mp_pose.PoseLandmark.RIGHT_INDEX].z],
            'left_thumb': [landmarks[self._mp_pose.PoseLandmark.LEFT_THUMB].x,
                        landmarks[self._mp_pose.PoseLandmark.LEFT_THUMB].y,
                        landmarks[self._mp_pose.PoseLandmark.LEFT_THUMB].z],
            'right_thumb': [landmarks[self._mp_pose.PoseLandmark.RIGHT_THUMB].x,
                            landmarks[self._mp_pose.PoseLandmark.RIGHT_THUMB].y,
                            landmarks[self._mp_pose.PoseLandmark.RIGHT_THUMB].z],
            'left_hip': [landmarks[self._mp_pose.PoseLandmark.LEFT_HIP].x,
                        landmarks[self._mp_pose.PoseLandmark.LEFT_HIP].y,
                        landmarks[self._mp_pose.PoseLandmark.LEFT_HIP].z],
            'right_hip': [landmarks[self._mp_pose.PoseLandmark.RIGHT_HIP].x,
                        landmarks[self._mp_pose.PoseLandmark.RIGHT_HIP].y,
                        landmarks[self._mp_pose.PoseLandmark.RIGHT_HIP].z],
            'left_knee': [landmarks[self._mp_pose.PoseLandmark.LEFT_KNEE].x,
                        landmarks[self._mp_pose.PoseLandmark.LEFT_KNEE].y,
                        landmarks[self._mp_pose.PoseLandmark.LEFT_KNEE].z],
            'right_knee': [landmarks[self._mp_pose.PoseLandmark.RIGHT_KNEE].x,
                        landmarks[self._mp_pose.PoseLandmark.RIGHT_KNEE].y,
                        landmarks[self._mp_pose.PoseLandmark.RIGHT_KNEE].z],
            'left_ankle': [landmarks[self._mp_pose.PoseLandmark.LEFT_ANKLE].x,
                        landmarks[self._mp_pose.PoseLandmark.LEFT_ANKLE].y,
                        landmarks[self._mp_pose.PoseLandmark.LEFT_ANKLE].z],
            'right_ankle': [landmarks[self._mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                            landmarks[self._mp_pose.PoseLandmark.RIGHT_ANKLE].y,
                            landmarks[self._mp_pose.PoseLandmark.RIGHT_ANKLE].z],
            'left_heel': [landmarks[self._mp_pose.PoseLandmark.LEFT_HEEL].x,
                        landmarks[self._mp_pose.PoseLandmark.LEFT_HEEL].y,
                        landmarks[self._mp_pose.PoseLandmark.LEFT_HEEL].z],
            'right_heel': [landmarks[self._mp_pose.PoseLandmark.RIGHT_HEEL].x,
                        landmarks[self._mp_pose.PoseLandmark.RIGHT_HEEL].y,
                        landmarks[self._mp_pose.PoseLandmark.RIGHT_HEEL].z],
            'left_foot_index': [landmarks[self._mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x,
                                landmarks[self._mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y,
                                landmarks[self._mp_pose.PoseLandmark.LEFT_FOOT_INDEX].z],
            'right_foot_index': [landmarks[self._mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x,
                                landmarks[self._mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y,
                                landmarks[self._mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].z]
        }

        return self.pose_to_landmarks(frame_keypoints_3d, bodyLandmarks2d, timestamp)
    

    def pose_to_landmarks(self, pose, bodyLandmarks2d, timestamp):
        raw = pose

        # Additional keypoints
        hips_x = (pose['left_hip'][0] + pose['right_hip'][0]) / 2
        hips_y = (pose['left_hip'][1] + pose['right_hip'][1]) / 2
        hips_z = (pose['left_hip'][2] + pose['right_hip'][2]) / 2
        hips = [hips_x, hips_y, hips_z]

        chest_x = (pose['left_shoulder'][0] + pose['right_shoulder'][0]) / 2
        chest_y = (pose['left_shoulder'][1] + pose['right_shoulder'][1]) / 2
        chest_z = (pose['left_shoulder'][2] + pose['right_shoulder'][2]) / 2
        chest = [chest_x, chest_y, chest_z]

        spine_x = (hips_x + chest_x) / 2
        spine_y = (hips_y + chest_y) / 2
        spine_z = (hips_z + chest_z) / 2
        spine = [spine_x, spine_y, spine_z]

        jaw_x = (pose['mouth_right'][0] + pose['mouth_left'][0]) / 2
        jaw_y = (pose['mouth_right'][1] + pose['mouth_left'][1]) / 2
        jaw_z = (pose['mouth_right'][2] + pose['mouth_left'][2]) / 2
        jaw = [jaw_x, jaw_y, jaw_z]

        body_landmarks = BodyLandmarks3d(
                                        raw=raw,
                                        timestamp=timestamp,
                                        nose=Landmark("nose", "3d", pose['nose'], False),
                                        left_eye_inner=Landmark("left_eye_inner", "3d", pose['left_eye_inner'], False),
                                        left_eye=Landmark("left_eye", "3d", pose['left_eye'], False),
                                        left_eye_outer=Landmark("left_eye_outer", "3d", pose['left_eye_outer'], False),
                                        right_eye_inner=Landmark("right_eye_inner", "3d", pose['right_eye_inner'], False),
                                        right_eye=Landmark("right_eye", "3d", pose['right_eye'], False),
                                        right_eye_outer=Landmark("right_eye_outer", "3d", pose['right_eye_outer'], False),
                                        left_ear=Landmark("left_ear", "3d", pose['left_ear'], False),
                                        right_ear=Landmark("right_ear", "3d", pose['right_ear'], False),
                                        mouth_left=Landmark("mouth_left", "3d", pose['mouth_left'], False),
                                        mouth_right=Landmark("mouth_right", "3d", pose['mouth_right'], False),
                                        left_shoulder=Landmark("left_shoulder", "3d", pose['left_shoulder'], False),
                                        right_shoulder=Landmark("right_shoulder", "3d", pose['right_shoulder'], False),
                                        left_elbow=Landmark("left_elbow", "3d", pose['left_elbow'], False),
                                        right_elbow=Landmark("right_elbow", "3d", pose['right_elbow'], False),
                                        left_wrist=Landmark("left_wrist", "3d", pose['left_wrist'], False),
                                        right_wrist=Landmark("right_wrist", "3d", pose['right_wrist'], False),
                                        left_pinky=Landmark("left_pinky", "3d", pose['left_pinky'], False),
                                        right_pinky=Landmark("right_pinky", "3d", pose['right_pinky'], False),
                                        left_index=Landmark("left_index", "3d", pose['left_index'], False),
                                        right_index=Landmark("right_index", "3d", pose['right_index'], False),
                                        left_thumb=Landmark("left_thumb", "3d", pose['left_thumb'], False),
                                        right_thumb=Landmark("right_thumb", "3d", pose['right_thumb'], False),
                                        left_hip=Landmark("left_hip", "3d", pose['left_hip'], False),
                                        right_hip=Landmark("right_hip", "3d", pose['right_hip'], False),
                                        left_knee=Landmark("left_knee", "3d", pose['left_knee'], False),
                                        right_knee=Landmark("right_knee", "3d", pose['right_knee'], False),
                                        left_ankle=Landmark("left_ankle", "3d", pose['left_ankle'], False),
                                        right_ankle=Landmark("right_ankle", "3d", pose['right_ankle'], False),
                                        left_heel=Landmark("left_heel", "3d", pose['left_heel'], False),
                                        right_heel=Landmark("right_heel", "3d", pose['right_heel'], False),
                                        left_foot_index=Landmark("left_foot_index", "3d", pose['left_foot_index'], False),
                                        right_foot_index=Landmark("right_foot_index", "3d", pose['right_foot_index'], False),
                                        jaw=Landmark("jaw", "3d", jaw, True),
                                        chest=Landmark("chest", "3d", chest, True),
                                        spine=Landmark("spine", "3d", spine, True),
                                        hips=Landmark("hips", "3d", hips, True),
                                        # 2D Keypoints
                                        bodyLandmarks2d=bodyLandmarks2d
        )

        return body_landmarks
        