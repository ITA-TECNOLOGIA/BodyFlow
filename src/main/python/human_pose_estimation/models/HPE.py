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

import numpy as np
from human_pose_estimation.models.HPE2D import HPE2D
from human_pose_estimation.models.HPE3D import HPE3D
from human_pose_estimation.models.PersonDetector import PersonDetector
from human_pose_estimation.models.Tracking import Tracking
from human_pose_estimation.common_pose.BodyLandmarks import BodyLandmarks3d
import logging
import copy
import cv2
import datetime

class HPE:
    """
    End-to-end human pose estimator. It needs a 2D and a 3D pose estimator which
    lifts to 3D the 2D position.

    Note: Sometimes it does not outputs all input frames,
    this is because for the people in the end of the video
    the buffer is not full and it finishes outputing those 
    keypoints of the poeople who have their buffer full
    """
    def __init__(self, hpe_2d: HPE2D, hpe_3d: HPE3D, window_length:int, person_detector: PersonDetector, tracking: Tracking):
        self._hpe_2d = hpe_2d
        self._hpe_3d = hpe_3d
        self._window_length = window_length

        #self._person_tracker = PersonDetector(trackAlgorithm) # Track algorithm: ['regular', 'deepSort']
        self._person_detector = person_detector
        self._tracking = tracking
        self._buffer = {}
        self._timestamps = []
    
    def add_frame_buffer(self, person_id, img_w, img_h, converted_keypoints, timestamp, keypoints_2d, frame, person_bbox):
        bbox = copy.deepcopy(person_bbox)
        if keypoints_2d is None: # No keypoints in this frame
            if self.buffer_empty(person_id):
                 # Skip this frame
                 return False
            else:
                # Repeat frame
                keypoints_2d = copy.deepcopy(self._buffer[person_id]["buffer"][0]["keypoints_2d"])
                bbox = copy.deepcopy(self._buffer[person_id]["buffer"][0]["bbox"]) #person_bbox
                keypoints_2d._bbox = bbox
                keypoints_2d.repeated = True
                converted_keypoints = copy.deepcopy(self._buffer[person_id]["buffer"][0]["converted_keypoints"])


        memory_value = {
                "img_w" : img_w,
                "img_h" : img_h,
                "converted_keypoints" : converted_keypoints,
                "timestamp" : timestamp,
                "keypoints_2d" : keypoints_2d,
                "frame" : frame,
                "bbox" : bbox
            }
        if person_id not in list(self._buffer.keys()):  # New person
            self._buffer[person_id] = {
                "time_without_update" : 0, # If reached certain age, should be destroyed
                "destroying" : False,
                "frame_destroying" : 0,
                "buffer" : [memory_value] * ((self._window_length // 2) + 1)
                }
        else: # Update person
            if self.buffer_full(person_id): # Buffer full, drop last frame
                self._buffer[person_id]["buffer"].pop()
            self._buffer[person_id]["buffer"].insert(0, memory_value)

        return self.buffer_full(person_id)


    def buffer_empty(self, person_id):
        if person_id not in list(self._buffer.keys()):
            return True
        else:
            return self._buffer[person_id]["buffer"] == []
    
    def buffer_full(self, person_id):
        if person_id not in list(self._buffer.keys()):
            return False
        else:
            return len(self._buffer[person_id]["buffer"]) == self._window_length

    
    def init_buffers(self, frame, timestamp) -> bool:
        any_buffer_init = self.add_frame(frame, timestamp)
        return any_buffer_init

    def destroy_buffer(self, person_id, add_frame):
        buffer = self._buffer[person_id]["buffer"]
        half_window = (self._window_length // 2)
        ########################
        #  TODO IF A BUFFER IS NOT FULL THEN IT DOES NOT OUTPUT POSE 
        #  TODO, it should be repeated to at least predict the only poses available
        ########################
        #added_artificial_frames = 0
        #while not self.buffer_full(person_id): # For last frames that buffer is not full but we want HPE
        #    self.add_frame_buffer(person_id, buffer[0]["img_w"], 
        #                            buffer[0]["img_h"],
        #                            buffer[0]["converted_keypoints"],
        #                            buffer[0]["timestamp"],
        #                            buffer[0]["keypoints_2d"])
            #added_artificial_frames += 1
        #if added_artificial_frames > 0:
        #    self._buffer[person_id]["frame_destroying"] += half_window - added_artificial_frames

        self._buffer[person_id]["destroying"] = True
        expected_timestamps = half_window + 1
        buffer_ended = self._buffer[person_id]["frame_destroying"] == expected_timestamps
        if buffer_ended:
            return None
        else:
            frame_no = (self._window_length // 2) #+ 1  # TODO sure about this???
            input_2D_no = np.array([x["converted_keypoints"] for x in buffer])
            bboxes = np.array([x["bbox"] for x in buffer])
            body_landmarks = self._hpe_3d.get_3d_keypoints(buffer[frame_no]["img_w"], 
                                                             buffer[frame_no]["img_h"],
                                                             buffer[frame_no]["keypoints_2d"],
                                                             buffer[frame_no]["timestamp"],
                                                             input_2D_no,
                                                             buffer[frame_no]["frame"],
                                                             bboxes
                                                             )
            if add_frame:
                self.add_frame_buffer(person_id, buffer[0]["img_w"], 
                                    buffer[0]["img_h"],
                                    buffer[0]["converted_keypoints"],
                                    buffer[0]["timestamp"]+1,
                                    buffer[0]["keypoints_2d"],
                                    buffer[0]["frame"],
                                    buffer[0]["bbox"]
                                    )

            self._buffer[person_id]["frame_destroying"] += 1
            return body_landmarks



    def destroy_all_buffers(self) -> BodyLandmarks3d:
        """
        It returns body landmarks until de buffer is destroyed. Once destroyed,
        it returns None
        """
        keypoints_3d_per_person = {}
        for person_id in list(self._buffer.keys()):
            if self.buffer_full(person_id):
                keypoints_3d = self.destroy_buffer(person_id, add_frame=True)
                if keypoints_3d is not None:
                    keypoints_3d_per_person[person_id] = keypoints_3d
        if keypoints_3d_per_person == {}:
            return None
        return keypoints_3d_per_person

    def predict_pose(self) -> BodyLandmarks3d:
        """
        It predicts the corresponding pose.
        """
        keypoints_3d_per_person = {}
        ids_to_remove = []
        frame_no = self._window_length // 2

        for person_id, data in self._buffer.items():
            buffer = data["buffer"]
            if self.buffer_full(person_id):
                if data["destroying"]:
                    keypoints_3d = self.destroy_buffer(person_id, add_frame=False)
                    if keypoints_3d is None:  # Buffer ended
                        ids_to_remove.append(person_id)
                    else:
                        keypoints_3d_per_person[person_id] = keypoints_3d
                else:
                    input_2D_no = np.array([x["converted_keypoints"] for x in buffer])
                    bboxes = np.array([x["bbox"] for x in buffer])
                    # Add: full image, bbox
                    keypoints_3d = self._hpe_3d.get_3d_keypoints(buffer[frame_no]["img_w"], 
                                                                buffer[frame_no]["img_h"],
                                                                buffer[frame_no]["keypoints_2d"],
                                                                buffer[frame_no]["timestamp"],
                                                                input_2D_no,
                                                                buffer[frame_no]["frame"],
                                                                bboxes,
                                                                )
                    keypoints_3d_per_person[person_id] = keypoints_3d
        
        for id_to_remove in ids_to_remove:
            self._buffer.pop(id_to_remove)

        # All people timestamps should be the same
        if keypoints_3d_per_person != {}:
            timestamps = [x.timestamp for x in list(keypoints_3d_per_person.values())]
            assert all(x==timestamps[0] for x in timestamps)
            self._timestamps = [x for x in self._timestamps if x > timestamps[0]]
            return keypoints_3d_per_person, timestamps[0]
        else:
            timestamp = self._timestamps[0]
            self._timestamps.remove(timestamp)
            return None, timestamp
    

    def add_frame(self, frame, timestamp) -> bool:
        """
        It adds a new frame into the buffer, returning true if the buffer is complete,
        meaning that it overrides the oldest frame. If false, it means the that buffer is
        not still full.
        """
        # Predicts where peope are and identify them (Person detector + tracking)
        bboxs, scores = self._person_detector.predict(frame)
        # TODO update anyway? To increase age time withot observing 
        bounding_boxes = None if bboxs is None else self._tracking.update(bboxs, scores, frame)

        img_size = frame.shape
        img_w = img_size[1]
        img_h = img_size[0]
        not_keypoints = []

        any_buffer_init = False
        if bounding_boxes is not None:
            for person_id, person_bbox in bounding_boxes.items():
                # https://stackoverflow.com/questions/56115874/how-to-convert-bounding-box-x1-y1-x2-y2-to-yolo-style-x-y-w-h
                x1, y1, x2, y2 = person_bbox
                # May can be out of the image boundaries
                x1 = np.clip(x1, 0, img_w)
                x2 = np.clip(x2, 0, img_w)
                y1 = np.clip(y1, 0, img_h)
                y2 = np.clip(y2, 0, img_h)
                person_bbox = [x1, y1, x2, y2]
                bounding_boxes[person_id] = copy.deepcopy(person_bbox)
                
                if not (int(x1) < int(x2) and int(y1) < int(y2)):
                    not_keypoints.append(person_id)
                    continue
                cropped_frame = frame[int(y1):int(y2), int(x1):int(x2)]
                keypoints_2d = self._hpe_2d.get_frame_keypoints(frame, cropped_frame, person_bbox) # Note that person bbox does not match with frame, it is used for offset keypoints 2d
                if keypoints_2d is None: # Maybe the pose detector does not detect the keypoints in the given frame
                    not_keypoints.append(person_id)
                    continue
                converted_keypoints = self._hpe_3d.translate_keypoints_2d(keypoints_2d)
                """
                # Create a copy of the image
                image_with_keypoints = frame.copy()

                # Loop through the keypoints and draw circles on the image
                for keypoint in converted_keypoints:
                    x, y = keypoint
                    cv2.circle(image_with_keypoints, (int(x), int(y)), 3, (0, 0, 255), -1)

                # Get the current timestamp
                timestamppp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

                # Save the image with keypoints using the unique timestamp
                cv2.imwrite(f"debug/keypointsss_{timestamppp}.png", image_with_keypoints)
                """

                person_buffer_init = self.add_frame_buffer(person_id, img_w, img_h, converted_keypoints, timestamp, keypoints_2d, frame, person_bbox)
                any_buffer_init = any_buffer_init or person_buffer_init
                self._buffer[person_id]["time_without_update"] = 0

        for person_id in list(self._buffer.keys()):
            if bounding_boxes is not None:
                if (person_id not in list(bounding_boxes.keys())) or (person_id in not_keypoints):  # One of the stored person is not detected
                    self._buffer[person_id]["time_without_update"] += 1
                    person_buffer_init = self.add_frame_buffer(person_id, img_w, img_h, None, timestamp, None, frame, None)
                    any_buffer_init = any_buffer_init or person_buffer_init
            else:
                self._buffer[person_id]["time_without_update"] += 1
                person_buffer_init = self.add_frame_buffer(person_id, img_w, img_h, None, timestamp, None, frame, None)

            # This person will never appear again, should be removed
            if self._buffer[person_id]["time_without_update"] > self._tracking.max_age:
                self._buffer[person_id]["destroying"] = True
        
        timestamps = [x["buffer"][0]["timestamp"] for x in list(self._buffer.values())]
        assert all(x==timestamps[0] for x in timestamps)
        self._timestamps.append(timestamp)

        return any_buffer_init
            