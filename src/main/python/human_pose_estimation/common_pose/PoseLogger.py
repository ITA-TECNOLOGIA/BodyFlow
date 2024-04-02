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

import zmq
from human_pose_estimation.common_pose.BodyLandmarks import BodyLandmarks3d
import os
from datetime import datetime
import csv
from collections.abc import MutableMapping
import logging

def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str ='.') -> MutableMapping:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class PoseLogger:
    """
    This class saves the poses in a list and, when indicated, it stores all of them
    in a .csv. It also sends the poses to and endpoint, if specified.
    """
    def __init__(self, port: int, filename=None):
        context = zmq.Context()
        self._port = port
        if port != -1:
            self._socket = context.socket(zmq.REP)
            self._socket.bind(f'tcp://*:{port}')

        # Logging
        self._poses = []
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True) # Create dataset directory
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if filename is None:
            self._filename = os.path.join(logs_dir, f"log_{timestamp}.csv")
        else:
            self._filename = os.path.join(logs_dir,filename)

    def send_pose(self, body_landmarks: BodyLandmarks3d):
        """
        It sends the poses to an endpoint.
        """
        """
        TODO: THIS DOES NOT WORK ANYMORE, msg should be formatted
        """
        self._socket.recv()
        self._socket.send_json(body_landmarks.get_msg())

    def log_pose(self, body_landmarks: BodyLandmarks3d):
        """
        It stores the pose in a list and, if specified, sends them to an endopoint.
        """
        if body_landmarks is not None:
            self._poses.append(body_landmarks)
            if self._port != -1:
                self.send_pose(body_landmarks)
    
    def get_poses(self):
        return self._poses

    def export_csv(self):
        """
        Export poses into a .csv.
        """
        logging.info("Exporting %d poses to a csv...", len(self._poses))
        with open(self._filename, 'w') as f:
            writer = csv.writer(f)
            header = []
            for pose in self._poses:
                for person_id, bodyLandmarks in pose.items():
                    pose_dict = flatten_dict(bodyLandmarks.get_msg())
                    if header == []:
                        header = list(pose_dict.keys())
                        header = ['timestamp', 'id'] + header
                        writer.writerow(header)

                    row = [bodyLandmarks.timestamp, person_id]
                    row.extend(list(pose_dict.values()))
                    writer.writerow(row)
        
        logging.info("Poses exported to %s", self._filename)
        