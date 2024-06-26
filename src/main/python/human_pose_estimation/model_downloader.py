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

import gdown
import os
import urllib.request

def download_models():
    """
    Download all the models weights needed for the correct performance of the algorithms.
    """
    # If google drive throws access denied error, run:
    # pip install -U --no-cache-dir gdown --pre
    os.makedirs(os.path.join("data", "input"), exist_ok=True) # Create input directory
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True) # Create models directory
    # ---------- MHFormer model
    # https://github.com/Vegetebird/MHFormer
    url = 'https://drive.google.com/file/d/1kaznEDkcElFT2s_NmLDnJESvNa2RNplJ/view?usp=sharing'
    output = os.path.join(models_dir, 'mhformer_model_351.pth')
    gdown.download(id="1kaznEDkcElFT2s_NmLDnJESvNa2RNplJ", output=output, quiet=False, use_cookies=False)

    url = 'https://drive.google.com/file/d/1cV3vYx7uCFMu4D5NYvKqfMtL6OzQXTMA/view?usp=sharing'
    output = os.path.join(models_dir, 'mhformer_model_243.pth')
    gdown.download(id="1cV3vYx7uCFMu4D5NYvKqfMtL6OzQXTMA", output=output, quiet=False, use_cookies=False)

    url = 'https://drive.google.com/file/d/1Qg7BcPpDGdrSQzezLFWRa9p9V-bxlWKo/view?usp=sharing'
    output = os.path.join(models_dir, 'mhformer_model_81.pth')
    gdown.download(id="1Qg7BcPpDGdrSQzezLFWRa9p9V-bxlWKo", output=output, quiet=False, use_cookies=False)

    url = 'https://drive.google.com/file/d/1MKCa3voZNxpF71h22447nHeGc1wRkF8m/view?usp=sharing'
    output = os.path.join(models_dir, 'mhformer_model_27.pth')
    gdown.download(id="1MKCa3voZNxpF71h22447nHeGc1wRkF8m", output=output, quiet=False, use_cookies=False)

    # ---------- CPN (MHFormer 2D detector)
    url = 'https://drive.google.com/drive/folders/1_ENAMOsPM7FXmdYRbkwbFHgzQq_B_NQA'
    # Yolo
    output = os.path.join(models_dir, 'yolov3.weights')
    gdown.download(id="1gWZl1VrlLZKBf0Pfkj4hKiFxe8sHP-1C", output=output, quiet=False, use_cookies=False)
    # Pose HRNet
    output = os.path.join(models_dir, 'pose_hrnet_w48_384x288.pth')
    gdown.download(id="1CpyZiUIUlEjiql4rILwdBT4666S72Oq4", output=output, quiet=False, use_cookies=False)

    # Lightweight (2D detector)
    # https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch
    url = "https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth"
    urllib.request.urlretrieve(url, os.path.join(models_dir,'lightweight_checkpoint_iter_370000.pth'))


if __name__ == "__main__":
    download_models()
    print("Models downloaded!")