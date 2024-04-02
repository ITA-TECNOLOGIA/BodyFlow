#!/usr/bin/env python

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

from setuptools import setup, find_packages
setup(
    name='bodyflow',
    version='1.0.32',
    description='BodyFlow ITA',
    author='cmaranes, ahernandez, ilopez, agimeno, mvega, psalvo, rhoyo, raznar',
    author_email='cmaranes@ita.es;ahernandez@ita.es;ilopez@ita.es',
    url='https://git.itainnova.es/bigdata/misc/bodyflow.git',    
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10.11",
    data_files =[],
    long_description="""\
      BodyFlow is a human pose estimation and activity recognition open source repo.
      """,
      classifiers=[
          "License :: OSI Approved :: MIT License",
          "Programming Language :: Python",
          "Development Status :: 4 - Beta",
          "Intended Audience :: Developers",
          "Topic :: Internet",
      ],
      keywords=['human pose estimation', 'human activity recognition', 'tracking', 'multipose estimation'],
      license='MIT License',
      install_requires=['numpy>=1.21.6', 'Cython>=0.29.32', 'pycocotools==2.0.6', 'opencv-python>=4.5.5.64',
                        'mediapipe>=0.8.9.1', 'gdown>=4.4.0', 'yacs>=0.1.8', 'numba>=0.53.1', 'scikit-image>=0.17.2',
                        'filterpy>=1.4.5', 'ipython>=7.16.3', 'einops>=0.4.1', 'timm>=0.5.4', 'pandas>=1.3.5', 'zmq==0.0.0',
                        'local-attention==1.4.3', 'chardet==5.1.0', 'scikit-learn==1.0.2', 'scipy==1.7.3',
                        'pytorch-lightning==1.5', 'mlflow==1.30', 'torchmetrics==0.11.1', 'seaborn==0.12.2', 'deep-sort-realtime==1.3.2',
                        'open3d==0.17.0', 'loguru==0.7.0', 'smplx==0.1.28', 'fvcore==0.1.5.post20221221', 'portalocker==2.7.0',
                        'termcolor==2.3.0', 'iopath==0.1.10', 'trimesh==3.21.5', 'PyOpenGL==3.1.0', 'freetype-py==2.3.0',
                        'pyglet==1.5.27', 'pyrender==0.1.45', 'torchvision==0.13']
    )
