# Pose Estimation Algorithms

This document provides detailed information about each 2D and 3D pose estimation algorithm available in the **bodyflow** repository. It aims to help you decide which algorithms best suit your custom 3D pose detection pipeline.

## Table of Contents

- [2D Pose Estimation Predictors](#2d-pose-estimation-predictors)
  - [1. MediaPipe](#1-mediapipe)
  - [2. Cascade Pyramid Network (CPN)](#2-cascade-pyramid-network-cpn)
  - [3. Lightweight Pose Estimation](#3-lightweight-pose-estimation)
- [3D Pose Estimation Predictors](#3d-pose-estimation-predictors)
  - [1. MHFormer](#1-mhformer)
  - [2. MediaPipe (3D)](#2-mediapipe-3d)
  - [3. MotionBert](#3-motionbert)
  - [4. VideoPose3D](#4-videopose3d)
  - [5. MixSTE](#5-mixste)
  - [6. ExPose](#6-expose)
- [How to Choose the Right Algorithms](#how-to-choose-the-right-algorithms)
- [Installation Instructions](#installation-instructions)



## 2D Pose Estimation Predictors

### 1. MediaPipe

**Status**: *Included*

**Description**: MediaPipe is a cross-platform framework developed by Google that provides high-fidelity 2D pose estimation. It provides efficient performance using minimal computational resources, running effectively on low-end devices with limited VRAM (approximately 1.5 GB).

**Features**:

- **Fast Processing**: Optimized for speed, suitable for applications requiring immediate feedback.
- **Lightweight**: Designed to run efficiently on resource-constrained devices.
- **Feet Model**: Mediapipe also includes foot pose prediction, not present in all models.

**Considerations**:

- **Accuracy**: While fast, it may not be as accurate as more complex models in challenging scenarios.

**Best Suited For**:

- Applications needing quick 2D pose estimation.
- Web applications with limited computational resources.
- When foot estimation is needed.

---

### 2. Cascade Pyramid Network (CPN)

**Status**: *Included*

**Description**: CPN is a top-down 2D pose estimation framework that refines keypoint predictions through a cascade of pyramid networks, improving accuracy especially for hard-to-detect joints.

**Features**:

- **High Accuracy**: Excels in detecting keypoints even in complex poses or occlusions.
- **Robustness**: Performs well across various challenging scenarios.

**Considerations**:

- **Computational Demand**: More resource-intensive than lightweight models.
- **Inference Speed**: Slower compared to models like MediaPipe.

**Best Suited For**:

- Applications where accuracy is critical.
- Scenarios involving complex human poses or occlusions.

---

### 3. Lightweight Pose Estimation

**Status**: *Included*

**Description**: An efficient, multi-person 2D pose estimation model designed for speed and minimal computational cost, making it suitable for applications with very limited hardware capabilities.

**Features**:

- **Efficiency**: Optimized for minimal computational resources. Ideal for devices with limited processing power or VRAM.

**Considerations**:

- **Accuracy**: May not perform as well as heavier models in complex situations.
- **Detail**: Less precise in detecting fine-grained keypoint movements.

**Best Suited For**:

- Applications requiring fast processing.
- Use cases with strict hardware limitations.



## 3D Pose Estimation Predictors

### 1. MHFormer

**Status**: *Included*

**Description**: MHFormer is a transformer-based model that leverages multi-hypothesis predictions for 3D human pose estimation from video sequences. Unlike traditional models that predict a single 3D pose, MHFormer generates multiple hypotheses for the 3D pose estimation at each frame, better capturing depth ambiguities and occlusions.

**Features**:

- **High Accuracy**: Provides precise 3D pose estimations.
- **Temporal Modeling**: Utilizes sequential video frames to improve prediction consistency over time, understanding motion patterns and dynamics whilst reducing jitter across frames.

**Considerations**:

- **Resource Intensive**: Requires significant computational power.
- **Input Requirement**: Designed for video input rather than single images.

**Best Suited For**:

- Applications needing accurate 3D pose estimation from videos.
- Research projects focusing on advanced human motion analysis.

---

### 2. MediaPipe (3D)

**Status**: *Included*

**Description**: An extension of MediaPipe's 2D pose estimation to 3D, providing efficient 3D keypoint detection.

**Features**:

- **Fast Processing**: Maintains high speed suitable for applications requiring fast response.
- **Integrated Solution**: Combines 2D and 3D estimation in a single framework, no need to select a 2D model.
- **Feet Model**: Mediapipe also includes foot pose prediction.

**Considerations**:

- **Accuracy**: Less accurate compared to specialized 3D models.
- **Depth Precision**: May struggle with precise depth estimations in complex scenes.

**Best Suited For**:

- Applications requiring quick 3D pose estimation without intensive computation.
- Prototyping and demos where ease of use is a priority.
- When foot estimation is needed.

---

### 3. MotionBert

**Status**: *Requires Installation*

**Installation Instructions**: [MotionBert Installation](src/main/python/human_pose_estimation/models/predictors_3d/motionbert/MotionBert_installation.md)

**Description**: MotionBert is a transformer-based model that learns motion representations for 3D human pose estimation from videos. The authors present a unified perspective on tackling various human-centric video tasks by learning human motion representations from large-scale and heterogeneous data resources.

**Features**:

- **Motion Representation Learning**: Excels in capturing temporal dynamics.
- **High Accuracy**: Provides detailed and accurate 3D poses.

**Considerations**:

- **Complex Setup**: Requires additional installation steps in Bodyflow.
- **Computational Resources**: Demands powerful hardware for optimal performance.

**Best Suited For**:

- Advanced applications in motion analysis.
- Projects that can accommodate additional setup complexity.

---

### 4. VideoPose3D

**Status**: *Requires Installation*

**Installation Instructions**: [VideoPose3D Installation](src/main/python/human_pose_estimation/models/predictors_3d/videopose/VideoPose3D_installation.md)

**Description**: VideoPose3D lifts 2D keypoints to 3D space using temporal convolutional networks, processing entire video sequences to improve accuracy and produce smoother 3D pose estimations.

**Features**:

- **Temporal Convolution**: Utilizes temporal information for smoother predictions.
- **Efficiency**: Balances between accuracy and computational demand.

**Considerations**:

- **Video Input**: Requires sequential frames for optimal performance.
- **Installation**: Additional steps needed to integrate into your pipeline.
- **Quality of 2D Keypoints**: Relies on the accuracy of input 2D keypoints.

**Best Suited For**:

- Applications where video data is available.
- Ideal when high-quality 2D keypoint data is available, allowing the model to focus on 3D keypoint lifting.

---

### 5. MixSTE

**Status**: *Requires Installation*

**Installation Instructions**: [MixSTE Installation](src/main/python/human_pose_estimation/models/predictors_3d/mixste/MixSTE_installation.md)

**Description**: MixSTE is a state-of-the-art transformer that mixes spatio-temporal embeddings, simultaneously learning relationships between joints and temporal dynamics.

**Features**:

- **High Precision**: Delivers top-tier accuracy in pose estimation.
- **Advanced Modeling**: Employs sophisticated techniques for capturing spatial and temporal dependencies.

**Considerations**:

- **Resource Demand**: Requires significant computational power.
- **Complexity**: May be overkill for simple applications.

**Best Suited For**:

- Projects where the highest accuracy is necessary.
- Applications that can leverage powerful hardware setups.

---

### 6. ExPose

**Status**: *Requires Installation*

**Installation Instructions**: [ExPose Installation](src/main/python/human_pose_estimation/models/predictors_3d/expose_lib/ExPose_installation.md)

**Description**: ExPose estimates expressive 3D human poses, including facial expressions and hand gestures, providing a comprehensive model of the human body.

**Features**:

- **Comprehensive Modeling**: Captures body pose, facial expressions, and hand articulations.
- **Detail-Rich Output**: Ideal for applications needing fine-grained human models, generating detailed 3D meshes that can be directly used in animation pipelines.
- **Integrated Solution**: Combines 2D and 3D estimation in a single framework, no need to select a 2D model.
- **Feet Model**: Mediapipe also includes foot pose prediction.

**Considerations**:

- **Computationally Intensive**: Demands robust hardware. It's the most demanding model available.
- **Installation Complexity**: Requires careful setup and configuration.

**Best Suited For**:

- Virtual reality and animation projects.
- Applications requiring detailed human interaction modeling.


## How to Choose the Right Algorithms

When selecting the appropriate 2D and 3D predictors for your project, consider the following factors:

1. **Application Requirements**:
   - **Fast Performance**: Choose MediaPipe or Lightweight models for immediate feedback.
   - **High Accuracy**: Opt for CPN (2D) and MHFormer or MixSTE (3D) for precision.

2. **Computational Resources**:
   - **Limited Hardware**: Use lightweight models that are less resource-intensive.
   - **Powerful Hardware**: Leverage advanced models like MixSTE or ExPose for detailed estimations.

3. **Input Data Type**:
   - **Single Images**: Ensure the model can handle non-sequential data.
   - **Video Sequences**: Use models that exploit temporal information for better accuracy.

4. **Complexity and Customization**:
   - **Ease of Use**: Included models are ready to use with minimal setup.
   - **Advanced Features**: Models requiring installation offer more customization at the cost of complexity.

5. **Use Case Specifics**:
   - **Facial Expressions and Hands**: ExPose is specialized for detailed modeling.
   - **Feet Detection**: Ensure the model has feet keypoint detection if needed.


## Example Combinations and Their Use Cases

Selecting the right combination of 2D and 3D predictors is crucial for optimizing performance and accuracy based on your application's requirements. Here are some example combinations and the contexts in which they are most effective. We have selected, if the algorithm has that option, a window size of 243. Sample video has been extracted from [here](https://www.pexels.com/es-es/video/una-mujer-bailando-libremente-al-aire-libre-sobre-un-pavimento-de-hormigon-3044160/).

### 1. **MediaPipe (3D)**

**Use Case**: Fast response applications on resource-constrained devices. Note that no 2D detector is being used (Dummy2D), because MediaPipe is an end-to-end solution.

**Scenario**:

- **Fitness Apps**: Providing feedback on exercise form.
- **Gesture Recognition**: Control systems using body movements.
- **Web Applications**: Where computational resources are limited.

**Why This Combination?**:

- **Efficiency**: Predictor is optimized for speed and can run quickly.
- **Ease of Integration**: Being a single framework, easy integration.
- **Low Resource Consumption**: Suitable for devices with limited processing power.

**Example Result**:

![MediaPipe Result](../../../../../data/demos/videos/dummy_mediapipe3d.gif)

---

### 2. **CPN (2D) + MHFormer (3D)**

**Use Case**: High-accuracy motion analysis in controlled environments.

**Scenario**:

- **Sports Analytics**: Detailed analysis of athletes' movements for performance improvement.
- **Medical Diagnostics**: Precise movement tracking for rehabilitation monitoring.
- **Research Projects**: Studies requiring accurate human motion capture.

**Why This Combination?**:

- **High Accuracy**: CPN provides detailed 2D keypoints, which MHFormer lifts accurately into 3D space.
- **Temporal Consistency**: MHFormer utilizes video sequences, enhancing the reliability of predictions over time.
- **Robustness**: Effective in scenarios with complex poses and occlusions.

![CPN and MHFormer Result](../../../../../data/demos/videos/cpn_mhformer.gif)

---

### 3. **Lightweight Pose Estimation (2D) + VideoPose3D (3D)**

**Use Case**: Balanced performance for resource-limited systems with moderate accuracy requirements.

**Scenario**:

- **Fitness Applications**: Offline analysis of exercise routines where processing can occur after recording.
- **Educational Tools**: Activities involving movement tracking where fast feedback is critical.
- **Rehabilitation Monitoring**: Analyzing patient movements in clinical settings where computational resources are critical.

**Why This Combination?**:

- **Real-Time Capability**: Lightweight 2D estimation ensures quick processing.
- **Temporal Enhancement**: VideoPose3D leverages short video sequences for better 3D estimations without significant delay.
- **Resource Optimization**: Balances computational load while providing reasonable accuracy.

![Lightweight and VideoPose3D Result](../../../../../data/demos/videos/lightweight_videopose3d.gif)

---

### 4. **CPN (2D) + MixSTE (3D)**

**Use Case**: Cutting-edge applications demanding the highest accuracy.

**Scenario**:

- **Film and Animation**: High-fidelity motion capture for character animation.
- **Biomechanical Analysis**: Detailed study of human movement mechanics.
- **Advanced Robotics**: Precise human-robot interaction requiring exact human pose estimation.

**Why This Combination?**:

- **Superior Accuracy**: Both predictors are among the most accurate in their categories.
- **Advanced Modeling**: MixSTE's transformer architecture captures intricate spatio-temporal dependencies.
- **Professional Applications**: Ideal where accuracy cannot be compromised, and resources are available.


![CPN and MixSTE Result](../../../../../data/demos/videos/cpn_mixste.gif)

---

### 5. **CPN (2D) + ExPose (3D)**

**Use Case**: High-detail modeling where both accuracy and expressiveness are required.

**Scenario**:

- **Digital Avatar Creation**: Generating realistic avatars for games or virtual meetings.
- **Human-Computer Interaction Research**: Studying nuanced human expressions and gestures.
- **Ergonomic Studies**: Detailed posture and movement analysis in workplace settings.

**Why This Combination?**:

- **Detailed Input**: CPN provides accurate 2D keypoints while ExPose provides detailed modeling.
- **Comprehensive Output**: Captures the full range of human expression, including micro-movements. Also generates mesh pose prediction.
- **Professional Quality**: Suitable for industries where detail and accuracy are paramount.

![CPN and Expose Result](../../../../../data/demos/videos/cpn_expose.gif)
![Expose Mesh](../../../../../data/demos/videos/vizexpose.gif)

## Factors to Consider When Combining Predictors

When choosing combinations, consider the following:

1. **Accuracy vs. Speed**:

   - **Prioritize Accuracy**: Use high-accuracy models like CPN and MixSTE when precision is crucial.
   - **Prioritize Speed**: Opt for MediaPipe or Lightweight models for applications requiring fast processing.

2. **Computational Resources**:

   - **Limited Resources**: Combine lightweight models to ensure smooth performance on less powerful devices.
   - **Abundant Resources**: Use resource-intensive models for maximum accuracy when hardware allows.

3. **Application Requirements**:

   - **Expressiveness Needed**: Incorporate ExPose for detailed facial and hand movements.
   - **Motion Dynamics**: Use temporal models like MHFormer or MotionBert for applications focusing on movement over time.

4. **Ease of Integration**:

   - **Included Models**: For quick setup, use combinations of models included by default.
   - **Custom Installations**: If willing to invest time in setup, models requiring installation offer advanced features.

5. **Input Data Type**:

   - **Single Images**: Ensure both predictors can handle image-based input.
   - **Video Sequences**: Use predictors that leverage temporal information for enhanced performance.


## Additional Tips for Selecting Combinations

- **Testing**: Before finalizing, test different combinations to see which best meets your application's needs.
- **Scalability**: Consider how the combination will perform as the application scales or as user numbers increase.
- **Licensing**: Ensure that the models' licenses are compatible with your project's requirements.


## Installation Instructions

For the predictors that are not included by default, please follow the respective installation guides:

- **MotionBert**: [Installation Instructions](src/main/python/human_pose_estimation/models/predictors_3d/motionbert/MotionBert_installation.md)
- **VideoPose3D**: [Installation Instructions](src/main/python/human_pose_estimation/models/predictors_3d/videopose/VideoPose3D_installation.md)
- **MixSTE**: [Installation Instructions](src/main/python/human_pose_estimation/models/predictors_3d/mixste/MixSTE_installation.md)
- **ExPose**: [Installation Instructions](src/main/python/human_pose_estimation/models/predictors_3d/expose_lib/ExPose_installation.md)

Please ensure all dependencies are met and follow the guides carefully to integrate these models into your pipeline successfully.