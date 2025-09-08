# Real-Time Pose Estimation and Feedback for Bharatanatyam Dance Poses

This project uses computer vision and AI-based similarity scoring to evaluate Bharatanatyam dance postures in real-time. By normalizing human body landmarks, the system provides pose-invariant comparisons and dynamic feedback to learners and practioners. Switch on your webcam and strike a pose!

Beyond its immediate application in cultural preservation and dance pedagogy, this work demonstrates ideas in pose representation, similarity learning, and human-computer interaction.  

## Tech Stack  
- Python 3.9+  
- [MediaPipe](https://developers.google.com/mediapipe) for pose estimation  
- OpenCV for real-time video handling  
- NumPy for math & normalization  
- (Optional) scikit-learn / TensorFlow → for future ML model training  

### How to Use
1. Have your instructor strike the perfect Bharatanatyam pose you'd like to immitate.
2. If you're just practising online, just upload your reference poses :3
3. Switch on your webcam and start practising!
   
## Steps to Get Started
#### 1. Clone the repo  
```
git clone https://github.com/your-username/ai-natyam-coach.git
cd ai-natyam-coach
```
#### 2. Install dependencies
```
conda create -n natyam python=3.9
conda activate natyam
pip install opencv-python mediapipe numpy
```
#### 3. Run the Pose Detection Engine to save a Reference Pose
``` python pose_detector.py ```

#### 4. Strike a pose and press S to save it

#### 5. Run the Feedback System
``` python pose_detector.py ```


The system tracks your body landmarks, compares them to reference poses, and gives you a match score + real-time feedback on how to achieve the perfectly balanced graceful Bharatanatyam pose.
The Dynamic feedback generation engine identifies specific body parts that deviate from the reference pose.  
Lightweight implementation tested on Macbook M2 Pro

## Contributions
This is an ongoing research-oriented project. Contributions and collaborations are welcome.
