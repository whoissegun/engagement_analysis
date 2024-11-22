<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO --> 
<br /> 
<div align="center"> 
  <a href="https://github.com/whoissegun/engagement_analysis"> 
    <img src="assets/U.GIF" alt="Logo" width="400"> 
  </a> 
  <h3 align="center">Lokdin: Engagement Detection Tool Using Computer Vision</h3> 
  <p align="center">
    Lokdin is a real-time engagement detection tool that can be utilized for various levels of analysis across multiple domains, including education, driving, healthcare, and corporate productivity. It leverages computer vision techniques to classify human engagement levels, providing actionable insights to enhance performance, safety, and efficiency.
    <br />
    <strong>Lokdin won 3rd place overall at the prestigious <a href="https://cuhacking.com">CUHacking Hackathon</a>!</strong>
  </p>
    <!-- <a href="https://github.com/your_username/lokdin">
      <strong>Explore the docs »</strong>
    </a> -->
    <br /> 
    <a href="assets/DEMO.mov">View Demo</a> · <a href="https://github.com/whoissegun/engagement_analysis/issues">Report Bug</a> · 
    <a href="https://github.com/whoissegun/engagement_analysis/issues">Request Feature</a> 
  </p> 
</div> 

<!-- TABLE OF CONTENTS --> 
<details> 
  <summary>Table of Contents</summary> 
    <ol> 
      <li>
        <a href="#about-lokdin">About Lokdin</a>
        <ul> 
          <li><a href="#built-with">Built With</a></li> 
        </ul> 
      </li> 
      <li>
        <a href="#target-use-cases">Target Use Cases</a>
      </li>
      <li>
        <a href="#technical-development-plan">Technical Development Plan
        </a>
      </li> 
      <li>
        <a href="#mediapipe-important-landmarks">MediaPipe Important Landmarks
        </a>
      </li>
       <li><a href="#key-algorithms">Key Algorithms</a></li> 
      <li>
        <a href="#challenges-and-mitigation">Challenges and Mitigation</a>
      </li> 
      <li> 
        <a href="#getting-started">Getting Started</a> 
        <ul> 
          <li><a href="#prerequisites">Prerequisites</a></li> 
          <li><a href="#installation">Installation</a></li> 
        </ul> 
      </li> 
      <li>
        <a href="#usage">Usage</a>
        <ul> 
          <li><a href="#live-demo">Live Demo</a></li> 
          <li><a href="#visualization-and-heatmap">Visualization and Heatmap</a></li> 
        </ul> 
      </li> 
      <li>
        <a href="#contributing">Contributing</a>
      </li> 
      <li>
        <a href="#license">License</a>
      </li> 
      <li>
        <a href="#acknowledgments">Acknowledgments</a>
      </li> 
  </ol> 
</details>


<!-- ABOUT THE PROJECT -->
## About Lokdin

Lokdin is a real-time engagement detection tool designed to provide actionable insights across various industries. By leveraging computer vision techniques, it helps in enhancing productivity, safety, and efficiency.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)

* OpenCV
* MediaPipe
* NumPy
* Keras

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- TARGET USE CASES -->
## Target Use Cases

Lokdin offers solutions tailored to the needs of various industries:

### Education (E-Learning Platforms)

- **Problem**: Educators struggle to monitor student engagement in virtual classrooms.
- **Solution**: Real-time monitoring of student attention, enabling teachers to adjust their teaching strategies.
- **Output**: Alerts for disengagement, analytics on participation trends.

### Corporate Training & Meetings

- **Problem**: Inefficiency in assessing employee attention during remote meetings or training.
- **Solution**: Integration into platforms like Microsoft Teams or Zoom for engagement tracking during sessions.
- **Output**: Post-meeting engagement reports and real-time feedback.

### Driver Monitoring Systems

- **Problem**: Distracted or drowsy driving leads to accidents.
- **Solution**: In-car systems detect fatigue and distraction, triggering alerts for safety.
- **Output**: Alerts for "Drowsy," "Distracted," or "Alert and Engaged."

### Healthcare and Therapy

- **Problem**: Difficulty in assessing patient engagement during telehealth or therapy sessions.
- **Solution**: Real-time feedback for healthcare providers on patient attention and emotional distress.
- **Output**: Metrics like "Engaged," "Distressed," or "Passive Listening."

### Entertainment and Media

- **Problem**: Need for feedback on audience engagement during content creation or testing.
- **Solution**: Engagement analytics for audience reactions during ads, movies, or games.
- **Output**: Insights into which segments capture or lose attention.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- TECHNICAL DEVELOPMENT PLAN -->
## Technical Development Plan

Lokdin employs a combination of pre-trained models, feature engineering, and real-world data to deliver accurate and actionable engagement insights.

### Model Development

Lokdin's model development employs a custom-built deep learning architecture trained on synthetic datasets tailored to the project's engagement detection requirements.

- **Base Model Creation**: 
  The model is designed from scratch, optimized specifically for engagement classification. By utilizing synthetic datasets, it overcomes the challenges of limited public datasets for niche applications like engagement analysis. These datasets simulate diverse scenarios, including virtual classrooms, driving, and telehealth sessions, ensuring robustness across use cases.

- **Feature Extraction**: 
  The custom model extracts and analyzes:
  - **Core Features**: Facial expressions, gaze tracking, head movements, and posture.
  - **Use Case-Specific Features**:
    - **Drivers**: Eye gaze, yawning detection using the Mouth Aspect Ratio (MAR), drooping eyelids.
    - **Students**: Interaction with screen/material, posture dynamics.
    - **Healthcare**: Emotional cues, differentiating between passive and active engagement.

- **Output Classes**:
  - **Binary Classification**: The model determines if a subject is "Engaged" or "Not Engaged."
  - **Multi-Class Outputs**: Provides detailed engagement levels like "Highly Engaged," "Moderately Engaged," and context-aware classes such as "Drowsy" for drivers or "Confused" for students.

### Data Structure Sent to the Model

The extracted engagement metrics are encapsulated within the `FaceFeatures` data class, which is passed to the model for analysis:

```python
@dataclass
class FaceFeatures:
    head_pitch: float  # Vertical head movement
    head_yaw: float  # Horizontal head movement
    head_roll: float  # Rotational head movement
    gaze_x: float  # Horizontal gaze position
    gaze_y: float  # Vertical gaze position
    eye_contact_duration: float  # Time spent maintaining eye contact
    gaze_variation_x: float  # Gaze variability in horizontal direction
    gaze_variation_y: float  # Gaze variability in vertical direction
    face_confidence: float  # Confidence score of face detection
    landmarks_stability: float  # Stability of face landmarks
    time_since_head_movement: float  # Time since last significant head movement
    time_since_gaze_shift: float  # Time since last significant gaze shift
    mar: float  # Mouth Aspect Ratio, used for yawning detection
    blink_ratio: float  # Blink ratio for detecting blinks
    is_blinking: bool  # Boolean flag for blink detection
    is_focused: bool  # Boolean flag for determining focus
    distraction_duration: float  # Duration of distraction
    eye_contact_detected: bool  # Boolean flag for detecting eye contact
    yawn_detected: bool  # Boolean flag for detecting yawns
```

### Data Pipeline

- **Data Capture**: Uses OpenCV and MediaPipe for frame analysis.
- **Storage**: Granular storage options with anonymization protocols.

### Training and Optimization

- Incorporates synthetic and real-world datasets.
- Adapts a multitask learning approach for cross-domain generalization.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- IMPORTANT LANDMARKS -->
## Mediapipe Important Landmarks

Lokdin leverages MediaPipe's **Facial Landmark Detection** model to analyze engagement-related features in real-time. The model detects 468 unique landmarks on the face, which are used for calculating various metrics such as head pose, gaze, and mouth aspect ratio.

Below are the key landmarks utilized in Lokdin:

### Key Facial Landmarks
- **Nose Tip**: Landmark `1` – Used for head pose estimation.
- **Chin**: Landmark `152` – Determines vertical head position for pose estimation.
- **Left Eye Outer Corner**: Landmark `33` – Aids in gaze tracking and blink detection.
- **Right Eye Outer Corner**: Landmark `263` – Complements gaze and blink calculations.
- **Left Mouth Corner**: Landmark `61` – Used for yawning detection via mouth aspect ratio.
- **Right Mouth Corner**: Landmark `291` – Completes mouth aspect ratio calculations.
- **Left Iris Center**: Landmark `468` – Central point for gaze tracking.
- **Right Iris Center**: Landmark `473` – Central point for gaze tracking.

### Eye-Specific Landmarks for Blink Detection
- **Left Eye Points**: `[33, 160, 158, 133, 153, 144]`
- **Right Eye Points**: `[362, 385, 387, 263, 373, 380]`

### Mouth-Specific Landmarks for Yawn Detection
- **Top Lip Points**: Landmarks `13` and `14`
- **Bottom Lip Points**: Landmarks `17` and `18`

### Visualization
To better understand the positioning of these landmarks, refer to the following annotated image showing all 468 landmarks from MediaPipe's facial detection model:

![Facial Landmarks Image](assets/landmarks.jpg)

This visualization helps illustrate how these landmarks are mapped onto a user's face for real-time engagement analysis. 

<br>

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CHALLENGES AND MITIGATION -->
## Challenges and Mitigation

### Challenges

1. **Data Privacy**: Ensuring compliance with GDPR and similar regulations.
2. **Data Scarcity**: Limited datasets for engagement classification.
3. **Model Bias**: Avoiding demographic bias in predictions.

### Mitigation Strategies

- Implement local processing for privacy-critical applications.
- Regular audits for model fairness.
- Diverse data collection strategies to ensure inclusivity.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites
Ensure you have the following installed:

- Python 3.7 or higher
- OpenCV
- MediaPipe
- NumPy
- TensorFlow
- Keras

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/whoissegun/engagement_analysis.git
   ```
2. Navigate to Directoy
   ```sh
   cd engagement_analysis
   ```
3. Download all dependencies using pip or pip3 (MAC)
   ```sh
   pip3 install -r requirements.txt
   ```
4. Navigate to backend/processing
  ```sh
   cd backend/processing/
   ```
5. 1 For the visualization including the faceMesh overlay
  ```sh
   python3 main_processor.py
   ```
5. 2 For the heatMap
  ```sh
   python3 heatmap.py
   ```
6. click 'q' key to close the window and terminate the application

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

### Live Demo
Click the image below for a live demo of Lokdin in action:
<a href="https://github.com/whoissegun/engagement_analysis"> 
    <img src="assets/DEMO.GIF" alt="Live Demo" width="10000"> 
</a>

### Visualization and Heatmap
Lokdin includes a heatmap feature to provide a comprehensive visualization of engagement metrics over time. Click the image below to view the heatmap in action:
<a href="https://github.com/whoissegun/engagement_analysis"> 
    <img src="assets/Heatmap DEMO.GIF" alt="Heatmap Visualization" width="400"> 
</a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Special thanks to the following for their contributions and support:

* **CUHacking Hackathon** - For providing the platform where Lokdin was conceptualized and developed. [CUHacking](https://cuhacking.ca)
* [OpenCV](https://opencv.org/) - For enabling efficient computer vision processing.
* [Mediapipe](https://mediapipe.dev/) - For providing the facial landmark detection model.
* **Team Lokdin** - For the collaborative effort that made this project a success.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/whoissegun/engagement_analysis/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
