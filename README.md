![image alt](https://github.com/aabbas77-web/AliSoft/blob/main/AliSoft128Transparent.png)
# [AliSoft](https://hodhods.com) YOLO Pose Video Control
# By Dr. Ali Abbas aabbas7@gmail.com
# Programming Language: Python

# 🎥 YOLO Pose Video Control

A real-time computer vision app that uses **human body motion** to **control video playback** using **Ultralytics YOLOv8 Pose Estimation**.

![image alt](https://github.com/aabbas77-web/yolo_pose_video_control/blob/main/images/sport.jpg)<br/>

![image alt](https://github.com/aabbas77-web/yolo_pose_video_control/blob/main/images/motorcycle.jpg)<br/>

![image alt](https://github.com/aabbas77-web/yolo_pose_video_control/blob/main/images/walk.jpg)<br/>

---

🎯 **Core idea:**
Instead of using a keyboard or controller, your *motion* drives the video.
- If you stop moving, the playback pauses.
- If you move faster, the video speeds up — creating a responsive, immersive experience.

---

## 🧠 Features

- 🧍  Detects and tracks body pose in real time using **YOLOv8**.
- 🎮 Controls video playback speed or pauses it based on detected **body movement**.
- 🚲 Optional animated **bike overlay** for fun visualization.
- 📊 Displays **speed and distance** estimated in meters.
- 🧭 Built-in **Help** and **About** overlays with smooth fade effects.
- 🧩 On-screen **menu system** (toggle via mouse or keyboard shortcuts).

---

📁 **Included Files:**
- `yolo_pose_video_control.py` — main algorithm
- `requirements.txt` — dependencies
- `README.md` — setup, usage, and video source
- `run_app.bat` — one-click launcher

🎥 **Video Source:**
All running videos are courtesy of [Virtual Running Video](https://www.youtube.com/@virtualrunningvideo)

🔧 **Tech Stack:**
`Python`, `OpenCV`, `Ultralytics YOLOv8`, `NumPy`

🖼️ **Visuals:**
1️⃣ YOLO Pose overlay with live keypoints and motion tracking  
2️⃣ Real-time motion info (Speed/Distance)  
3️⃣ Feature summary banner

---

## 🧰 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/aabbas77-web/yolo-pose-video-control.git
cd yolo-pose-video-control
```

### 2️⃣ Install dependencies
Make sure Python 3.8+ is installed, then run:
```bash
pip install -r requirements.txt
```

### 3️⃣ Download a YOLO pose model
The default model is `yolov8n-pose.pt`, which is downloaded automatically when you run the app.

---

## ▶️ Usage

### Run the app
```bash
python yolo_pose_video_control.py
```

### Controls
| Key | Action |
|-----|--------|
| **m** | Toggle on-screen menu |
| **p** | Toggle pose mini-view (bottom-right) |
| **s** | Toggle skeleton overlay |
| **d** | Enable/disable detection |
| **h** | Open Help overlay |
| **a** | Open About overlay |
| **ESC / Quit** | Exit the app |

### Mouse
- Click **menu items** to toggle features.
- Click outside or on **×** to close Help/About overlays.

---

## ⚙️ How It Works

1. YOLOv8 detects human pose keypoints in real time.  
2. The app computes **average pixel displacement** per frame.  
3. This motion is converted into **speed (m/s)** and **distance (m)** using camera FOV geometry.  
4. The app **pauses** the video when the subject is still, and **plays faster** when movement increases.  
5. Visual overlays (pose view, motion info, bike animation) are drawn dynamically.

---

## 📸 Example Use Cases

- Sports motion analysis 🏃‍♀️  
- Fitness tracking & gesture control 🏋️‍♂️  
- Fun AI-powered video demos 🎬  

---

## 👨‍💻 Author

**Ali Abbas**  
PhD-qualified Software Engineer — Computer Vision, GIS, and AI  
- 📧 Email: [aabbas7@gmail.com](mailto:aabbas7@gmail.com)  
- 🌐 [GitHub](https://github.com/aabbas77-web)  
- 💼 [LinkedIn](https://www.linkedin.com/in/ali-abbas-45799710b)  

---

## 📜 License
This project is open source under the [MIT License](LICENSE).

💬 I’d love to connect with fellow AI engineers and computer vision researchers interested in **motion-based interaction systems**, **real-time tracking**, or **human-centered AI applications**.

#AI #ComputerVision #YOLOv8 #DeepLearning #HumanPoseEstimation #OpenCV #Ultralytics #MachineLearning #Python #Innovation
