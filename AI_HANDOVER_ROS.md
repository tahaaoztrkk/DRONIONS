# AI HANDOVER DOCUMENT: DRONIONS ROS/GAZEBO INTEGRATION
> **To the next AI Assistant:** Please read this entire document carefully before taking any action. It contains the current architecture of the DRONIONS project and the user's immediate goal.

## 1. Project Context
**DRONIONS** is an autonomous drone tracking system built with a **Hybrid Architecture**:
- **Phase 1 (VLM Searching):** Uses Gemini 1.5 Flash (`google-genai` SDK) to search for a target via natural language prompts. It also has a **Visual Memory Bank** (checks the `memory/` folder for reference images and performs few-shot matching if found).
- **Phase 2 (YOLO Tracking):** Once Gemini confirms the target is visible, the system switches to `ultralytics` YOLO-World (Zero-Shot) and `supervision` ByteTrack to track the object at 30 FPS.
- **Voice Assistant:** Uses `SpeechRecognition` + `PyAudio` (background STT) and `pyttsx3` (background TTS).

## 2. Current Status
The Python pipeline (`main.py`) runs perfectly on Windows using a standard IP camera (`cv2.VideoCapture`). The user is now migrating this repository to a **Linux** environment to test it in a **ROS & Gazebo** simulation.

## 3. Your Immediate Goal
Your task is to integrate this Python pipeline with ROS (Robot Operating System) and Gazebo. You should create a ROS Wrapper (e.g., `dronions_ros_node.py`) or modify `main.py` without breaking the core Hybrid Architecture.

### Key Integration Points Needed:
1. **Camera Input (Subscribe):**
   - Replace `cv2.VideoCapture(CAMERA_SOURCE)` with a ROS Subscriber listening to the Gazebo drone camera topic (e.g., `/drone/camera/image_raw`).
   - Use `cv_bridge` to convert `sensor_msgs/Image` to an OpenCV `np.ndarray` (BGR).
2. **Flight Commands (Publish):**
   - Currently, `navigation/navigator.py` produces string decisions (e.g., `"TURN_RIGHT"`, `"MOVE_FORWARD"`).
   - You need to create a ROS Publisher that translates these strings into `geometry_msgs/Twist` messages and publishes them to `/cmd_vel` (or the specific drone topic like `mavros_msgs`).
3. **Environment:**
   - Ensure you ask the user which ROS version (ROS 1 Noetic vs ROS 2 Humble) they are running before writing the Node.

## 4. Instructions for You (The AI)
- Do **not** remove the Gemini or YOLO logic.
- Do **not** remove the Voice Assistant threads (`assistant/speech.py`, `assistant/listen.py`), they should run concurrently with the ROS spin loop.
- Focus strictly on adapting the inputs/outputs to ROS standard messages.
- Wait for the user to specify their ROS version and drone package, then propose an implementation plan.
