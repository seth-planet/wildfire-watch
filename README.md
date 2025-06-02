# 🔥 Wildfire Watch

**Wildfire Watch** is a personal, open-source, edge-deployed wildfire detection and response platform leveraging Frigate NVR, Hailo AI acceleration, MQTT messaging, and GPIO-based pump control.

---

## 🚀 Features
- Real-time wildfire and smoke detection from RTSP camera streams
- Automatic consensus-based triggering of fire suppression pumps
- Edge-ready deployment via Balena and Docker Compose
- Future-ready for extended object detection (people, cars, license plates)

---

## 📋 Getting Started

### Requirements
- Raspberry Pi 5
- Hailo TPU (AI accelerator)
- Compatible RTSP security cameras
- MQTT broker (included)
- GPIO-controlled relay (pump and valve control)

### Quickstart

```bash
git clone https://github.com/seth-planet/wildfire-watch.git
cd wildfire-watch
docker-compose up --build
````

### Balena Deployment

* Create a new Balena app and set device-specific environment variables as needed.
* Push the project to your Balena fleet:

```bash
balena push wildfire-watch
```

---

## 📡 MQTT Topics

* `fire/detection`: Wildfire detections from cameras
* `fire/cam_info`: Camera heartbeat and metadata
* `fire/confirmed`: Consensus alerts triggering fire pump

---

## 🛠 Roadmap

*

---

## 📜 License

MIT License © 2025 Seth Price

## Disclaimer

Wildfire Watch is provided under the MIT License for educational, experimental, and non-commercial use.  
It is intended to aid in early detection and response to wildfires but does **not** guarantee safety or loss prevention.  
**Use at your own risk.**

The authors and contributors are **not responsible for any damage, injury, or loss of property** resulting from the use, malfunction, or failure of this system.

Always consult fire safety professionals and local regulations before relying on any automated fire response systems.
