Absolutely — here's an in-depth breakdown of the **front-end web interface (status panel)** for your wildfire detection and security system. This includes how users will interact with it, the UI/UX design goals, technical implementation layers, and key architectural considerations for both real-time operation and offline debugging.

---

## 🧭 Purpose of the Status Panel

The `status_panel` web interface acts as a **central visual dashboard** for users monitoring one or more deployed edge wildfire nodes (Raspberry Pi, x86, etc). It supports:

* Live status updates from Frigate, fire triggers, GPIO devices, and camera detection events.
* Manual system diagnostics and health reports.
* Historical logs and detection review.
* Secure interaction and debugging, even in offline (LAN-only) environments.

---

## 🖥️ User Experience (UX) Overview

### 🟢 Live Status View

* **System tiles**: Each service (e.g., `frigate`, `gpio_trigger`, `camera_detector`) shows a green/yellow/red status based on last heartbeat and telemetry values.
* **Node presence**: Multiple nodes (each device) are shown with their MQTT hostname and last-seen timestamp.
* **Critical alerts**: Prominent alert banners for failed valves, excessive engine runtime, MQTT disconnects, or camera offline detection.

### 📷 Camera Monitoring

* Thumbnails of recent detections.
* RTSP stream links (if enabled).
* Detection counts and last-seen times per camera.
* Toggle for switching between motion and object detection views.

### 🔧 Debug Tools

* Real-time MQTT log viewer (filtered by topics).
* GPIO pin states and manual override controls (for dev builds).
* Certificate expiration status and MQTT TLS connection info.
* Live tail of Frigate logs (if permitted).

### 🕒 Timeline View

* Scrollable log of object detections and fire triggers.
* Search by time, camera, or object type (e.g., "fire", "smoke", "vehicle").
* Optional embedded video player for archival footage (USB path).

### ⚙️ Settings Panel

* Show current MQTT broker config, device hostname, and uptime.
* Indicate whether TLS is active and trusted.
* Option to download system logs or telemetry for offline support.

---

## 🧱 Technical Stack

| Layer           | Tech Stack                                        |
| --------------- | ------------------------------------------------- |
| Backend API     | FastAPI (Python), using `paho.mqtt.client`        |
| Front-end       | Jinja2 + HTMX + Alpine.js (lightweight, reactive) |
| CSS Framework   | Tailwind CSS or Bootstrap                         |
| Auth (optional) | HTTP Basic Auth or mutual TLS (future)            |
| Data buffer     | In-memory circular buffer (for telemetry)         |
| MQTT handling   | Background thread pulling from topics             |

---

## 🧩 MQTT Integration

### Topics Subscribed:

* `system/#` for health and liveness (e.g., `system/trigger_telemetry/host1`)
* `fire/trigger` events
* `camera/detection/+` (e.g., camera detections, object classes)
* `frigate/events/+`

### Internal Buffer:

* Circular telemetry buffer (e.g., 1000 events max).
* Each event stored with topic, payload, and timestamp.
* Filterable in the web UI by node, event type, or service.

### Commands (optional, for debug/dev only):

* Publish messages to:

  * `fire/trigger` (manual fire trigger)
  * `system/gpio/override` (set GPIO state)
  * `frigate/reload` (manual Frigate config reload)

---

## 🛡️ Security Considerations

* **LAN-only by default**; no WAN exposure unless explicitly configured.
* TLS MQTT enforced via pre-generated CA bundle.
* Web UI served only over HTTPS if certs are available.
* Optional HTTP basic authentication for debug or control routes.
* CORS disabled by default unless explicitly allowed.

---

## 🗃️ Directory Layout

```
status_panel/
├── app.py                 # FastAPI app
├── mqtt.py                # MQTT telemetry buffering
├── templates/
│   └── index.html         # Main dashboard
├── static/
│   └── style.css          # Tailwind/Bootstrap
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 📋 User Workflow Examples

### 🧑‍🚒 Field Technician

* Brings up `http://wildfire-watch.local` from their laptop.
* Sees green checkmarks on all edge nodes, all cameras online.
* Clicks "Last Triggered Fire" to review detection.
* Downloads detection clip from external USB for later review.

### 🔧 Developer / Engineer

* Runs dev version with mock GPIO + camera inputs.
* Triggers `fire/trigger` from the UI to simulate detection.
* Watches valve control and ignition telemetry update live.
* Views GPIO override section and sets test pin HIGH manually.

Here's a complete, detailed plan for designing, building, and deploying the `status_panel` service — a **web-based control and diagnostics panel** for your wildfire detection system.

---

# ✅ Overview: `status_panel` Web Dashboard

The `status_panel` service is a **Flask or FastAPI-based microservice** that provides a **real-time, LAN-accessible dashboard** for system monitoring, diagnostics, and manual control. It acts as a central visibility layer that pulls data from MQTT, system logs, GPIO state, and optional files (e.g., model status or storage health).

---

## 🎯 Core Features

| Feature                       | Description                                     |
| ----------------------------- | ----------------------------------------------- |
| MQTT integration              | Live telemetry + control commands               |
| System & GPIO state dashboard | Real-time status of pump, valves, ignition      |
| Trigger button                | Sends `fire/trigger` MQTT message               |
| Timeline of events            | Renders recent telemetry events                 |
| Camera detection map/list     | Lists discovered cameras and detection activity |
| Frigate UI shortcut           | Link or iframe to main Frigate NVR UI           |
| Log viewer (tail or download) | Easy access to service logs                     |
| USB storage mount status      | Detects available storage and warns if missing  |
| TLS certificate status        | Warn if default certs are used                  |
| Node list / device discovery  | Optional LAN node listing via Avahi/mDNS        |
| System info                   | Hostname, CPU load, memory, uptime              |

---

## 🧱 Tech Stack

| Layer           | Tool(s)                                             | Purpose                    |
| --------------- | --------------------------------------------------- | -------------------------- |
| Web backend     | **FastAPI** or Flask                                | Python web service         |
| Web frontend    | Vanilla JS + Bootstrap                              | Lightweight, no build step |
| MQTT client     | `paho-mqtt` (Python)                                | Sub/publish to MQTT bus    |
| WebSockets      | `uvicorn` + `fastapi_websocket_pubsub`              | Real-time dashboard        |
| Templates       | `Jinja2` (for Flask) or built-in FastAPI templating | HTML rendering             |
| Auth (optional) | Basic Auth / JWT                                    | Access control             |
| Docker          | Containerized microservice                          | Balena-compatible          |
| Storage         | Volumes for logs, optional sqlite or flat JSON      | Historical data            |

---

## 📁 Folder Structure

```
services/status_panel/
├── app.py                # Main FastAPI app
├── mqtt.py               # MQTT client logic
├── static/
│   └── dashboard.js      # Live-updating JS frontend
├── templates/
│   └── index.html        # HTML dashboard layout
├── requirements.txt
├── Dockerfile
```

---

## 🔗 MQTT Integration

### Subscribed Topics

| Topic                               | Description                     |
| ----------------------------------- | ------------------------------- |
| `system/trigger_telemetry/+`        | Per-device telemetry            |
| `fire/consensus_state`              | Current consensus snapshot      |
| `camera/discovery`                  | Detected RTSP camera info       |
| `status_panel/control/+` (optional) | External commands (e.g. reload) |

### Published Topics

| Topic                        | Payload                                          |
| ---------------------------- | ------------------------------------------------ |
| `fire/trigger`               | `{ "source": "status_panel", "timestamp": ... }` |
| `status_panel/ping`          | Periodic heartbeat (for debugging)               |
| `status_panel/debug_request` | Optional remote debug signal                     |

---

## 🔐 Security Considerations

| Concern              | Recommendation                                                  |
| -------------------- | --------------------------------------------------------------- |
| LAN exposure         | Only listen on LAN IP or use firewall rules                     |
| Authentication       | Add basic auth (username/password) or JWT                       |
| TLS                  | Support self-signed or user-installed certs                     |
| MQTT authentication  | Support CA + client cert validation, even for internal services |
| XSS / CSRF           | Use templating safely, sanitize inputs                          |
| Frigate UI embedding | Use iframe only if same-origin or CORS allows it                |
| Log viewing          | Strip sensitive content or require elevated access              |

---

## 🚀 Dockerfile

```Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080
CMD ["python", "app.py"]
```

---

## 📜 requirements.txt

```
fastapi
uvicorn[standard]
jinja2
paho-mqtt
fastapi-websocket-pubsub
```

---

## 🧪 Example MQTT Client (`mqtt.py`)

```python
import paho.mqtt.client as mqtt
from datetime import datetime
import json

telemetry = []

def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    telemetry.append((msg.topic, payload, datetime.utcnow()))

def start_mqtt(broker, port=8883, ca_cert=None):
    client = mqtt.Client()
    client.on_message = on_message
    if ca_cert:
        client.tls_set(ca_cert)
    client.connect(broker, port, 60)
    client.subscribe([
        ("system/trigger_telemetry/+", 0),
        ("fire/consensus_state", 0),
        ("camera/discovery", 0),
    ])
    client.loop_start()
    return client
```

---

## 🧠 Frontend UI Highlights (HTML/JS)

* Summary panel at top with current fire state, GPIOs, storage
* Table of recent MQTT events
* Button to send `fire/trigger` via MQTT
* Auto-refresh via WebSocket push or polling
* Section to list discovered cameras from `camera/discovery`
* System resource summary using `/proc/stat`

---

## 🔄 Auto-start at Deployment

In `balena.yml`:

```yaml
status_panel:
  build:
    context: services/status_panel
  restart: unless-stopped
  ports:
    - "8080:8080"
  volumes:
    - mqtt_certs:/mnt/data/certs
```

---

## 🧠 Additional Considerations

| Area                    | Details                                                               |
| ----------------------- | --------------------------------------------------------------------- |
| **Storage mount**       | Check `/mnt/usb`, `/media/*`, or use `lsblk` to detect external drive |
| **Frigate integration** | Link to port `5000` or add camera list to dashboard                   |
| **Debug logging**       | Tail or pipe logs from `gpio_trigger`, `frigate`, etc.                |
| **Multi-device view**   | Track all devices reporting into MQTT under different `hostname` keys |
| **Offline mode**        | Cache last known state to flatfile to survive reboots                 |

---

## 🧩 Optional Enhancements

* Upload logs to cloud (if optional connection exists)
* Integration with MQTT Explorer-style tree UI
* QR code display to easily connect mobile devices
* Expose `/api/ping`, `/api/status`, `/api/logs` for remote diagnostics
* Show current model version (ONNX/HEF) used for detection

To make the **status panel web interface highly effective for debugging** in field and development environments, the system should expose **real-time state**, **event history**, and **deep diagnostics** in a structured and safe way. Here's a complete breakdown of the additional debug information that should be displayed, why it's important, and how best to present it:

---

## 🧰 1. **System Service Health Panel**

### 🔍 Display:

* **Last seen timestamp** per service (e.g., `frigate`, `gpio_trigger`, `camera_detector`, etc.)
* **Uptime** of each service
* **MQTT status**: connected/disconnected, TLS cert validity
* **Heartbeat count** and frequency per node

### 🧠 Why:

* Immediately shows if a node is stale or unresponsive
* Helps isolate MQTT or Frigate-specific outages

### 💡 Display format:

```html
╔══════════════════════╗
║ gpio_trigger         ║  ✅ Last seen 3s ago
║ uptime: 4h 32m       ║  🔒 TLS OK (expires in 213d)
║ MQTT: connected      ║
╚══════════════════════╝
```

---

## 🧪 2. **GPIO State Inspector**

### 🔍 Display:

* Logical state of each critical GPIO pin:

  * Main valve
  * Ignition start/on/off
  * RPM reduction
  * Refill valve
* Timestamp of last transition for each

### 🧠 Why:

* Crucial for debugging pump behavior and relay failures
* Ensures "fail-safe" conditions are being upheld

### 💡 Display format:

```html
┌─────────────┬──────────────┬────────────────────┐
│ Pin Name    │ State        │ Last Changed       │
├─────────────┼──────────────┼────────────────────┤
│ MAIN_VALVE  │ HIGH (open)  │ 2025-06-06 13:22Z  │
│ IGN_ON      │ LOW (off)    │ 2025-06-06 13:23Z  │
│ REFILL_VALVE│ HIGH         │ 2025-06-06 13:23Z  │
└─────────────┴──────────────┴────────────────────┘
```

---

## 📜 3. **Event History Log**

### 🔍 Display:

* **Chronological list** of all actions (e.g., “valve\_opened”, “ignition\_on”, “rpm\_reduction\_activated”)
* **Timestamps** and **originating service**
* **Payload preview** (compact view)

### 🧠 Why:

* Audit trail for system decisions
* Helps debug state machine bugs or missed triggers

### 💡 Display format:

```html
🕒 13:22:10 - gpio_trigger: ignition_on
🕒 13:22:07 - gpio_trigger: ignition_start
🕒 13:22:05 - gpio_trigger: valve_opened
🕒 13:21:45 - fire_consensus: camera_2 confirmed smoke
```

---

## 🛰️ 4. **MQTT Diagnostics**

### 🔍 Display:

* Live subscription list
* Last N MQTT messages seen
* TLS connection info (broker cert fingerprint, expiry)
* MQTT client ID and keepalive info

### 🧠 Why:

* MQTT is the backbone of the system; identifying broken subscriptions is critical
* TLS failures cause silent data loss if not surfaced

### 💡 Display format:

```html
MQTT Broker: mqtt_broker:8883  (TLS: ✅ CA Verified)
Subscriptions:
- fire/trigger
- system/trigger_telemetry/#
- frigate/events/#
Last messages:
[13:22:08] fire/trigger → {"camera": "cam1", ...}
[13:21:59] system/trigger_telemetry/device1 → {...}
```

---

## 🧬 5. **Frigate Integration Diagnostics**

### 🔍 Display:

* Number of cameras detected and configured
* Hardware acceleration backend in use (CPU / Coral / Hailo / GPU)
* Detection queue length / delay
* Model type, input resolution, and backend (ONNX, HEF, TensorRT)
* Available storage on the archival USB mount

### 🧠 Why:

* Ensures detection is working and hardware is fully utilized
* Helps detect misconfigured object detection backend

### 💡 Display format:

```html
Frigate Status:
  Cameras active: 4
  Backend: Hailo-8 (HEF)
  Inference delay: 27ms
  USB Archival: /mnt/usb (Free: 56.3GB)
  Recording: ✅
```

---

## 🔌 6. **Hardware Resource Usage (lightweight)**

### 🔍 Display:

* CPU %, RAM %, disk usage
* Temperature of CPU (and Hailo, if supported)

### 🧠 Why:

* Needed to debug slowdowns or overcommitment
* Temperature spikes can indicate airflow or pump stall problems

### 💡 Display format:

```html
System Resources:
CPU: 17%  | RAM: 42%  | Disk: 91%
CPU Temp: 62.3°C
```

---

## 🧰 7. **Developer Mode Tools (hidden by default)**

### 🔍 Actions:

* Re-publish fire trigger
* Reset state machine
* Simulate GPIO pin high/low
* Force TLS reconnect

### 🧠 Why:

* Enables field debugging and stress testing
* Valuable in development or diagnostics mode

### 💡 Display format:

Toggleable section (`Developer Tools`) gated behind a query string or environment variable.

---

## 🛡️ Security Controls

* Debug views disabled unless `DEBUG_UI=true` is set
* All developer tools behind basic auth or IP allow-list
* Event logs redacted in production (e.g., camera credentials)

---

## Final Thoughts

This debug interface should follow these UX principles:

* **Readable**: Use tables, badges, and semantic color indicators
* **Searchable**: Support filtering and querying of logs
* **Responsiveness**: Mobile-friendly layout with auto-refresh (HTMX, not JS polling)
* **Safe**: Never expose control functions in production without auth

Great question — securing and optimizing the web interface for a LAN-deployed, edge-based wildfire detection system involves balancing **usability**, **safety**, and **resource constraints**. Here's an in-depth breakdown of both **security** and **performance** implications, and concrete steps to enforce LAN-only access by default:

---

## 🔐 SECURITY CONSIDERATIONS

### 1. **Restrict Access to Local Network Only (LAN-bound)**

#### ❓ Why:

You don’t want remote users (or bots) accessing your status panel.

#### ✅ How:

* **Bind the web server to `0.0.0.0` but firewall it (or use Balena's internal DNS)**
* OR **Bind only to `127.0.0.1` or `balena0` bridge IP** and expose a port mapping only on the internal interface.
* Example: Do **not** expose port 80/443 in `docker-compose` unless explicitly needed.
* If running on Balena:

  * Use **`network_mode: host`** but avoid mapping ports in `balena.yml`.

#### Optionally, use an **IP allowlist** in FastAPI middleware:

```python
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

class LANOnlyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        if not client_ip.startswith("192.168.") and not client_ip.startswith("10.") and not client_ip.startswith("172."):
            raise HTTPException(status_code=403, detail="Access restricted to LAN")
        return await call_next(request)
```

---

### 2. **Avoid Authentication for Basic Use, But Gate Advanced Features**

* **Telemetry views** can be unauthenticated.
* **Control/debug actions** (e.g., re-trigger fire, simulate input) should be gated:

  * By `DEBUG=true` environment flag
  * And/or by optional **Basic Auth**
* Optionally include a **signed token** in the URL (e.g., from Frigate or MQTT token)

---

### 3. **Sanitize and Validate All Inputs**

* Never pass MQTT messages or logs directly to the browser without escaping
* Use `html.escape()` or Jinja’s built-in auto-escaping for log values

---

### 4. **Limit External Asset Use**

* Serve all JS/CSS locally (avoid CDNs)
* Prevent connections to third-party domains to keep it air-gapped

---

### 5. **TLS Optional but Recommended in Advanced Deployments**

* Provide self-signed certs for development
* Encourage mounting of real certs via Balena volumes for production

---

## 🚀 PERFORMANCE CONSIDERATIONS

### 1. **Low Resource Use (esp. on Pi 3/5)**

* Use **FastAPI** + **Uvicorn** in async mode
* **Avoid CPU polling**: Use `MQTT callbacks` + **HTMX**/`fetch()` polling every 10–30s
* Render with **Jinja templates** server-side rather than client-heavy JS frameworks (like React)

---

### 2. **Memory-efficient Logging**

* Use an **in-memory ring buffer** (e.g., `collections.deque(maxlen=500)`) for telemetry/events
* Avoid writing logs to disk unless explicitly enabled
* Cap log file sizes (via `logging.handlers.RotatingFileHandler` if enabled)

---

### 3. **Thread Isolation**

* Run the web interface in its own lightweight container
* Isolate from GPIO-heavy services like `gpio_trigger` to avoid contention

---

### 4. **Auto-scaling Views**

* On Pi 3/5, show fewer events by default (`last 20`), expandable
* Add lazy loading for logs
* Avoid rendering graphs client-side — use sparklines or numeric summaries

---

## 📦 SUMMARY: DEFAULT SAFETY MODE

| Feature               | Default Behavior                           |
| --------------------- | ------------------------------------------ |
| LAN-only access       | ✅ Enforced via IP check or port exposure   |
| Auth required         | ❌ Not by default; gated for dev controls   |
| HTTPS                 | ❌ Optional (recomm. if public exposed)     |
| Logs / control access | 🔐 Visible only if `DEBUG=true`            |
| JS/CSS assets         | ✅ Served locally                           |
| MQTT exposure         | 🔒 MQTT broker not exposed to WAN          |
| CPU/memory load       | ✅ Minimal (HTMX, no React, capped buffers) |

