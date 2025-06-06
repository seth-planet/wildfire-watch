# Hardware Requirements

Complete hardware guide for building a Wildfire Watch system.

## Minimum System Requirements

### Option 1: Raspberry Pi Setup
- **Board:** Raspberry Pi 5 (8GB RAM)
- **Storage:** 128GB A2-rated microSD or NVMe SSD
- **AI Accelerator:** Hailo-8L M.2 module
- **Power:** Official 27W USB-C PSU
- **Case:** With active cooling

### Option 2: x86 PC Setup  
- **CPU:** Intel i5-8500 or AMD Ryzen 5 3600
- **RAM:** 16GB DDR4
- **Storage:** 256GB NVMe SSD
- **AI Accelerator:** Coral M.2/PCIe or NVIDIA GPU
- **Network:** Gigabit Ethernet

## Camera Requirements

### Minimum Specifications
- **Resolution:** 1920x1080 (1080p)
- **Protocol:** RTSP H.264/H.265
- **Framerate:** 15+ FPS
- **Night Vision:** IR illumination
- **Weather Rating:** IP66 or better

### Recommended Cameras
| Model | Price | Resolution | Features |
|-------|-------|------------|----------|
| Dahua IPC-HFW2431S | $80 | 4MP | Starlight, H.265, ONVIF |
| Hikvision DS-2CD2043G2 | $95 | 4MP | AcuSense, H.265+, PoE |
| Amcrest IP4M-1051 | $60 | 4MP | Budget option, PoE |
| Reolink RLC-810A | $75 | 4K | Person/vehicle detection |

### Camera Placement
- **Coverage:** Minimum 2 cameras with overlapping views
- **Height:** 8-12 feet for optimal detection
- **Angle:** 15-30° downward tilt
- **Spacing:** 50-100 feet apart

## AI Accelerators

### Coral TPU
**Best for:** Low power, proven reliability
- **M.2 A+E:** $35 - For mini PCs
- **M.2 B+M:** $35 - For NVMe slots  
- **USB:** $60 - Universal compatibility
- **Performance:** 4 TOPS, ~15ms inference
- **Important:** Requires Python 3.8 for tflite_runtime compatibility

### Hailo-8L
**Best for:** Raspberry Pi 5 native support
- **Price:** $70-80
- **Performance:** 13 TOPS, ~20ms inference
- **Power:** 2.5W typical
- **Driver:** Pre-installed on Pi OS

### Hailo-8
**Best for:** Maximum edge performance
- **Price:** $150-200
- **Performance:** 26 TOPS, ~10ms inference
- **Power:** 5W typical
- **Form Factor:** M.2 A+E or B+M

### NVIDIA GPU
**Best for:** Multiple cameras, development
- **Minimum:** GTX 1650 (budget)
- **Recommended:** RTX 3060 or T400
- **Performance:** 5-10ms inference
- **Power:** 75W+

## Pump Control Hardware

### Relay Module
- **Type:** 2-4 channel relay board
- **Rating:** 10A @ 250VAC minimum
- **Isolation:** Optocoupler required
- **Voltage:** 3.3V or 5V trigger

### Pump Specifications
- **Type:** Centrifugal or positive displacement
- **Flow Rate:** 30-50 GPM minimum
- **Pressure:** 40-60 PSI
- **Power:** 120/240VAC or 12/24VDC

### Safety Components
- **Float Switch:** For tank level sensing
- **Pressure Switch:** Prevent dry running
- **E-Stop Button:** Physical emergency stop
- **Circuit Breaker:** Appropriate amperage

## Network Equipment

### PoE Switch
- **Ports:** 8+ PoE+ ports
- **Power Budget:** 150W minimum
- **Features:** VLAN support recommended
- **Models:** Ubiquiti US-8-150W, TP-Link TL-SG108MPE

### Network Design
- **Topology:** Dedicated camera VLAN
- **Bandwidth:** 50-100 Mbps per camera
- **Cabling:** Cat5e minimum, Cat6 preferred
- **Distance:** PoE limited to 100m

## Power System (Off-Grid)

### Battery Bank
- **Capacity:** 200Ah @ 12V minimum
- **Type:** AGM or LiFePO4
- **Voltage:** 12V or 24V system

### Solar Charging
- **Panels:** 400W minimum
- **Controller:** MPPT 30A+
- **Inverter:** Pure sine wave, 500W+

### Power Consumption
| Component | Watts | Daily (Wh) |
|-----------|-------|------------|
| Pi 5 + Hailo | 10W | 240 |
| 4 Cameras | 20W | 480 |
| Network Switch | 15W | 360 |
| **Total** | **45W** | **1080** |

## Complete Bill of Materials

### Basic 4-Camera System
| Item | Quantity | Unit Price | Total |
|------|----------|------------|-------|
| Raspberry Pi 5 8GB | 1 | $80 | $80 |
| Hailo-8L M.2 | 1 | $75 | $75 |
| 128GB A2 microSD | 1 | $25 | $25 |
| Pi 5 Active Cooler Case | 1 | $15 | $15 |
| 27W USB-C PSU | 1 | $12 | $12 |
| Dahua IPC-HFW2431S | 4 | $80 | $320 |
| 8-Port PoE Switch | 1 | $80 | $80 |
| 4-Channel Relay | 1 | $15 | $15 |
| Float Switch | 1 | $20 | $20 |
| Cat6 Cable (500ft) | 1 | $80 | $80 |
| Weatherproof Box | 1 | $40 | $40 |
| **Total** | | | **$762** |

### High-Performance System
| Item | Quantity | Unit Price | Total |
|------|----------|------------|-------|
| Intel NUC i5 | 1 | $400 | $400 |
| 16GB DDR4 | 1 | $60 | $60 |
| 256GB NVMe | 1 | $40 | $40 |
| Coral M.2 TPU | 1 | $35 | $35 |
| Hikvision 4MP | 8 | $95 | $760 |
| 16-Port PoE Switch | 1 | $200 | $200 |
| Industrial Relay | 1 | $50 | $50 |
| Pressure Sensors | 2 | $30 | $60 |
| **Total** | | | **$1,605** |

## Installation Tools

### Required
- Ethernet cable tester
- Crimping tool + RJ45 ends
- Drill + masonry bits
- Weatherproof connectors
- Cable clips/conduit
- Multimeter

### Recommended  
- PoE tester
- IP camera tester
- Label maker
- Heat shrink tubing
- Dielectric grease
- Security torx bits

## Supplier Recommendations

### USA
- **Cameras:** B&H Photo, Nelly's Security
- **SBCs:** Adafruit, SparkFun, DigiKey
- **Networking:** UI.com, Amazon
- **Electrical:** Home Depot, Grainger

### International
- **Europe:** RS Components, Farnell
- **Asia:** AliExpress (longer lead times)
- **Australia:** Core Electronics, Jaycar

## Expansion Options

### Weather Station
- Add wind speed/direction sensors
- Temperature/humidity monitoring
- Integration via MQTT

### Additional Sensors
- Smoke detectors (interconnected)
- Thermal cameras for hot spots
- Water flow meters

### Remote Monitoring
- 4G/5G cellular modem
- Satellite internet (Starlink)
- LoRaWAN for sensor mesh
