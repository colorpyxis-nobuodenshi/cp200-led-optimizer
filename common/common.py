LED_COUNT = 13
LED_CONTROLLER_COM_PORT = "COM10"
PERTIER_CONTROLLER_COM_PORT = "COM9"

emit_mode = "low"
sync_mode = "internal"

optimizer_factor = 1.0

linear_correction = True
white_luminance = 300.0

state = {
    "on": True,
    "luminance": 0,
    "radiance": 0,
    "XYZ": [],
    "lxy": [],
    "lab": [],
    "ct": 0,
    "spectrum": [],
    "led.value": [],
    "led.value2": [],
    "device":
    {
        "temperature": {"room": 0, "board": 0},
        "quantity.of.light": 0.0,
        "emit.mode": "low",
        "sync.mode": "internal"
    }
}
