
import sys
import importlib.util

def check_dependencies():
    packages = {
        "cv2": "opencv-python",
        "pyaudio": "PyAudio",
        "sounddevice": "sounddevice" # Alternative to pyaudio
    }
    
    print("Checking Sensor Dependencies...")
    results = {}
    for name, pip_name in packages.items():
        spec = importlib.util.find_spec(name)
        results[name] = spec is not None
        status = "✅ Found" if spec else "❌ Missing"
        print(f"  - {name:<15}: {status}")
    
    return results

if __name__ == "__main__":
    results = check_dependencies()
    
    # Try to list devices if modules found
    if results["cv2"]:
        try:
            import cv2
            print("\nTesting Camera...")
            # Windows index 0 usually default camera
            cap = cv2.VideoCapture(0) 
            if cap.isOpened():
                print("  ✅ Camera (Index 0) detected!")
                ret, frame = cap.read()
                if ret:
                    print(f"  ✅ Capture successful: Resolution {frame.shape[1]}x{frame.shape[0]}")
                cap.release()
            else:
                print("  ⚠️ Camera (Index 0) failed to open.")
        except Exception as e:
            print(f"  ❌ Camera error: {e}")

    if results["pyaudio"]:
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            print(f"\nTesting Audio ({p.get_device_count()} devices found)...")
            info = p.get_default_input_device_info()
            print(f"  ✅ Default Mic: {info['name']}")
            p.terminate()
        except Exception as e:
            print(f"  ⚠️ Microphone error: {e}")
