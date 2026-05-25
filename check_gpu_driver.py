"""
Check Intel GPU driver version and OpenVINO GPU configuration
Berdasarkan: https://blog.openvino.ai/blog-posts/install-gpu-drivers-windows-ubuntu
"""
import subprocess
import sys

print("=" * 80)
print("Intel GPU Driver & OpenVINO Configuration Check")
print("=" * 80)

# Check Windows version
print("\n[1] Windows Version:")
try:
    result = subprocess.run(["cmd", "/c", "ver"], capture_output=True, text=True)
    print(result.stdout.strip())
except Exception as e:
    print(f"Error: {e}")

# Check Intel GPU driver version via wmic
print("\n[2] Intel GPU Driver Version:")
try:
    result = subprocess.run(
        ["wmic", "path", "win32_VideoController", "get", "name,driverversion"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
except Exception as e:
    print(f"Error: {e}")

# Check OpenVINO version
print("\n[3] OpenVINO Version:")
try:
    import openvino
    print(f"OpenVINO version: {openvino.__version__}")
except Exception as e:
    print(f"Error: {e}")

# Check if GPU is available in OpenVINO
print("\n[4] OpenVINO GPU Detection:")
try:
    try:
        from openvino.runtime import Core
    except ImportError:
        from openvino import Core

    core = Core()
    devices = core.available_devices
    print(f"Available devices: {devices}")

    if "GPU" in devices:
        print("\n[OK] GPU detected by OpenVINO")
        try:
            full_name = core.get_property("GPU", "FULL_DEVICE_NAME")
            print(f"GPU Name: {full_name}")
        except:
            pass
    else:
        print("\n[WARNING] GPU NOT detected by OpenVINO")
        print("Possible causes:")
        print("  1. Intel GPU driver not installed")
        print("  2. Driver version too old")
        print("  3. GPU not supported by OpenVINO")

except Exception as e:
    print(f"Error: {e}")

# Check Ultralytics version
print("\n[5] Ultralytics Version:")
try:
    import ultralytics
    print(f"Ultralytics version: {ultralytics.__version__}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 80)
print("RECOMMENDATIONS:")
print("=" * 80)

print("""
Berdasarkan dokumentasi Intel OpenVINO:

1. UPDATE DRIVER (RECOMMENDED):
   - Download Intel Arc & Iris Xe Graphics BETA driver:
     https://www.intel.ca/content/www/ca/en/download/729157/intel-arc-iris-xe-graphics-beta-windows.html

   - Atau driver stable terbaru:
     https://www.intel.ca/content/www/ca/en/products/docs/discrete-gpus/arc/software/drivers.html

   - Install dan restart sistem
   - Test kembali dengan: python test_openvino_direct.py

2. VERIFY DRIVER INSTALLATION:
   - Buka Device Manager (devmgmt.msc)
   - Expand "Display adapters"
   - Right-click Intel Iris Xe Graphics > Properties > Driver
   - Check driver version dan date

3. IF STILL FAILS:
   - Gunakan CPU untuk inference (sudah dikonfigurasi di config.py)
   - File issue di: https://github.com/openvinotoolkit/openvino/issues

4. ALTERNATIVE SOLUTIONS:
   - Export model dengan INT8 quantization (kadang work saat FP16/FP32 fail)
   - Gunakan ONNX Runtime dengan DirectML backend
   - Downgrade ke OpenVINO 2024.x (versi stable sebelumnya)

KNOWN ISSUE:
FP16/FP32 models dapat menghasilkan NaN output di Intel GPU dengan
OpenVINO versi tertentu. Ini adalah known bug yang dilaporkan di:
https://community.intel.com/t5/Intel-Distribution-of-OpenVINO/FP16-model-Inference-on-GPU-gives-all-Nan-values-in-output-array/td-p/1386858
""")

print("\n" + "=" * 80)
