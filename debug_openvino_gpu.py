"""
Debug OpenVINO Intel GPU issue - cek apakah GPU terdeteksi dengan benar
"""
try:
    # Try different import paths for different OpenVINO versions
    try:
        from openvino.runtime import Core
    except ImportError:
        from openvino import Core

    print("OpenVINO Core Information:")
    print("=" * 80)

    core = Core()

    # List available devices
    devices = core.available_devices
    print(f"\nAvailable devices: {devices}")

    # Get device properties
    for device in devices:
        print(f"\n[Device: {device}]")
        try:
            props = core.get_property(device, "FULL_DEVICE_NAME")
            print(f"  Full name: {props}")
        except:
            pass

        try:
            props = core.get_property(device, "OPTIMIZATION_CAPABILITIES")
            print(f"  Optimization capabilities: {props}")
        except:
            pass

    # Test loading model
    print("\n" + "=" * 80)
    print("\nTesting model loading on different devices:")

    model_path = "4th-trainingyolo26m640_openvino_model/4th-trainingyolo26m640.xml"

    for device in ["CPU", "GPU"]:
        if device in devices or f"{device}.0" in devices:
            try:
                print(f"\n[{device}] Loading model...")
                compiled_model = core.compile_model(model_path, device)
                print(f"  [OK] Model loaded successfully on {device}")

                # Get input/output info
                input_layer = compiled_model.input(0)
                output_layer = compiled_model.output(0)
                print(f"  Input shape: {input_layer.shape}")
                print(f"  Output shape: {output_layer.shape}")

            except Exception as e:
                print(f"  [FAIL] Failed to load on {device}: {e}")
        else:
            print(f"\n[{device}] Not available")

except ImportError:
    print("ERROR: openvino.runtime not found")
    print("Install with: pip install openvino")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
