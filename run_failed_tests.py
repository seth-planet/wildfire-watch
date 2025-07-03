#!/usr/bin/env python3.12
"""
Run all failed tests from the pytest report
"""
import subprocess
import sys

# List of all failed tests from the pytest report
failed_tests = [
    "tests/test_consensus.py::TestDetectionProcessing::test_process_frigate_event",
    "tests/test_consensus.py::TestConsensusLogic::test_multi_camera_consensus",
    "tests/test_consensus.py::TestConsensusLogic::test_consensus_with_camera_priority",
    "tests/test_consensus.py::TestConsensusLogic::test_cooldown_period",
    "tests/test_consensus.py::TestConsensusLogic::test_consensus_with_different_object_types",
    "tests/test_consensus.py::TestConsensusLogic::test_time_window_expiry",
    "tests/test_consensus.py::TestConsensusLogic::test_camera_state_updates",
    "tests/test_consensus.py::TestConsensusEdgeCases::test_rapid_fire_detections",
    "tests/test_consensus.py::TestConsensusEdgeCases::test_camera_reconnection",
    "tests/test_consensus.py::TestConsensusEdgeCases::test_mqtt_reconnection",
    "tests/test_consensus.py::TestConsensusEdgeCases::test_large_scale_deployment",
    "tests/test_consensus.py::TestConsensusEdgeCases::test_memory_leak_prevention",
    "tests/test_consensus.py::TestResilience::test_mqtt_disconnect_handling",
    "tests/test_consensus.py::TestResilience::test_invalid_mqtt_messages",
    "tests/test_consensus.py::TestResilience::test_concurrent_detections",
    "tests/test_consensus.py::TestSystemIntegration::test_full_detection_flow",
    "tests/test_consensus.py::TestSystemIntegration::test_telemetry_reporting",
    "tests/test_consensus.py::TestSystemIntegration::test_error_recovery",
    "tests/test_detect.py::TestCameraDiscovery::test_discover_onvif_cameras",
    "tests/test_detect.py::TestCameraDiscovery::test_discover_ws_discovery",
    "tests/test_detect.py::TestCameraDiscovery::test_discover_subnet_scan",
    "tests/test_detect.py::TestRTSPValidation::test_validate_rtsp_stream",
    "tests/test_detect.py::TestRTSPValidation::test_find_optimal_rtsp_url",
    "tests/test_detect.py::TestRTSPValidation::test_rtsp_timeout_handling",
    "tests/test_detect.py::TestMACTracking::test_get_camera_mac_address",
    "tests/test_detect.py::TestMACTracking::test_load_known_cameras",
    "tests/test_detect.py::TestMACTracking::test_save_known_cameras",
    "tests/test_detect.py::TestFrigateIntegration::test_generate_frigate_config",
    "tests/test_detect.py::TestFrigateIntegration::test_reload_frigate",
    "tests/test_detect.py::TestFrigateIntegration::test_update_frigate_detectors",
    "tests/test_detect.py::TestResilience::test_continuous_discovery",
    "tests/test_detect.py::TestResilience::test_mqtt_reconnection",
    "tests/test_detect.py::TestResilience::test_camera_offline_handling",
    "tests/test_detect.py::TestResilience::test_concurrent_camera_processing",
    "tests/test_model_converter_e2e.py::ModelConverterE2ETests::test_onnx_conversion",
    "tests/test_model_converter_e2e.py::ModelConverterE2ETests::test_hailo_conversion_with_python310",
    "tests/test_model_converter_e2e.py::ModelConverterE2ETests::test_multi_size_conversion",
    "tests/test_model_converter_e2e.py::ModelConverterE2ETests::test_qat_model_conversion",
    "tests/test_integration_e2e.py::TestE2EIntegration::test_service_startup_order"
]

def run_test(test_path):
    """Run a single test and return result"""
    cmd = [
        sys.executable, "-m", "pytest", 
        test_path, 
        "-xvs",
        "--tb=short",
        "--timeout=120"  # 2 minute timeout per test
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Test timed out after 180 seconds"
    except Exception as e:
        return False, "", str(e)

def main():
    print(f"Running {len(failed_tests)} failed tests...")
    print("=" * 80)
    
    passed = []
    failed = []
    
    for i, test in enumerate(failed_tests, 1):
        print(f"\n[{i}/{len(failed_tests)}] Running: {test}")
        success, stdout, stderr = run_test(test)
        
        if success:
            print("✓ PASSED")
            passed.append(test)
        else:
            print("✗ FAILED")
            failed.append(test)
            if stderr:
                print(f"Error: {stderr[:200]}...")
    
    print("\n" + "=" * 80)
    print(f"\nSummary:")
    print(f"  Total tests: {len(failed_tests)}")
    print(f"  Passed: {len(passed)} ({len(passed)/len(failed_tests)*100:.1f}%)")
    print(f"  Failed: {len(failed)} ({len(failed)/len(failed_tests)*100:.1f}%)")
    
    if failed:
        print(f"\nFailed tests:")
        for test in failed:
            print(f"  - {test}")

if __name__ == "__main__":
    main()