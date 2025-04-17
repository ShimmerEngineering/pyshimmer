import time
import sys
import numpy as np
sys.path.append(r'C:\Users\Acer-User\git\pyshimmer')

from serial import Serial
from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket

def stream_cb(pkt: DataPacket) -> None:   
    print(f'Received new data packet: ') 
    for chan in pkt.channels:
        print(f'channel: ' + str(chan)) 
        print(f'value: ' + str(pkt[chan]))  
    print('') 

def calibrate_inertial_sensor_data(data, alignment, sensitivity, offset):
    """Applies calibration
    Based on the theory outlined by Ferraris F, Grimaldi U, and Parvis M.  
    in "Procedure for effortless in-field calibration of three-axis rate gyros and accelerometers" Sens. Mater. 1995; 7: 311-30.            
    C = [R^(-1)] .[K^(-1)] .([U]-[B])
    where.....
    [C] -> [3 x n] Calibrated Data Matrix 
    [U] -> [3 x n] Uncalibrated Data Matrix
    [B] ->  [3 x n] Replicated Sensor Offset Vector Matrix 
    [R^(-1)] -> [3 x 3] Inverse Alignment Matrix
    [K^(-1)] -> [3 x 3] Inverse Sensitivity Matrix
    n = Number of Samples
    """
    
    # [U] - [B]
    data_minus_offset = np.array(data) - np.array(offset)
    
    # [R^(-1)] Alignment Matrix Inverse
    alignment = np.array(alignment).reshape(3,3)
    if np.all(alignment == 0):
        am_inv = np.eye(3) # Identity Matrix
    else:
        try:
            am_inv = np.linalg.inv(alignment) # Inverse Matrix
        except np.linalg.LinAlgError:
            am_inv = np.eye(3)
            print("Alignment Matrix not invertible - Using Identity Matrix")

    # [K^(-1)] Sensitivity Matrix Inverse
    if np.all(sensitivity == 0):
        sm_inv = np.eye(3) # Identity Matrix
    else:
        try:
            sm_inv = np.linalg.inv(np.diag(sensitivity)) # Inverse Matrix
        except np.linalg.LinAlgError:
            sm_inv = np.eye(3)
            print("Sensitivity Matrix not invertible - Using Identity Matrix")
    
    # C = [R^(-1)] * [K^(-1)] * ([U] - [B])
    calibrated = am_inv @ sm_inv @ data_minus_offset
    return calibrated.flatten().tolist()

def main(args=None):    
    # serial = Serial('COM5', DEFAULT_BAUDRATE)
    serial = Serial('COM14', DEFAULT_BAUDRATE)
    shim_dev = ShimmerBluetooth(serial)

    shim_dev.initialize()

    dev_name = shim_dev.get_device_name()
    print(f'My name is: {dev_name}')
    
    dev_hardware_ver = shim_dev.get_device_hardware_version()
    print(f'My hardware version is: {dev_hardware_ver.name}')

    info = shim_dev.get_firmware_version() 
    print("- firmware: [" + str(info[0]) + "]")
    print("- version: [" + str(info[1].major) + "." + str(info[1].minor) + "." + str(info[1].rel) + "]")
    
    calibration = shim_dev.get_all_calibration()
    print(f'Calibration: {calibration}')
    print(f'Number of Sensors: {calibration._num_sensors}')
    print(f'Number of Bytes: {calibration._num_bytes}')
    
    # Example - Use first sensor LN_ACCEL (index 0)
    sens_num = 0
    offset = calibration.get_offset_bias(sens_num)
    sensitivity = calibration.get_sensitivity(sens_num)
    ali_raw = calibration.get_ali_mat(sens_num)
    alignment = [[ali_raw[0], ali_raw[1], ali_raw[2]],
                 [ali_raw[3], ali_raw[4], ali_raw[5]],
                 [ali_raw[6], ali_raw[7], ali_raw[8]]]
    
    print("Offset Bias:", offset)
    print("Sensitivity:", sensitivity)
    print("Alignment Matrix:")
    for row in alignment:
        print(" ", row)
        
    # Example raw accelerometer data (X, Y, Z)
    raw_data = [100, 100, 100] # To-do (use stream data)
    calibrated = calibrate_inertial_sensor_data(raw_data, alignment, sensitivity, offset)
    
    print("\nExample raw data:", raw_data)
    print("Calibrated data (m/s^2 approx):", calibrated)
    
    # shim_dev.add_stream_callback(stream_cb)

    # shim_dev.start_streaming()
    # time.sleep(0.5)
    # shim_dev.stop_streaming()

    # shim_dev.shutdown()


if __name__ == '__main__':
    main()