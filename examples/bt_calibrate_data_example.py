import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import collections
sys.path.append(r'C:\Users\Acer-User\git\pyshimmer')

from serial import Serial
from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket
from pyshimmer.dev.channels import EChannelType, ESensorGroup
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import CheckButtons, Button
from threading import Thread

fig, axs = plt.subplots(4, 1, figsize=(10, 7.5), sharex=True)
plt.subplots_adjust(left=0.3, right=0.95, top=0.95, bottom=0.05, hspace=0.25)
sensor_plot_order = ['ACCEL_LN', 'GYRO', 'MAG', 'ACCEL_WR']

# Plot Buffers
data_buffer = {sensor: {'X': collections.deque(maxlen=200),
                        'Y': collections.deque(maxlen=200),
                        'Z': collections.deque(maxlen=200)}
               for sensor in sensor_plot_order}
enabled_sensors = []

# UI Buttons
sensor_labels = ['ACCEL_LN', 'GYRO', 'MAG', 'ACCEL_WR']
sensor_map = {'ACCEL_LN': ESensorGroup.ACCEL_LN,
              'GYRO': ESensorGroup.GYRO,
              'MAG': ESensorGroup.MAG,
              'ACCEL_WR': ESensorGroup.ACCEL_WR}

check_ax = plt.axes([0.05, 0.42, 0.2, 0.15])  
check = CheckButtons(check_ax, sensor_labels, [False] * 4)

btn_start_ax = plt.axes([0.05, 0.32, 0.1, 0.05])  
btn_stop_ax = plt.axes([0.15, 0.32, 0.1, 0.05]) 
btn_clear_ax = plt.axes([0.05, 0.24, 0.2, 0.05])
btn_start = Button(btn_start_ax, 'Start')
btn_stop = Button(btn_stop_ax, 'Stop')
btn_clear = Button(btn_clear_ax, 'Clear Plots')

line_objs = {
    sensor: {
        axis: None for axis in ['X', 'Y', 'Z']
    } for sensor in sensor_plot_order
}

def init_plot():
    for ax, sensor in zip(axs, sensor_plot_order):
        ax.set_xlim(0, 200)
        ax.set_ylim(-1, 1)
        ax.set_title(f'{sensor} Calibrated Data')
        for axis in ['X', 'Y', 'Z']:
            (line,) = ax.plot([], [], label=axis)
            line_objs[sensor][axis] = line
        ax.legend()
    return [line for sensor_lines in line_objs.values() for line in sensor_lines.values()]

def update_plot(_):
    for sensor, ax in zip(sensor_plot_order, axs):
        for axis in ['X', 'Y', 'Z']:
            buf = data_buffer[sensor][axis]
            line = line_objs[sensor][axis]
            line.set_data(range(len(buf)), list(buf))

        # Adjust dynamic y-limits
        all_vals = np.concatenate([data_buffer[sensor][a] for a in ['X', 'Y', 'Z']])
        if len(all_vals):
            min_y, max_y = np.min(all_vals), np.max(all_vals)
            margin = (max_y - min_y) * 0.1 # 10% margin
            if margin == 0:
                margin = 0.001
            ax.set_ylim(min_y - margin, max_y + margin)
        ax.set_xlim(0, 200)

    return [line for sensor_lines in line_objs.values() for line in sensor_lines.values()]

def make_stream_cb(calibration):
    def stream_cb(pkt: DataPacket) -> None:
        
        # print(f'received new data packet: ') 
        # for chan in pkt.channels:
        #     print(f'channel: ' + str(chan)) 
        #     print(f'value: ' + str(pkt[chan]))     
        # print('') 

        internal_map = {
            'ACCEL_LN': [EChannelType.ACCEL_LN_X, EChannelType.ACCEL_LN_Y, EChannelType.ACCEL_LN_Z],
            'GYRO': [EChannelType.GYRO_MPU9150_X, EChannelType.GYRO_MPU9150_Y, EChannelType.GYRO_MPU9150_Z],
            'MAG': [EChannelType.MAG_LSM303DLHC_X, EChannelType.MAG_LSM303DLHC_Y, EChannelType.MAG_LSM303DLHC_Z],
            'ACCEL_WR': [EChannelType.ACCEL_LSM303DLHC_X, EChannelType.ACCEL_LSM303DLHC_Y, EChannelType.ACCEL_LSM303DLHC_Z],
        }

        for sensor_name in enabled_sensors:
            channels = internal_map[sensor_name]
            if all(ch in pkt.channels for ch in channels):
                raw = [pkt[ch] for ch in channels]
                idx = list(sensor_plot_order).index(sensor_name)
                offset = calibration.get_offset_bias(idx)
                sensitivity = calibration.get_sensitivity(idx)
                ali_raw = calibration.get_ali_mat(idx)
                alignment = [[ali_raw[0], ali_raw[1], ali_raw[2]],
                             [ali_raw[3], ali_raw[4], ali_raw[5]],
                             [ali_raw[6], ali_raw[7], ali_raw[8]]]
                calib = calibrate_inertial_sensor_data(raw, alignment, sensitivity, offset)
                data_buffer[sensor_name]['X'].append(calib[0])
                data_buffer[sensor_name]['Y'].append(calib[1])
                data_buffer[sensor_name]['Z'].append(calib[2])
    return stream_cb

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
    return [round(val, 3) for val in calibrated.flatten().tolist()]

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
    
    # Calibrated Stream Data
    shim_dev.add_stream_callback(make_stream_cb(calibration))
    
    # Button Callbacks
    def on_checkbox_clicked(label):
        if label in enabled_sensors:
            enabled_sensors.remove(label)
        else:
            enabled_sensors.append(label)
        print(f"Selected sensors: {enabled_sensors}")

    def on_start_clicked(event):
        sensor_groups = [sensor_map[s] for s in enabled_sensors]
        shim_dev.set_sensors(sensor_groups)
        shim_dev.start_streaming()
        print("Streaming started")

    def on_stop_clicked(event):
        shim_dev.stop_streaming()
        print("Streaming stopped")
        
    def on_clear_clicked(event):
        for sensor in data_buffer:
            for axis in data_buffer[sensor]:
                data_buffer[sensor][axis].clear()
        print("Plot cleared")
    
    check.on_clicked(on_checkbox_clicked)
    btn_start.on_clicked(on_start_clicked)
    btn_stop.on_clicked(on_stop_clicked)
    btn_clear.on_clicked(on_clear_clicked)
    
    # Start real-time plotting
    ani = FuncAnimation(fig, update_plot, init_func=init_plot, interval=100, blit=False)
    plt.show()
    shim_dev.shutdown()


if __name__ == '__main__':
    main()