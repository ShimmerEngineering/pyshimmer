import time
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import collections

sys.path.append(r'C:\Github\pyshimmer')
from threading import Thread
from serial import Serial
from pyshimmer.dev.channels import ChDataTypeAssignment, ChannelDataType, EChannelType, ESensorGroup
from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket

# Global variables
data_buffer = None
fig, ax = plt.subplots()
lines = {}

def init_plot():
    """Initialize an empty plot."""
    ax.set_xlim(0, 200)  
    ax.set_ylim(-100, 100)  
    return lines.values()

def update_plot(_):
    """Update the plot dynamically."""
    if data_buffer:
        ax.clear()
        for chan, values in data_buffer.items():
            ax.plot(values, label=f'Channel {chan}')
        ax.legend()
    return lines.values()

def stream_cb(pkt: DataPacket) -> None:
    """Callback function to receive streaming data."""
    global data_buffer

    if data_buffer is None:
        # Initialize buffer for each channel, excluding TIMESTAMP
        data_buffer = {chan: collections.deque(maxlen=200) for chan in pkt.channels if chan != EChannelType.TIMESTAMP}

    print(f'Received new data packet:')
    for chan in pkt.channels:
        if chan == EChannelType.TIMESTAMP:
            continue  # Skip timestamp channel

        print(f'Channel: {chan}, Value: {pkt[chan]}')
        data_buffer[chan].append(pkt[chan])  # Store in rolling buffer

def main(args=None):    
    serial = Serial('COM13', DEFAULT_BAUDRATE)
    shim_dev = ShimmerBluetooth(serial)

    shim_dev.initialize()

    dev_name = shim_dev.get_device_name()
    print(f'My name is: {dev_name}')

    info = shim_dev.get_firmware_version()
    print("- firmware: [" + str(info[0]) + "]")
    print("- version: [" + str(info[1].major) + "." + str(info[1].minor) + "." + str(info[1].rel) + "]")
    
    shim_dev.add_stream_callback(stream_cb)
    sensor_list = [ESensorGroup.ACCEL_LN, ESensorGroup.ACCEL_WR]
    shim_dev.set_sensors(sensor_list);
    # Start a thread for streaming so the plot remains responsive
    def stream_thread():
        shim_dev.start_streaming()
        print("Streaming started. Press any key to stop...")
        input()  # Wait for user input before stopping
        shim_dev.stop_streaming()
        shim_dev.shutdown()
        print("Streaming stopped.")

    Thread(target=stream_thread, daemon=True).start()

    # Start real-time plotting
    ani = FuncAnimation(fig, update_plot, init_func=init_plot, interval=100, blit=False)
    plt.show()  # Start Matplotlib loop

if __name__ == '__main__':
    main()
