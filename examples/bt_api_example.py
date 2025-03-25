import time
import sys
sys.path.append(r'C:\Users\Acer-User\git\pyshimmer')

from serial import Serial

from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket


def stream_cb(pkt: DataPacket) -> None:   
    print(f'Received new data packet: ') 
    for chan in pkt.channels:
        print(f'channel: ' + str(chan)) 
        print(f'value: ' + str(pkt[chan]))     
    print('') 

def main(args=None):    
    # serial = Serial('COM5', DEFAULT_BAUDRATE)
    serial = Serial('COM14', DEFAULT_BAUDRATE)
    shim_dev = ShimmerBluetooth(serial)

    shim_dev.initialize()

    dev_name = shim_dev.get_device_name()
    print(f'My name is: {dev_name}')
    
    # dev_hardware_ver = shim_dev.get_device_hardware_version()
    # print(f'My hardware version is: {dev_hardware_ver}')

    info = shim_dev.get_firmware_version() 
    print("- firmware: [" + str(info[0]) + "]")
    print("- version: [" + str(info[1].major) + "." + str(info[1].minor) + "." + str(info[1].rel) + "]")
    
    # shim_dev.add_stream_callback(stream_cb)

    # shim_dev.start_streaming()
    # time.sleep(5.0)
    # shim_dev.stop_streaming()

    shim_dev.shutdown()


if __name__ == '__main__':
    main()