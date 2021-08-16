import serial
from datetime import datetime
 
sensor = "DH22"
serial_port = '/dev/ttyACM0'
baud_rate = 9600
path = "temp_logs/temp_log.txt"
ser = serial.Serial(serial_port, baud_rate)
with open(path, 'a', buffering=1) as f:
    while True:
        line = ser.readline().strip().decode("utf-8")
        time=datetime.now().strftime("%d/%m/%Y %H:%M:%S.%f")
        f.write(line+";"+time+"\n")