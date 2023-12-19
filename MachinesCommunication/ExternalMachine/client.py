import sys
import os
import socket
import argparse
from datetime import datetime

IP = socket.gethostbyname(socket.gethostname())
PORT = 4455
ADDR = (IP, PORT)
SIZE = 1024
FORMAT = "utf-8"
'''
Change the following filepaths according to the path and the name that has been using for the working directory on the external machine.
Note that the observation filepath depends on the external machine, while the prediction filepath depend only
on how the user on the external machine wants to name the prediction file. 
'''
OBS_FILENAME = 'external_obs.csv'
PRED_FILENAME = 'prediction.csv'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop_test", action='store_true',
                        help="Send a message to the server side to let it stop know that the test is ended: in this the server will stop listening accordingly.")
    args = parser.parse_args()

    # Starting a TCP socket:
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Connecting to the server:
    client.connect(ADDR)
        
    # Case in which a message containing a connection interrupt has been sending:
    if args.stop_test: 
        msg = 'Test ended'
        print("[SEND] Sending a test termination message . . .")
        client.send(msg.encode(FORMAT))
        server_msg = client.recv(SIZE).decode(FORMAT)
        print(f"[SERVER]: {server_msg}")
        client.close()
        return None

    filepath_obs = OBS_FILENAME
    filepath_pred = PRED_FILENAME

    # Opening and reading the data of the local external observation file:
    file_obs = open(filepath_obs, "r")
    data_obs = file_obs.read()

    obs_file_sent = False
    obs_attempts = 1
    while not obs_file_sent:

        # Sending the observation file data to the server:
        print("[SEND] Sending the observation file: attempt number {}".format(obs_attempts))
        client.send(data_obs.encode(FORMAT))
        msg = client.recv(SIZE).decode(FORMAT)
        print(f"[SERVER]: {msg}")
        if 'Error' not in msg:
            obs_file_sent = True
            obs_attempts += 1
    
    # Closing the observation file:
    file_obs.close()

    pred_file_received = False
    now = datetime.now()
    filepath_pred += '_' + now.strftime("%m-%d-%Y_%H:%M:%S")
    while not pred_file_received:
        
        # Receiving the prediction file data from the server:
        print("[RECV] Receiving the prediction file data . . .")
        
        try:
            data_pred = client.recv(SIZE).decode(FORMAT)
            # Saving the prediction file received:
            pred_file = open(filepath_pred, "w")
            pred_file.write(data_pred)
            msg = 'Prediction file received'
            pred_file_received = True
        except:
            msg = 'Error: prediction file not received or not readable' 

        client.send(msg.encode(FORMAT))
    
    # Closing the prediction file:
    pred_file.close()
    # Closing the connection from the server:
    client.close()

if __name__ == "__main__":
    main()