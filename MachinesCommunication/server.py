import sys
import os
import time
import socket
    
IP = socket.gethostbyname(socket.gethostname())
PORT = 4455
ADDR = (IP, PORT)
SIZE = 1024
FORMAT = "utf-8"
OBS_FILENAME = 'external_obs.csv'
PRED_FILENAME = 'prediction.csv'

def main():
    print("[STARTING] Server is starting.")
    # Starting a TCP socket:s
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Bind the IP and PORT to the server:
    server.bind(ADDR)
    
    # Server is listening, i.e., the server is now waiting for the client to be connected:
    server.listen()
    print("[LISTENING] Server is listening.")
    
    filename_obs = './MachinesCommunication/' + OBS_FILENAME
    filename_pred = './MachinesCommunication/' + PRED_FILENAME

    while True:
        local_obs_received = False
        
        while not local_obs_received:
            
            # Server has accepted the connection from the client:
            conn, addr = server.accept()
            print(f"\n\n[NEW CONNECTION] {addr} connected.")

            try:
                # Receiving the file data of the local external observation:
                print(f"[RECV] Receiving the local observation file data.")
                data_obs = conn.recv(SIZE).decode(FORMAT)
                # Case in which a message containing a connection interrupt has been received:
                if data_obs=='Test ended':
                    print(f"[EXTERNAL]: {data_obs}")
                    msg = 'Test termination message received'
                    conn.send(msg.encode(FORMAT))
                    # Closing the connection from the server:
                    conn.close()
                    print(f"[DISCONNECTED] {addr} disconnected.\n\n\n")
                    return None
                    
                obs_file = open(filename_obs, "w")
                obs_file.write(data_obs)
                msg = "Obseravation file data received"
                local_obs_received = True
            
            except:
                local_obs_received = False
                msg = "Error: Observation file data not received or not readable."
            
            # Sending message about the observation file to the client:
            conn.send(msg.encode(FORMAT))

        obs_file.close()
        
        pred_file_gen = False
        pred_file_wait_counts = 1
        while not pred_file_gen:
            loading_dot = ' .' 
            # Opening and reading the prediction file:
            if os.path.exists(filename_pred):
                prediction_file = open(filename_pred, 'r')
                pred_file_gen = True
            else:
                pred_file_gen = False
                if pred_file_wait_counts==5:
                    pred_file_wait_counts = 1
                msg = "Warning: prediction file still not available. Wait for it to be generated before sending"
                print(msg + loading_dot*pred_file_wait_counts, end='\r')
                # remove last stdout line:
                sys.stdout.write('\x1b[2K')
                pred_file_wait_counts += 1
                time.sleep(1)
                # It is not necessary to send the warning message to the client:
                #conn.send(error_msg.encode(FORMAT))
        
        # If the prediction file is available, then read and send it:
        if pred_file_gen:
            prediction_data = prediction_file.read()
            # Sending the file data of the prediction:
            conn.send(prediction_data.encode(FORMAT))
            # Reading a message from the external machine ('client') about the prediction file (if it has been received or not):
            msg = conn.recv(SIZE).decode(FORMAT)
            print(f"[EXTERNAL]: {msg}") # [CLIENT]

            # Closing the file sended:
            prediction_file.close()

            # Remove the prediction file in such a way to avoid to send old predictions:
            os.remove(filename_pred)

            # Local observation is reset to False, since after the prediction a new local observation is expected to be received:
            local_obs_received = False

if __name__ == "__main__":
    main()