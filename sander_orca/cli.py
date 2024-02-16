import socket

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 3004  # Port to listen on (non-privileged ports are > 1023)

def server():

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST,PORT))
        s.listen()
        while True:
            conn, addr = s.accept()
            with conn:
                print(f'Connection established by {addr}')
                while True:
                    inp = conn.recv(1024)
                    if not inp:
                        break
                    conn.sendall(b'model-fin')
                    print(inp.decode())

def orca_client():

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST,PORT))
        s.sendall(b'model-run')
        out = s.recv(1024)

    print(out.decode())

