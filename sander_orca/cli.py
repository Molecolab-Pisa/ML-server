import argparse
import socket

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 3004  # Port to listen on (non-privileged ports are > 1023)


def server():

    parser = argparse.ArgumentParser(description="ML server")

    parser.add_argument(
        "--workdir", type=str, help="Path to local directory on the node", required=True
    )

    parser.add_argument(
        "--model", type=str, help="Model selection: model_vac_1000", required=True
    )

    args = vars(parser.parse_args())

    if args["workdir"] is not None:
        workdir = args.pop("workdir")
    if args["model"] is not None:
        model_string = args.pop("model")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        from .models import available_models

        Model = available_models[model_string](workdir).load()
        while True:
            conn, addr = s.accept()
            with conn:
                print(f"Connection established by {addr}")
                while True:
                    inp = conn.recv(1024)
                    if not inp:
                        break
                    Model.run()
                    conn.sendall(b"model-fin")
                    print(inp.decode())


def orca_client():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(b"model-run")
        out = s.recv(1024)

    print(out.decode())
