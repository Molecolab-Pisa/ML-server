import os
import socket
import sys

from .socket_utils import recvall

HOST = "127.0.0.1"
DEFAULT_PORT = 3004


# ============================================================
# Helpers
# ============================================================


def get_port():
    port = os.getenv("ML_SERVER_PORT")
    if port is None:
        port = DEFAULT_PORT
    else:
        port = int(port)
    return port


# ============================================================
# Server
# ============================================================


def server_cli_parse():
    "command-line interface parser for the server"
    import argparse

    from .models import list_available_models

    def exitparse(func):
        """decorator that executes a function and exits
        the parsing stage prematurely
        """

        class ExitParser(argparse.Action):
            def __call__(self, parser, namespace, values, option_string):
                func()
                parser.exit()

        return ExitParser

    parser = argparse.ArgumentParser(description="ML-server")

    # required arguments
    required = parser.add_argument_group("required")

    required.add_argument(
        "--model_vac",
        required=True,
        default=None,
        type=str,
        help="QM model string selector (e.g., model_vac_gs)",
    )

    # optional arguments
    optional = parser.add_argument_group("optional")

    optional.add_argument(
        "--model_env",
        required=None,
        default=None,
        type=str,
        help="Environment model string selector (e.g., model_env_gs)",
    )

    optional.add_argument(
        "--filebased",
        action="store_true",
        required=None,
        help="Use file-based interface",
    )

    optional.add_argument(
        "--logfile",
        required=False,
        default=None,
        type=str,
        help="Absolute path to the log file. (default: standard output)",
    )

    optional.add_argument(
        "--workdir",
        required=False,
        default=os.getcwd(),
        type=str,
        help="Working directory, where input and output files are written. (default: directory where the server is called)",
    )

    optional.add_argument(
        "--list",
        required=False,
        nargs=0,
        action=exitparse(list_available_models),
        help="List the available ML models and exit.",
    )

    args = parser.parse_args()

    # set a logstream to point to stdout or open a IO stream to file if requested
    if args.logfile is None:
        args.logstream = sys.stdout
    else:
        args.logstream = open(args.logfile, "w")

    return args, parser


def server():
    import struct

    import jax.numpy as jnp
    import numpy as np

    from .models import available_models

    dtype = np.float64

    args, parser = server_cli_parse()

    def logprint(msg):
        "prints to file (stdout or logfile)"
        print(msg, file=args.logstream)

    logprint("Activating the ML-server...")
    logprint(f"Server running on PID {os.getpid()}")
    logprint("Server is running with the following arguments:")
    for arg in vars(args):
        logprint("\t{:40s} = {:40s}".format(arg, str(getattr(args, arg))))

    # read the port
    PORT = get_port()
    logprint(f"Requested opening a socket on {HOST}:{PORT}")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # bind and wait for connections
        s.bind((HOST, PORT))
        s.listen()

        # load the requested model
        if args.model_vac not in available_models.keys():
            raise ValueError("requested QM model is not available")
        model = available_models[args.model_vac](args.workdir).load()

        if args.model_env is not None:
            if args.model_env not in available_models.keys():
                raise ValueError("requested environment model is not available")
            model = available_models[args.model_env](
                args.workdir, model_vac=model
            ).load()

        # keep listening and accepting connections from clients
        while True:
            conn, (host, port) = s.accept()

            # open the socket representing the connection
            with conn:
                logprint(
                    f"Performed three-way handshake on host={host}, port={port}.\nClient accepted..."
                )

                # keep receiving input from the client
                while True:
                    inp = conn.recv(12)
                    if not inp:
                        break

                    # check whether a model-run is requested
                    cmd = inp.decode().strip()
                    print(cmd)

                    if args.filebased:
                        if cmd == "model-run":
                            logprint("Requested calculation by sander.\n")
                            # read input, predict, and write to file
                            model.run(filebased=True)
                            conn.sendall(b"model-fin   ")

                        elif cmd == "server-stop":
                            logprint(
                                "Received request of server stop.\nStopping now..."
                            )
                            sys.exit(0)

                    else:
                        if cmd == "model-run":
                            # receive data via socket
                            system_data = jnp.zeros((2, 1), dtype)
                            system_data = recvall(conn, system_data)
                            nqm, nmm = int(system_data[0]), int(system_data[1])

                            sh_qm = (nqm, 3)
                            coords_qm = jnp.zeros(sh_qm, dtype)
                            coords_qm = recvall(conn, coords_qm)

                            if nmm > 0:

                                sh_mm = (nmm, 4)
                                mmcoordchg = jnp.zeros(sh_mm, dtype)
                                mmcoordchg = recvall(conn, mmcoordchg)

                                coords_mm = mmcoordchg[:, :3]
                                charges_mm = mmcoordchg[:, 3]

                                # run the prediction
                                energy, grad_qm, grad_mm = model.run(
                                    coords_qm,
                                    coords_mm,
                                    charges_mm,
                                    filebased=False,
                                )
                                conn.sendall(grad_mm)
                            else:
                                energy, grad_qm = model.run(coords_qm, filebased=False)

                            conn.sendall(grad_qm)
                            conn.sendall(struct.pack("<d", energy))

                            # tell the client that we have finished
                            conn.sendall(b"model-fin   ")

                        elif cmd == "server-stop":
                            logprint(
                                "Received request of server stop.\nStopping now..."
                            )
                            sys.exit(0)


# ============================================================
# Clients
# ============================================================


def orca_client():
    "client sending a model-run request to the server"
    PORT = get_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(b"model-run   ")
        s.recv(12)


def stop_server():
    "client sending a server-stop request to the server"
    PORT = get_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(b"server-stop ")
        s.recv(12)
