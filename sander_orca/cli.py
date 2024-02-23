import sys
import os
import socket

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
DEFAULT_PORT = 3004  # Port to listen on (non-privileged ports are > 1023)


# ============================================================
# Helpers
# ============================================================

def get_port():
    port = os.getenv("SANDER_ORCA_PORT")
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

    parser = argparse.ArgumentParser(description="ML server")

    # required arguments
    required = parser.add_argument_group("required")

    required.add_argument(
        "--model",
        required=True,
        default=None,
        type=str,
        help="Model string selector (e.g., model_vac_1000)",
    )

    # optional arguments
    optional = parser.add_argument_group("optional")

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
    from .models import available_models
    import numpy as np

    args, parser = server_cli_parse()

    def logprint(msg):
        "prints to file (stdout or logfile)"
        print(msg, file=args.logstream)

    logprint("Activating the ML server...")
    logprint(f"Server running on PID {os.getpid()}")
    logprint("Server is running with the following arguments:")
    for arg in vars(args):
        logprint("\t{:40s} = {:40s}".format(arg, str(getattr(args, arg))))

    # read the port
    PORT = get_port()
    logprint(f"Requested opening a socket on {HOST}:{PORT}")

    # AF_INET: internet address family for IPv4
    # SOCK_STREAM: Transmission Control Protocol (TCP) socket type
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # bind and wait for connections
        s.bind((HOST, PORT))
        s.listen()

        # load the requested model
        if not args.model in available_models.keys():
            raise ValueError("requested model is not available")

        model = available_models[args.model](args.workdir).load()

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
                    inp = conn.recv(1024)
                    if not inp:
                        break

                    # check whether a model-run is requested
                    cmd = inp.decode()
                    if cmd == "model-run":
                        logprint(
                            "Requested calculation by sander.\nReading the ptchrg and inpfile..."
                        )

                        # read input, predict, and write to file
                        model.run()

                        # tell the client that we have finished
                        conn.sendall(b"model-fin")

                    elif cmd == "server-stop":
                        logprint("Received request of server stop.\nStopping now...")
                        sys.exit(0)


# ============================================================
# Clients
# ============================================================


def orca_client():
    "client sending a model-run request to the server"
    PORT = get_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(b"model-run")
        out = s.recv(1024)


def stop_server():
    "client sending a server-stop request to the server"
    PORT = get_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(b"server-stop")
        out = s.recv(1024)
