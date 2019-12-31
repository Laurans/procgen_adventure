import getpass
import json
import multiprocessing
import os
import platform
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import git
import pynvml


def is_git_repo(path):
    try:
        _ = git.Repo(path, search_parent_directories=True)
        return True
    except git.exc.InvalidGitRepositoryError:
        return False


class Meta:
    def __init__(self, out_dir="."):
        self.heartbeat_interval_seconds = 15
        self.out_dir = Path(out_dir).resolve()
        self.metadata_fname = "tracking-metadata.json"
        self.fname = str(self.out_dir / self.metadata_fname)
        self._shutdown = False
        try:
            self.data = json.load(open(self.fname))
        except (IOError, ValueError):
            self.data = {}

        self.setup()

    def setup(self):
        self.data["root"] = os.getcwd()
        try:
            import __main__

            self.data["program"] = __main__.__file__
        except (ImportError, AttributeError):
            self.data["program"] = "<python with no main file>"

        program = Path(self.data["root"]) / self.data["program"]
        if is_git_repo(self.data["root"]):
            repo = git.Repo(self.data["root"])
            self.data["git"] = {
                "remote": repo.remotes.origin.url,
                "commit": repo.head.commit.hexsha,
            }

            self.data["root"] = repo.working_tree_dir or self.data["root"]

        self.data["startedAt"] = datetime.utcfromtimestamp(time.time()).isoformat()

        try:
            username = getpass.getuser()
        except KeyError:
            username = str(os.getuid())

        self.data["host"] = socket.gethostname()
        self.data["username"] = username
        self.data["executable"] = sys.executable

        self.data["os"] = platform.platform(aliased=True)
        self.data["python"] = platform.python_version()

        # TODO get docker

        try:
            pynvml.nvmlInit()
            self.data["gpu"] = pynvml.nvmlDeviceGetName(
                pynvml.nvmlDeviceGetHandleByIndex(0)
            ).decode("utf8")
            self.data["gpu_count"] = pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError:
            pass

        try:
            self.data["cpu_count"] = multiprocessing.cpu_count()
        except NotImplementedError:
            pass

        if os.path.exists("/usr/local/cuda/version.txt"):
            self.data["cuda"] = (
                open("/usr/local/cuda/version.txt").read().split(" ")[-1].strip()
            )
        self.data["args"] = sys.argv[1:]
        self.data["state"] = "running"

    def write(self):
        with open(self.fname, "w") as f:
            s = json.dumps(self.data, indent=4)
            f.write(s)
            f.write("\n")
