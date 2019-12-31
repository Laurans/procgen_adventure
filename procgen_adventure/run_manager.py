import datetime
import os
import signal
import sys

import mlflow
import shortuuid
from meta import Meta
from stats import SystemStats

RUN_DIR = "./tracking_logs/"


def generate_id():
    # ~3t run ids (36**8)
    run_gen = shortuuid.ShortUUID(alphabet=list("0123456789abcdefghijklmnopqrstuvwxyz"))
    return run_gen.random(8)


class RunManager:
    def __init__(self, config, job_type, dry=False):
        if dry:
            prefix = "dryrun"
        else:
            prefix = "run"
        time_str = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_id = generate_id()
        self._run_dir = os.path.join(
            RUN_DIR, "{}-{}-{}".format(prefix, time_str, run_id)
        )
        os.makedirs(self._run_dir)
        self.config = config
        self._meta = Meta(out_dir=self._run_dir)
        self._meta.data["jobType"] = job_type
        self._meta.write()

        self._system_stats = SystemStats()
        self.sig = signal.SIGINT
        self.exit = False

    def __enter__(self):
        self.interrupted = False

        def handler(signum, frame):
            self.interrupted = True
            sys.exit()

        signal.signal(self.sig, handler)

        self._run = mlflow.start_run()
        mlflow.log_params(self.config)
        mlflow.log_artifact(self._meta.fname)

        # Write meta save artifact
        self._system_stats.start()

    def __exit__(self, *args):
        self._system_stats.shutdown()
        if self.interrupted:
            self._meta.data["state"] = "killed"
        elif args[0] is None:
            self._meta.data["state"] = "finished"
        else:
            self._meta.data["state"] = "failed"

        mlflow.set_tag("status", self._meta.data["state"])

        # Write meta save artifact
        self._meta.write()
        mlflow.log_artifact(self._meta.fname)

        mlflow.end_run()
