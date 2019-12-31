import collections
import os
import threading
import time
from numbers import Number

import mlflow
import psutil
from pynvml import *


class SystemStats:
    def __init__(self):
        try:
            nvmlInit()
            self.gpu_count = nvmlDeviceGetCount()
        except NVMLError as err:
            self.gpu_count = 0
        self.sampler = {}
        self.samples = 0
        self._shutdown = False
        self.pid = os.getpid()
        """Sample system stats every this many seconds, defaults to 2, min is 0.5"""
        self.sample_rate_seconds = 2
        """The number of samples to average before pushing, defaults to 15 valid range (2:30)"""
        self.samples_to_average = 15

        net = psutil.net_io_counters()
        self.network_init = {"sent": net.bytes_sent, "recv": net.bytes_recv}

        self._thread = threading.Thread(target=self._thread_body)
        self._thread.daemon = True
        self.step = 0

    def start(self):
        self._thread.start()

    @property
    def proc(self):
        return psutil.Process(pid=self.pid)

    def _thread_body(self):
        while True:
            stats = self.stats()
            for stat, value in stats.items():
                if isinstance(value, Number):
                    self.sampler[stat] = self.sampler.get(stat, [])
                    self.sampler[stat].append(value)
            self.samples += 1
            if self._shutdown or self.samples >= self.samples_to_average:
                self.flush()
                if self._shutdown:
                    break

            seconds = 0
            while seconds < self.sample_rate_seconds:
                time.sleep(0.1)
                seconds += 0.1
                if self._shutdown:
                    break

    def shutdown(self):
        self._shutdown = True
        try:
            self._thread.join()
        except RuntimeError:
            pass

    def flush(self):
        stats = self.stats()
        system_stats = {}
        for stat, value in stats.items():
            if isinstance(value, Number):
                samples = list(self.sampler.get(stat, [stats[stat]]))
                system_stats[stat] = round(sum(samples) / len(samples), 2)

        mlflow.log_metrics(system_stats, step=self.step)
        self.samples = 0
        self.sampler = {}
        self.step += 1

    def stats(self):
        stats = {}
        for i in range(0, self.gpu_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            try:
                util = nvmlDeviceGetUtilizationRates(handle)
                memory = nvmlDeviceGetMemoryInfo(handle)
                temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
                stats[f"gpu.{i}.gpu"] = util.gpu
                stats[f"gpu.{i}.memory"] = util.memory
                stats[f"gpu.{i}.memory_allocated"] = memory.used / memory.total * 100
                stats[f"gpu.{i}.temp"] = temp
            except NVMLError as err:
                pass

        net = psutil.net_io_counters()
        sysmem = psutil.virtual_memory()
        stats["cpu"] = psutil.cpu_percent()
        stats["memory"] = sysmem.percent
        stats["network"] = {
            "sent": net.bytes_sent - self.network_init["sent"],
            "recv": net.bytes_recv - self.network_init["recv"],
        }
        stats["disk"] = psutil.disk_usage("/").percent
        stats["proc.memory.availableMB"] = sysmem.available / 1048576.0
        try:
            stats["proc.memory.rssMB"] = self.proc.memory_info().rss / 1048576.0
            stats["proc.memory.percent"] = self.proc.memory_percent()
            stats["proc.cpu.threads"] = self.proc.num_threads()
        except psutil.NoSuchProcess:
            pass
        return stats
