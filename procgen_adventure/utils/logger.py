import csv
import datetime
import os
import sys
from contextlib import contextmanager

import numpy as np
import torch

from procgen_adventure.utils.tabulate import tabulate


def mkdir_p(path):
    os.makedirs(path, exist_ok=True)


class TerminalTablePrinter:
    def __init__(self):
        self.headers = None
        self.tabulars = []

    def print_tabular(self, new_tabular):
        if self.headers is None:
            self.headers = [x[0] for x in new_tabular]
        else:
            assert len(self.headers) == len(new_tabular)
        self.tabulars.append([x[1] for x in new_tabular])
        self.refresh()

    def refresh(self):
        import os

        rows, columns = os.popen("stty size", "r").read().split()
        tabulars = self.tabulars[-(int(rows) - 3) :]
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(tabulate(tabulars, self.headers))
        sys.stdout.write("\n")


class MyLogger:
    def __init__(self):
        self._prefixes = []
        self._prefix_str = ""

        self._tabular_prefixes = []
        self._tabular_prefix_str = ""

        self._tabular = []

        self._text_outputs = []
        self._tabular_outputs = []

        self._text_fds = {}
        self._tabular_fds = {}  # key: file_name, value: open file
        self._tabular_fds_hold = {}
        self._tabular_header_written = set()

        self._snapshot_dir = None
        self._snapshot_mode = "all"
        self._snapshot_gap = 1

        self._log_tabular_only = False
        self._header_printed = False
        self._disable_prefix = False

        self._tf_summary_dir = None
        self._tf_summary_writer = None

        self._disabled = False
        self._tabular_disabled = False

        self.table_printer = TerminalTablePrinter()
        # keys are file_names and values are the keys of the header of that tabular file
        self._tabular_headers = dict()

    def disable(self):
        self._disabled = True

    def disable_tabular(self):
        self._tabular_disabled = True

    def enable(self):
        self._disabled = False

    def enable_tabular(self):
        self._tabular_disabled = False

    def _add_output(self, file_name, arr, fds, mode="a"):
        if not self._disabled:
            if file_name not in arr:
                mkdir_p(os.path.dirname(file_name))
                arr.append(file_name)
                fds[file_name] = open(file_name, mode)

    def _remove_output(self, file_name, arr, fds):
        if not self._disabled:
            if file_name in arr:
                fds[file_name].close()
                del fds[file_name]
                arr.remove(file_name)

    def push_prefix(self, prefix):
        if not self._disabled:
            self._prefixes.append(prefix)
            self._prefix_str = "".join(self._prefixes)

    def add_text_output(self, file_name):
        if not self._disabled:
            self._add_output(file_name, self._text_outputs, self._text_fds, mode="a")

    def remove_text_output(self, file_name):
        if not self._disabled:
            self._remove_output(file_name, self._text_outputs, self._text_fds)

    def add_tabular_output(self, file_name):
        if not self._disabled:
            if file_name in self._tabular_fds_hold.keys():
                self._tabular_outputs.append(file_name)
                self._tabular_fds[file_name] = self._tabular_fds_hold[file_name]
            else:
                self._add_output(
                    file_name, self._tabular_outputs, self._tabular_fds, mode="w"
                )

    def remove_tabular_output(self, file_name):
        if not self._disabled:
            if file_name in self._tabular_header_written:
                self._tabular_header_written.remove(file_name)
            self._remove_output(file_name, self._tabular_outputs, self._tabular_fds)

    def hold_tabular_output(self, file_name):
        if not self._disabled:
            if file_name in self._tabular_outputs:
                self._tabular_outputs.remove(file_name)
                self._tabular_fds_hold[file_name] = self._tabular_fds.pop(file_name)

    def set_snapshot_dir(self, dir_name):
        os.system(f"mkdir -p {dir_name}")
        self._snapshot_dir = dir_name

    def get_snapshot_dir(self):
        return self._snapshot_dir

    def set_tf_summary_dir(self, dir_name):
        self._tf_summary_dir = dir_name

    def get_tf_summary_dir(self):
        return self._tf_summary_dir

    def set_tf_summary_writer(self, writer_name):
        self._tf_summary_writer = writer_name

    def get_tf_summary_writer(self):
        return self._tf_summary_writer

    def get_snapshot_mode(self):
        return self._snapshot_mode

    def set_snapshot_mode(self, mode):
        self._snapshot_mode = mode

    def set_snapshot_gap(self, gap):
        self._snapshot_gap = gap

    def set_log_tabular_only(self, log_tabular_only):
        self._log_tabular_only = log_tabular_only

    def get_log_tabular_only(self):
        return self._log_tabular_only

    def set_disable_prefix(self, disable_prefix):
        self._disable_prefix = disable_prefix

    def get_disable_prefix(self):
        return self._disable_prefix

    def log(self, s, with_prefix=True, with_timestamp=True):
        if not self._disabled:
            out = s

            if with_prefix and not self._disable_prefix:
                out = self._prefix_str + out

            if with_timestamp:
                now = datetime.datetime.now()
                timestamp = now.strftime("%Y-%m-%d %H:%M:%S.%f %Z")
                out = f"{timestamp} | {out}"

            if not self._log_tabular_only:
                # Also log to stdout
                print(out)
                for fd in list(self._text_fds.values()):
                    fd.write(out + "\n")
                    fd.flush()
                sys.stdout.flush()

    def logkv(self, key, val):
        if not self._disabled:
            self.record_tabular(key, val)

    def record_tabular(self, key, val, *args, **kwargs):
        if not self._disabled:
            self._tabular.append((self._tabular_prefix_str + str(key), str(val)))

    def push_tabular_prefix(self, key):
        if not self._disabled:
            self._tabular_prefixes.append(key)
            self._tabular_prefix_str = "".join(self._tabular_prefixes)

    def pop_tabular_prefix(self):
        if not self._disabled:
            del self._tabular_prefixes[-1]
            self._tabular_prefix_str = "".join(self._tabular_prefixes)

    def pop_prefix(self):
        if not self._disabled:
            del self._prefixes[-1]
            self._prefix_str = "".join(self._prefixes)

    def dump_tabular(self, *args, **kwargs):
        if not self._disabled:
            wh = kwargs.pop("write_header", None)
            if len(self._tabular) > 0:
                if self._log_tabular_only:
                    self.table_printer.print_tabular(self._tabular)
                else:
                    for line in tabulate(self._tabular).split("\n"):
                        self.log(line, *args, **kwargs)
                if not self._tabular_disabled:
                    tabular_dict = dict(self._tabular)
                    # Also write to the csv files
                    # This assumes that the keys in each iteration won't change
                    for tabular_file_name, tabular_fd in list(
                        self._tabular_fds.items()
                    ):
                        keys = tabular_dict.keys()
                        if tabular_file_name in self._tabular_headers:
                            # check against existing keys: if new keys re-write Header and pad with NaNs
                            existing_keys = self._tabular_headers[tabular_file_name]
                            if not set(existing_keys).issuperset(set(keys)):
                                joint_keys = set(keys).union(set(existing_keys))
                                tabular_fd.flush()
                                read_fd = open(tabular_file_name, "r")
                                reader = csv.DictReader(read_fd)
                                rows = list(reader)
                                read_fd.close()
                                tabular_fd.close()
                                tabular_fd = self._tabular_fds[
                                    tabular_file_name
                                ] = open(tabular_file_name, "w")
                                new_writer = csv.DictWriter(
                                    tabular_fd, fieldnames=list(joint_keys)
                                )
                                new_writer.writeheader()
                                for row in rows:
                                    for key in joint_keys:
                                        if key not in row:
                                            row[key] = np.nan
                                new_writer.writerows(rows)
                                self._tabular_headers[tabular_file_name] = list(
                                    joint_keys
                                )
                        else:
                            self._tabular_headers[tabular_file_name] = keys

                        writer = csv.DictWriter(
                            tabular_fd,
                            fieldnames=self._tabular_headers[tabular_file_name],
                        )  # list(
                        if wh or (
                            wh is None
                            and tabular_file_name not in self._tabular_header_written
                        ):
                            writer.writeheader()
                            self._tabular_header_written.add(tabular_file_name)
                            self._tabular_headers[tabular_file_name] = keys
                        # add NaNs in all empty fields from the header
                        for key in self._tabular_headers[tabular_file_name]:
                            if key not in tabular_dict:
                                tabular_dict[key] = np.nan
                        writer.writerow(tabular_dict)
                        tabular_fd.flush()
                del self._tabular[:]

    def save_itr_params(self, itr, params, force=False):
        if not self._disabled:
            if self._snapshot_dir:
                if self._snapshot_mode == "all" or force:
                    file_name = os.path.join(self.get_snapshot_dir(), f"itr_{int(itr)}")
                elif self._snapshot_mode == "last":
                    file_name = os.path.join(self.get_snapshot_dir(), "params.pkl")
                elif self._snapshot_mode == "gap":
                    print(itr, itr + 1 % self._snapshot_gap)
                    if itr == 0 or (itr + 1) % self._snapshot_gap == 0:
                        file_name = os.path.join(
                            self.get_snapshot_dir(), f"itr_{int(itr)}.pkl"
                        )
                    else:
                        return None
                elif self._snapshot_mode == "none":
                    return None
                else:
                    raise NotImplementedError

                breakpoint
                torch.save(params, file_name)

    def record_tabular_misc_stat(self, key, values, placement="back"):
        if not self._disabled:
            if placement == "front":
                prefix = ""
                suffix = key
            else:
                prefix = key
                suffix = ""

            if len(values) > 0:
                self.record_tabular(prefix + "Average" + suffix, np.average(values))
                self.record_tabular(prefix + "Std" + suffix, np.std(values))
                self.record_tabular(prefix + "Median" + suffix, np.median(values))
                self.record_tabular(prefix + "Min" + suffix, np.min(values))
                self.record_tabular(prefix + "Max" + suffix, np.max(values))
            else:
                self.record_tabular(prefix + "Average" + suffix, np.nan)
                self.record_tabular(prefix + "Std" + suffix, np.nan)
                self.record_tabular(prefix + "Median" + suffix, np.nan)
                self.record_tabular(prefix + "Min" + suffix, np.nan)
                self.record_tabular(prefix + "Max" + suffix, np.nan)

    @contextmanager
    def prefix(self, key):
        self.push_prefix(key)
        try:
            yield
        finally:
            self.pop_prefix()

    @contextmanager
    def tabular_prefix(self, key):
        self.push_tabular_prefix(key)
        yield
        self.pop_tabular_prefix()
