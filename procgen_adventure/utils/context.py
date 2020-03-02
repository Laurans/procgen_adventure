import datetime
from contextlib import contextmanager
from pathlib import Path

from procgen_adventure.utils.logger import logger

PROJECT_DIR = Path(__file__).parent.parent.parent
LOG_DIR = PROJECT_DIR / "experiments"


def get_log_dir(run_ID, name):
    yyyymmdd = datetime.datetime.today().strftime("%Y%m%d")
    log_dir = LOG_DIR / yyyymmdd / name
    return log_dir.resolve()


@contextmanager
def logger_context(run_ID, name, snapshot_mode="none"):
    log_dir = get_log_dir(run_ID, name)
    logger.set_snapshot_mode(snapshot_mode)
    logger.set_log_tabular_only(False)

    log_dir = log_dir / f"run_{run_ID}"
    exp_dir = log_dir.resolve()
    tabular_log_file = str(exp_dir / "progress.csv")
    text_log_file = str(exp_dir / "debug.log")

    logger.set_snapshot_dir(str(exp_dir))
    logger.add_text_output(text_log_file)
    logger.add_tabular_output(tabular_log_file)
    logger.push_prefix(f"{name}_{run_ID}")

    yield

    logger.remove_tabular_output(tabular_log_file)
    logger.remove_text_output(text_log_file)
    logger.pop_prefix()
