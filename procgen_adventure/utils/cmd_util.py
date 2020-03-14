def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse

    return argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )


def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument("--alg", help="Algorithm", type=str)
    return parser
