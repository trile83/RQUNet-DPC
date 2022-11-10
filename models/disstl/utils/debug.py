import logging
from functools import wraps

import pendulum

_logger = logging.getLogger(__name__)


def timing(start_msg=None, end_msg=None, print_args=False, logger=None, log_level=logging.INFO):
    """
    Decorator that prints human-readable timing information at end of function execution
    :param start_msg:   An optional message to print for function executes
    :param end_msg:     An optional message to print after function execution, timing and args appended with a comma
    :param print_args:  Print the function call values, defaults to False
    :param logger:      A logger to user to print messages, defaults to its own
    :param log_level:   Log level to use when printing messages, timing info not computed if level is not enabled
    :return: decorator function
    """
    logger = logger or _logger

    def decorator(f):
        @wraps(f)
        def wrap(*args, **kw):
            if logger.isEnabledFor(log_level):
                if print_args:
                    args_str = f", args:[{args!r}, {kw!r}]"
                else:
                    args_str = ""

                if start_msg:
                    logger.log(log_level, start_msg + args_str)

                start = pendulum.now()
                result = f(*args, **kw)
                in_words = (pendulum.now() - start).in_words()

                if end_msg:
                    msg = f"{end_msg}, took: {in_words}{args_str}"
                else:
                    msg = f"{f.__name__!r} took: {in_words}{args_str}"

                logger.log(log_level, msg)
            else:
                result = f(*args, **kw)
            return result

        return wrap

    return decorator
