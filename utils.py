import threading
from datetime import datetime
from time import time as get_time_sec
from termcolor import colored


class Logger(object):
    DEBUG = 4
    INFO = 3
    WARNING = 2
    ERROR = 1
    SHUTDOWN = 0

    def __init__(self, level=10, group=None):
        self._lock = threading.Lock()
        self._group = group if group is not None else ""
        self._level = level

    def __call__(self, *args, color=None):
        """
        打印日志到屏幕或者文件，用来输出调试信息,使用同print
        Args:
            args:元组，同print一样的用法
            color:打印的内容的颜色,'r','g','b'或者'y'
            black:是否把背景切换为黑色，默认为白色背景
        """
        assert color in [None, "r", "g", "b", "y"]
        if self._level >= 0:
            try:
                self._lock.acquire()
                t_s = "[%s %s]" % (datetime.now(), self._group)  # 获得当前时间字符串
                s = ""
                for arg in args:
                    s += str(arg) + " "
                if color is None and self._level >= Logger.DEBUG:
                    print("%s %s" % (t_s, s))
                elif color == "g" and self._level >= Logger.INFO:
                    print(colored("%s %s" % (t_s, s), "green"))
                elif color == "y" and self._level >= Logger.WARNING:
                    print(colored("%s %s" % (t_s, s), "yellow"))
                elif color == "r" and self._level >= Logger.ERROR:
                    print(colored("%s %s" % (t_s, s), "red"))
            except Exception as e:
                print(e)
            finally:
                self._lock.release()

    def set_log_level(self, level):
        self._level = level

    def debug(self, *message):
        self.__call__("DEBUG:", *message, color=None)

    def info(self, *message):
        self.__call__("INFO:", *message, color="g")

    def warning(self, *message):
        self.__call__("WARNING:", *message, color="y")

    def error(self, *message):
        self.__call__("ERROR:", *message, color="r")


def deprecated(
    only_once: bool = False, how: ["warning", "exception"] = "warning", instead=None
):
    """
    装饰器,用来表明一段代码（函数或者类）不推荐使用
    only_once:是否仅仅在第一次调用的时候输出警告
    how:
        warning:在屏幕中输出警告
        exception:抛出异常终止程序
    e.g.1
    >>>@deprecated
    >>>def test(x,y):
    >>>    return x+y
    >>>test(2,3)
    >>>test(2,3)
    >>>test(2,3)

    [2022-08-10 16:12:35.319215 ] WARNING: test is deprecated, please do not use it.
    [2022-08-10 16:12:35.319215 ] WARNING: test is deprecated, please do not use it.
    [2022-08-10 16:12:35.319215 ] WARNING: test is deprecated, please do not use it.

    e.g.2
    >>>@deprecated(how='warning',only_once=True)
    >>>def test(x,y):
    >>>    return x+y
    >>>test(2,3)
    >>>test(2,3)
    >>>test(2,3)

    [2022-08-10 16:12:35.319215 ] WARNING: test is deprecated, please do not use it.

    """

    class DeprecatedError(Exception):
        pass

    if not isinstance(only_once, bool):
        func = only_once

        def wrapper(*argc, **kwargs):
            logger.warning(
                "%s is deprecated, please do not use it." % (func.__qualname__)
            )
            return func(*argc, **kwargs)

        return wrapper

    def wrapper(func):
        def inner(*args, **kwargs):
            if not "__warn_times__" in globals():
                globals()["__warn_times__"] = set()
            warn_times = globals()["__warn_times__"]
            if not func in warn_times or (not only_once):
                msg = None
                if instead:
                    if hasattr(instead, "__qualname__"):
                        msg = "please use %s instead. " % (
                            instead
                            if isinstance(instead, str)
                            else instead.__qualname__
                        )
                    else:
                        msg = "please use %s instead. " % (
                            instead if isinstance(instead, str) else type(instead)
                        )
                else:
                    msg = "please do not use it. "
                if how == "warning":
                    logger.warning("%s is deprecated, %s." % (func.__qualname__, msg))
                else:
                    raise DeprecatedError(
                        "%s is deprecated, %s" % (func.__qualname__, msg)
                    )
            warn_times.add(func)
            return func(*args, **kwargs)

        return inner

    return wrapper


def test_code(only_once: bool = False, how: ["warning", "exception"] = "warning"):
    """
    装饰器,用来表明一段代码（函数或者类）不推荐在生产环境使用
    only_once:是否仅仅在第一次调用的时候输出警告
    how:
        warning:在屏幕中输出警告
        exception:抛出异常终止程序
    e.g.1
    >>>@test_code
    >>>def test(x,y):
    >>>    return x+y
    >>>test(2,3)
    >>>test(2,3)
    >>>test(2,3)
    [2022-08-12 11:14:45.480421 ] WARNING: test is a development code. Do not use it in a production deployment..
    """

    class DeprecatedError(Exception):
        pass

    if not isinstance(only_once, bool):
        func = only_once

        def wrapper(*argc, **kwargs):
            logger.warning(
                "WARNING: %s is a development code. Do not use it in a production deployment."
                % (func.__qualname__)
            )
            return func(*argc, **kwargs)

        return wrapper

    def wrapper(func):
        def inner(*args, **kwargs):
            if not "__warn_times__" in globals():
                globals()["__warn_times__"] = set()
            warn_times = globals()["__warn_times__"]
            if not func in warn_times or (not only_once):
                if how == "warning":
                    logger.warning(
                        "WARNING: '%s' is a development code. Do not use it in a production deployment."
                        % (func.__qualname__)
                    )
                else:
                    raise DeprecatedError(
                        "'%s' is a development code. Do not use it in a production deployment.."
                        % (func.__qualname__)
                    )
            warn_times.add(func)
            return func(*args, **kwargs)

        return inner

    return wrapper


class Timer(object):
    """
    上下文环境管理器，用来记录一段代码的执行总用时
    >>>with TimeTape("test"):
    >>>    for i in range(10000000):
    >>>        pass
    [2022-08-12 11:50:10.615994 ] process 'test' threw an exception after:0.4764 sec
    """

    def __init__(self, message=""):
        """
        得到一个TimeTape对象
        Args:
            message:这段程序的名称，方便区分不同程序
        Returns:
            一个上下文管理对象
        """
        self.message = message
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = get_time_sec()

    def __exit__(self, type, value, trace):
        self.end_time = get_time_sec()
        if type is None:
            logger.debug(
                "process '%s' exited after:%.4f sec."
                % (self.message, self.end_time - self.start_time)
            )
        else:
            logger.error(
                "process '%s' threw an exception and exited after:%.4f sec"
                % (self.message, self.end_time - self.start_time)
            )
        pass

    def __call__(self, message):
        if isinstance(message, str):
            self.message = message
            return self
        else:

            def timer(func):
                """
                装饰器,用来计算一段代码的执行时间
                @timer
                >>>def test(x,y):
                >>>    for i in range(1000000):
                >>>        pass
                >>>    return x+y
                >>>test(1,2)
                [2022-08-12 12:33:00.381650 ] process 'test' exited after:0.0200 sec
                """

                def wrapper(*argc, **kwargs):
                    s = get_time_sec()
                    ret = None
                    try:
                        ret = func(*argc, **kwargs)
                        t = get_time_sec()
                        logger.debug(
                            "process '%s' exited after:%.4f sec"
                            % (func.__qualname__, (t - s))
                        )
                    except Exception as e:
                        t = get_time_sec()
                        logger.error(
                            "process '%s' threw an exception and exited after:%.4f sec"
                            % (func.__qualname__, (t - s))
                        )
                        raise e
                    return ret

                wrapper.__doc__ = func.__doc__
                return wrapper

            return timer(message)


timer = Timer()
logger = Logger()
