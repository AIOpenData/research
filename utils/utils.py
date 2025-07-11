import datetime
import logging
import logging.handlers
import sys
import time

from contextlib import contextmanager


logger = logging.getLogger(__name__)


def init_log(base_level=logging.INFO):
    """
        初始化日志输出配置
    :param base_level: 日志输出级别
    :return:
    """
    _formatter = logging.Formatter("%(asctime)s: %(filename)s: %(lineno)d: %(levelname)s: %(message)s")
    logger = logging.getLogger()
    logger.setLevel(base_level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(_formatter)
    console_handler.setLevel(base_level)
    logger.addHandler(console_handler)


@contextmanager
def timer(name, logger, func=None):
    """
        一个自定义的计时器函数
    :param name: 函数名称
    :param logger: 一个自定义的日志输出器
    :param func: 一个自定义的计时输出函数
    :return:
    """
    t1 = time.time()
    logger.info("Start executing step: {}".format((name)))
    yield
    t2 = time.time()
    duration = t2 - t1
    logger.info("Finished executing step: {}".format((name)))
    if func:
        try:
            func(duration)
        except Exception as e:
            logger.error("failed to execute func {}".format(func.__qualname__))
            logger.error(e)
    else:
        logger.info(
            "Step name: {}; time cost: {}\n".format(name, str(datetime.timedelta(seconds=duration)))
        )
