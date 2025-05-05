from __future__ import absolute_import

from subprocess import CalledProcessError

from retrying import retry
from sagemaker_inference import model_server

from my_container import handler_service

HANDLER_SERVICE = handler_service.__name__


def _retry_if_error(exception):
    return isinstance(exception, CalledProcessError or OSError)


@retry(stop_max_delay=1000 * 30, retrty_on_exception=_retry_if_error)
def _start_model_server():
    model_server.start_model_server(handler_service=HANDLER_SERVICE)


def main():
    _start_model_server()
