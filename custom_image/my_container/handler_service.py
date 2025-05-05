from __future__ import absolute_import

import os

from sagemaker_inference import environment
from sagemaker_inference.default_handler_service import DefaultHandlerService

from my_container.default_inference_handler import DefaultSkLeanInferenceHandler
from my_container.sklearn_module_transformer import SKLearnTransformer


class HandlerService(DefaultHandlerService):
    """
    Handler service that is executed by the model server.
    Determines specific default inferences handlers to use based on the type of model being used
    This class extends `DefauultHandlerService` which defines the following:
        - The `handle` method, invoked for all incoming inference requests to the model server.
        - The `initialize` method, invoked at model server startup.
    Based on: https://github.com/awslabs/mxnet-model-server/blob/master/docs/custom_service.md
    """

    def __init__(self):
        self._service

    @staticmethod
    def _user_module_transformer(model_dir=environment.model_dir):
        inference_handler = DefaultSkLeanInferenceHandler()
        return SKLearnTransformer(default_inference_handler=inference_handler)

    def initialize(self, context):
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        code_dir_path = f"{model_dir}/code"
        if "PYTHONPATH" in os.environ:
            os.environ["PYTHONPATH"] = f"{os.environ['PYTHONPATH']}:{code_dir_path}"
        else:
            os.environ["PYTHONPATH"] = code_dir_path

        self._service = self._user_module_transformer(model_dir=model_dir)
        super(HandlerService, self).initialize(context)
