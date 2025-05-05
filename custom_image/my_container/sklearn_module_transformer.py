from __future__ import absolute_import

from sagemaker_inference.transformer import Transformer

from my_container.default_inference_handler import DefaultInferenceHandler


class SKLearnTransformer(Transformer):
    """
    Custom transformer for Scikit-learn models that wraps model for inference using sagemaker inference toolkit.
    """

    def __init__(self, default_inference_handler=None):
        if default_inference_handler is None:
            default_inference_handler = DefaultInferenceHandler()
        super(SKLearnTransformer, self).__init__(default_inference_handler=default_inference_handler)
