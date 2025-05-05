from __future__ import absolute_import

import textwrap
import numpy as np
from sagemaker_inference import content_types, decoder, encoder
from sagemaker_inference.default_inference_handler import DefaultInferenceHandler


class DefaultSkLeanInferenceHandler(DefaultInferenceHandler):
    """
    Custom inference handler for Scikit-learn models.
    """

    def default_model_fn(self, model_dir):
        """
        Loads a model. For scikit-learn, a default function to load a model is not provided.
        Users should provide a customized model_fn() in scrip.
        Args:
            model_dir (str): Directory where the model is stored.
        Returns:
            model: The loaded scikit-learn model.
        """
        raise NotImplementedError(
            textwrap.dedent(
                """
                A default model_fn is not provided for scikit-learn models.
                Please provide a custom model_fn in your script.
                See documentation for model_fn at https://github.com/aws/sagemaker-python-sdk
                """
            )
        )

    def default_input_fn(self, input_data, content_type):
        """Takes request and de-serializes it into a numpy array for prediction"""
        np_array = decoder.decode(input_data, content_type)
        return np_array.astype(np.float32) if content_type == content_types.UTF8_TYPES else np_array

    def default_predict_fn(self, input_data, model):
        """A default predict function that takes the input data and model and returns the prediction"""
        output = model.predict(input_data)
        return output

    def default_output_fn(self, prediction, accept):
        """Takes the prediction and serializes it into a response"""
        return encoder.encode(prediction, accept)
