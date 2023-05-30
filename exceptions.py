from pydantic.error_wrappers import ValidationError


class ModelNotFittedError(Exception):
    def __init__(self, message="Prediction cannot be made. Regressor is not fitted yet"):
        self.message = message
        super().__init__(self.message)


class IncorrectSamplesShapeInDatasetError(ValidationError):
    def __init__(self, message="Cannot create Dataset from Samples with different features length"):
        self.message = message
        super().__init__(self.message)


class InvalidNNeighborsError(Exception):
    def __init__(self, n_neighbors: int, message="Invalid n_neighbors={} specified"):
        self.message = message.format(n_neighbors)
        super().__init__(self.message)


class EnsembleModelFitWithoutComponentRegressorsRegisteredError(Exception):
    def __init__(self,  message="Cannot fit Ensemble model that has no component regressors registered"):
        self.message = message
        super().__init__(self.message)


class InferenceSampleHasUnexpectedShapeError(Exception):
    def __init__(
            self,
            expected_sample_shape: tuple[int, ],
            inference_sample_shape: tuple[int, ],
            message="Inference sample has unexpected shape: {}. Expected shape: {}"
    ):
        self.expected_sample_shape = expected_sample_shape
        self.inference_sample_shape = inference_sample_shape
        self.message = message.format(inference_sample_shape, expected_sample_shape)
        super().__init__(self.message)
