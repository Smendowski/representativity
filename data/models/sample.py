from pydantic import BaseModel, validator


class Sample(BaseModel):
    features: list[float | int]
    representativeness: float | None = None

    @validator("features", pre=True)
    def round_features_precision(cls, features):
        if isinstance(features, list):
            return [round(value, 5) if isinstance(value, float) else value for value in features]
        return features

    class Config:
        validate_assignment = True
        allow_mutation = True
