"""
Custom exception classes.
"""


class IrisAppException(Exception):
    """Base exception for iris application."""
    pass


class ModelLoadError(IrisAppException):
    """Raised when model loading fails."""
    pass


class PredictionError(IrisAppException):
    """Raised when prediction fails."""
    pass


class ValidationError(IrisAppException):
    """Raised when input validation fails."""
    pass


class ConfigurationError(IrisAppException):
    """Raised when configuration is invalid."""
    pass
