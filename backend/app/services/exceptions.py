"""Domain exceptions surfaced as HTTP errors."""


class ServiceError(Exception):
    status_code = 500


class BadRequestError(ServiceError):
    status_code = 400


class NotFoundError(ServiceError):
    status_code = 404


class UpstreamError(ServiceError):
    status_code = 502
