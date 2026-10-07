"""Small, dependency-free validator for the JSON Schema subset JMR exposes."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any
from urllib.parse import urlparse


class JSONSchemaValidationError(ValueError):
    """Raised when an untrusted model or tool payload violates its schema."""


def validate_json_schema(
    value: Any,
    schema: Mapping[str, Any],
    *,
    path: str = "$",
) -> None:
    """Validate the intentionally small JSON Schema vocabulary used by JMR.

    Supporting the vocabulary locally keeps model output and in-process MCP
    calls behind a real trust boundary without adding a heavyweight runtime
    dependency. Unknown schema keywords are ignored, while malformed schemas
    fail closed.
    """

    if not isinstance(schema, Mapping):
        raise TypeError("JSON schema must be an object")
    if "const" in schema and value != schema["const"]:
        raise JSONSchemaValidationError(f"{path} must equal the declared constant")
    enum = schema.get("enum")
    if enum is not None:
        if not isinstance(enum, Sequence) or isinstance(enum, (str, bytes)):
            raise TypeError("JSON schema enum must be an array")
        if value not in enum:
            raise JSONSchemaValidationError(f"{path} is not an allowed value")

    declared_type = schema.get("type")
    if declared_type is not None:
        types = (
            list(declared_type)
            if isinstance(declared_type, Sequence)
            and not isinstance(declared_type, (str, bytes))
            else [declared_type]
        )
        if not all(isinstance(item, str) for item in types):
            raise TypeError("JSON schema type must be a string or string array")
        if not any(_matches_type(value, item) for item in types):
            raise JSONSchemaValidationError(
                f"{path} must have type {' or '.join(types)}"
            )

    if isinstance(value, Mapping):
        _validate_object(value, schema, path)
    elif isinstance(value, list):
        _validate_array(value, schema, path)
    elif isinstance(value, str):
        _validate_string(value, schema, path)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        _validate_number(value, schema, path)

    for keyword in ("allOf", "anyOf", "oneOf"):
        subschemas = schema.get(keyword)
        if subschemas is None:
            continue
        if not isinstance(subschemas, list) or not all(
            isinstance(item, Mapping) for item in subschemas
        ):
            raise TypeError(f"JSON schema {keyword} must be an array of objects")
        matches = 0
        for item in subschemas:
            try:
                validate_json_schema(value, item, path=path)
            except JSONSchemaValidationError:
                continue
            matches += 1
        if keyword == "allOf" and matches != len(subschemas):
            raise JSONSchemaValidationError(f"{path} does not satisfy allOf")
        if keyword == "anyOf" and matches == 0:
            raise JSONSchemaValidationError(f"{path} does not satisfy anyOf")
        if keyword == "oneOf" and matches != 1:
            raise JSONSchemaValidationError(f"{path} does not satisfy oneOf")


def _matches_type(value: Any, expected: str) -> bool:
    return {
        "null": value is None,
        "object": isinstance(value, Mapping),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
    }.get(expected, False)


def _validate_object(
    value: Mapping[str, Any], schema: Mapping[str, Any], path: str
) -> None:
    if not all(isinstance(key, str) for key in value):
        raise JSONSchemaValidationError(f"{path} must use string object keys")
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        raise TypeError("JSON schema properties must be an object")
    required = schema.get("required", [])
    if not isinstance(required, list) or not all(
        isinstance(item, str) for item in required
    ):
        raise TypeError("JSON schema required must be a string array")
    missing = [name for name in required if name not in value]
    if missing:
        raise JSONSchemaValidationError(
            f"{path} is missing required field(s): {', '.join(missing)}"
        )
    additional = schema.get("additionalProperties", True)
    for key, item in value.items():
        item_schema = properties.get(key)
        if item_schema is None:
            if additional is False:
                raise JSONSchemaValidationError(
                    f"{path} contains unknown field {key!r}"
                )
            if isinstance(additional, Mapping):
                validate_json_schema(item, additional, path=f"{path}.{key}")
            continue
        if not isinstance(item_schema, Mapping):
            raise TypeError("JSON schema property definitions must be objects")
        validate_json_schema(item, item_schema, path=f"{path}.{key}")


def _validate_array(value: list[Any], schema: Mapping[str, Any], path: str) -> None:
    minimum = schema.get("minItems")
    maximum = schema.get("maxItems")
    if isinstance(minimum, int) and len(value) < minimum:
        raise JSONSchemaValidationError(f"{path} has too few items")
    if isinstance(maximum, int) and len(value) > maximum:
        raise JSONSchemaValidationError(f"{path} has too many items")
    if schema.get("uniqueItems") is True:
        fingerprints = [repr(item) for item in value]
        if len(fingerprints) != len(set(fingerprints)):
            raise JSONSchemaValidationError(f"{path} must contain unique items")
    item_schema = schema.get("items")
    if item_schema is not None:
        if not isinstance(item_schema, Mapping):
            raise TypeError("JSON schema items must be an object")
        for index, item in enumerate(value):
            validate_json_schema(item, item_schema, path=f"{path}[{index}]")


def _validate_string(value: str, schema: Mapping[str, Any], path: str) -> None:
    minimum = schema.get("minLength")
    maximum = schema.get("maxLength")
    if isinstance(minimum, int) and len(value) < minimum:
        raise JSONSchemaValidationError(f"{path} is shorter than minLength")
    if isinstance(maximum, int) and len(value) > maximum:
        raise JSONSchemaValidationError(f"{path} is longer than maxLength")
    pattern = schema.get("pattern")
    if isinstance(pattern, str) and re.search(pattern, value) is None:
        raise JSONSchemaValidationError(f"{path} does not match its pattern")
    string_format = schema.get("format")
    try:
        if string_format == "date":
            date.fromisoformat(value)
        elif string_format == "date-time":
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError
        elif string_format == "uri":
            parsed_uri = urlparse(value)
            if not parsed_uri.scheme:
                raise ValueError
    except ValueError as exc:
        raise JSONSchemaValidationError(
            f"{path} is not a valid {string_format}"
        ) from exc


def _validate_number(value: int | float, schema: Mapping[str, Any], path: str) -> None:
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    exclusive_minimum = schema.get("exclusiveMinimum")
    exclusive_maximum = schema.get("exclusiveMaximum")
    if isinstance(minimum, (int, float)) and value < minimum:
        raise JSONSchemaValidationError(f"{path} is below minimum")
    if isinstance(maximum, (int, float)) and value > maximum:
        raise JSONSchemaValidationError(f"{path} is above maximum")
    if isinstance(exclusive_minimum, (int, float)) and value <= exclusive_minimum:
        raise JSONSchemaValidationError(f"{path} is below exclusiveMinimum")
    if isinstance(exclusive_maximum, (int, float)) and value >= exclusive_maximum:
        raise JSONSchemaValidationError(f"{path} is above exclusiveMaximum")


__all__ = ["JSONSchemaValidationError", "validate_json_schema"]
