"""Extensions for OpenAPI schema"""

import re

from openapi.exceptions import EnumDescriptionError

ENUM_DESCRIPTION_RE = re.compile(r"\w*\*\s`(?P<key>.*)`\s\-\s(?P<description>.*)")


def _iter_described_enums(schema, *, name=None, is_root=True):
    """
    Create an iterator over all enums with descriptions
    """
    if is_root:
        for item_name, item in schema.items():
            yield from _iter_described_enums(item, name=item_name, is_root=False)
    elif isinstance(schema, list):
        for item in schema:
            yield from _iter_described_enums(item, name=name, is_root=is_root)
    elif isinstance(schema, dict):
        if "enum" in schema and "description" in schema:
            yield name, schema

        yield from _iter_described_enums(
            schema.get("properties", []), name=name, is_root=is_root
        )
        yield from _iter_described_enums(
            schema.get("oneOf", []), name=name, is_root=is_root
        )
        yield from _iter_described_enums(
            schema.get("allOf", []), name=name, is_root=is_root
        )
        yield from _iter_described_enums(
            schema.get("anyOf", []), name=name, is_root=is_root
        )


def postprocess_x_enum_descriptions(result, generator, request, public):  # noqa: ARG001
    """
    Take the drf-spectacular generated descriptions and
    puts it into the x-enum-descriptions property.
    """

    # your modifications to the schema in parameter result
    schemas = result.get("components", {}).get("schemas", {})

    for name, schema in _iter_described_enums(schemas):
        lines = schema["description"].splitlines()
        descriptions_by_value = {}
        for line in lines:
            match = ENUM_DESCRIPTION_RE.match(line)
            if match is None:
                continue

            key = match["key"]
            description = match["description"]

            # sometimes there are descriptions for empty values
            # that aren't present in `"enums"`
            if key in schema["enum"]:
                descriptions_by_value[key] = description

        if len(descriptions_by_value.values()) != len(schema["enum"]):
            msg = f"Unable to find descriptions for all enum values: {name}"
            raise EnumDescriptionError(msg)

        if descriptions_by_value:
            schema["x-enum-descriptions"] = [
                descriptions_by_value[value] for value in schema["enum"]
            ]

    return result


def add_channels_routes(result, generator, request, public):  # noqa: ARG001
    """
    Append ai_chatbots AsyncHttpConsumer endpoints to the OpenAPI schema.
    """
    paths = result["paths"]
    paths["/http/recommendation_agent/"] = {
        "post": {
            "operationId": "RecommendationAgentV0",
            "description": "Recommendation agent endpoint via AsyncHttpConsumer",
            "tags": ["Channels"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "message": {
                                    "type": "string",
                                    "description": "The user's message to the AI",
                                },
                                "model": {
                                    "type": "string",
                                    "description": "The LLM model to use",
                                },
                                "temperature": {
                                    "type": "number",
                                    "format": "float",
                                    "description": "The LLM temperature to use",
                                },
                                "instructions": {
                                    "type": "string",
                                    "description": "System prompt (admins only)",
                                },
                                "clear_history": {
                                    "type": "boolean",
                                    "description": "Whether to clear chat history",
                                },
                                "thread_id": {
                                    "type": "string",
                                    "description": "The thread id to use",
                                },
                            },
                            "required": ["message"],
                        }
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "Recommendation Agent stream",
                    "content": {"text/event-stream": {"schema": {"type": "string"}}},
                }
            },
        }
    }
    paths["/http/syllabus_agent/"] = {
        "post": {
            "operationId": "SyllabusAgentV0",
            "description": "Syllabus agent endpoint via AsyncHttpConsumer",
            "tags": ["Channels"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "message": {
                                    "type": "string",
                                    "description": "The user's message to the AI",
                                },
                                "course_id": {
                                    "type": "string",
                                    "description": "The course id",
                                },
                                "collection_name": {
                                    "type": "string",
                                    "description": "Vector embedding collection name",
                                },
                                "model": {
                                    "type": "string",
                                    "description": "The LLM model to use",
                                },
                                "temperature": {
                                    "type": "number",
                                    "format": "float",
                                    "description": "The LLM temperature to use",
                                },
                                "instructions": {
                                    "type": "string",
                                    "description": "System prompt (admins only)",
                                },
                                "clear_history": {
                                    "type": "boolean",
                                    "description": "Whether to clear chat history",
                                },
                                "thread_id": {
                                    "type": "string",
                                    "description": "The thread id to use",
                                },
                            },
                            "required": ["message", "course_id"],
                        }
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "Recommendation Agent stream",
                    "content": {"text/event-stream": {"schema": {"type": "string"}}},
                }
            },
        }
    }
    return result
