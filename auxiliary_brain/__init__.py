"""Pure runtime pieces for the Hermes Auxiliary Brain plugin."""

from .config import VALID_MODES, BrainConfig, ConfigError
from .local_api import (
    DEFAULT_ENDPOINTS,
    EndpointProbe,
    LocalAPIError,
    OpenAICompatibleClient,
    discover_endpoint,
    list_models,
    normalize_base_url,
    probe_endpoint,
)
from .store import (
    DATASET_FORMAT_VERSION,
    BrainStore,
    CorrectionRecord,
    EventRecord,
    PredictionRecord,
)
from .tasks import (
    BUILTIN_TASKS,
    DEFAULT_REGISTRY,
    TaskParseError,
    TaskRegistry,
    TaskSpec,
    build_messages,
    extract_json_object,
    get_task,
    list_tasks,
    parse_task_result,
    validate_json,
)

__all__ = [
    "BUILTIN_TASKS",
    "BrainConfig",
    "BrainStore",
    "ConfigError",
    "CorrectionRecord",
    "DEFAULT_ENDPOINTS",
    "DEFAULT_REGISTRY",
    "DATASET_FORMAT_VERSION",
    "EndpointProbe",
    "EventRecord",
    "LocalAPIError",
    "OpenAICompatibleClient",
    "PredictionRecord",
    "TaskParseError",
    "TaskRegistry",
    "TaskSpec",
    "VALID_MODES",
    "build_messages",
    "discover_endpoint",
    "extract_json_object",
    "get_task",
    "list_models",
    "list_tasks",
    "normalize_base_url",
    "parse_task_result",
    "probe_endpoint",
    "validate_json",
]
