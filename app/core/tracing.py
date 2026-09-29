from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register

from app.core.config import Settings


def setup_tracing(settings: Settings) -> None:
    if not settings.tracing_enabled:
        return
    provider = register(
        endpoint=settings.phoenix_endpoint,
        project_name=settings.phoenix_project,
        batch=True,
        verbose=False,
    )
    LangChainInstrumentor().instrument(tracer_provider=provider)
