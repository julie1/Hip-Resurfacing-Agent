from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
from opentelemetry.sdk.resources import Resource
from dotenv import load_dotenv
import nest_asyncio
import logfire
import base64
import os
import uuid
from dataclasses import dataclass
from typing import Optional, Dict, Any
# CHANGED: added propagate_attributes to imports
from langfuse import Langfuse, get_client, observe, propagate_attributes
from datetime import datetime

load_dotenv()

def scrubbing_callback(match: logfire.ScrubMatch):
    """Preserve the Langfuse session ID."""
    if (
        match.path == ("attributes", "langfuse.session.id")
        and match.pattern_match.group(0) == "session"
    ):
        return match.value

@dataclass
class LangfuseTracer:
    """Helper class to create traces in Langfuse using the v4 API."""
    langfuse: Langfuse

    def create_trace(self, name: str, user_id: Optional[str] = None,
                   session_id: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   input_data: Optional[Dict[str, Any]] = None,
                   external_id: Optional[str] = None) -> str:
        """Create a new trace and return the trace ID."""

        if external_id:
            trace_id = Langfuse.create_trace_id(seed=external_id)
        else:
            trace_id = None

        try:
            # CHANGED: start_as_current_span → start_as_current_observation
            with self.langfuse.start_as_current_observation(
                name=name,
                langfuse_trace_id=trace_id
            ) as span:
                # CHANGED: span.update_trace() decomposed:
                #   - user_id/session_id/metadata → propagate_attributes()
                #   - input → span.set_trace_io()
                with propagate_attributes(
                    user_id=user_id,
                    session_id=session_id,
                    # NOTE: metadata must now be dict[str, str]; values >200 chars are dropped
                    metadata={k: str(v) for k, v in (metadata or {}).items()},
                ):
                    pass  # attributes propagate to this observation and all children

                if input_data:
                    span.set_trace_io(input=input_data)

                actual_trace_id = self.langfuse.get_current_trace_id()
                return actual_trace_id

        except Exception as e:
            print(f"Error creating trace: {e}")
            return uuid.uuid4().hex

    def create_span(self, name: str, input_data: Optional[Dict[str, Any]] = None) -> str:
        """Create a span within current trace context and return the span ID."""
        try:
            # CHANGED: start_as_current_span → start_as_current_observation
            with self.langfuse.start_as_current_observation(name=name) as span:
                if input_data:
                    span.update(input=input_data)

                return self.langfuse.get_current_trace_id()

        except Exception as e:
            print(f"Error creating span: {e}")
            return uuid.uuid4().hex

    def create_generation(self, name: str,
                         input_data: Optional[Dict[str, Any]] = None,
                         model: Optional[str] = None) -> str:
        """Create a generation within current trace context."""
        try:
            # CHANGED: start_as_current_generation → start_as_current_observation with as_type="generation"
            with self.langfuse.start_as_current_observation(
                name=name,
                as_type="generation",
                model=model
            ) as generation:
                if input_data:
                    generation.update(input=input_data)

                return self.langfuse.get_current_trace_id()

        except Exception as e:
            print(f"Error creating generation: {e}")
            return uuid.uuid4().hex

    def update_current_trace(self,
                           output: Optional[Dict[str, Any]] = None,
                           metadata: Optional[Dict[str, Any]] = None,
                           tags: Optional[list] = None):
        """Update the current trace with output, metadata, or tags."""
        try:
            # CHANGED: update_current_trace() decomposed:
            #   - output → set_current_trace_io()
            #   - metadata/tags → propagate_attributes()
            if output is not None:
                self.langfuse.set_current_trace_io(output=output)

            if metadata is not None or tags is not None:
                propagate_kwargs = {}
                if metadata is not None:
                    # NOTE: metadata must now be dict[str, str]
                    propagate_kwargs['metadata'] = {k: str(v) for k, v in metadata.items()}
                if tags is not None:
                    propagate_kwargs['tags'] = tags
                with propagate_attributes(**propagate_kwargs):
                    pass

        except Exception as e:
            print(f"Error updating current trace: {e}")

    def add_score(self, name: str, value: float,
                     comment: Optional[str] = None,
                     trace_id: Optional[str] = None):
            """Add a score to a trace using v4 API."""
            try:
                if trace_id:
                    score_params = {
                        "name": name,
                        "value": value,
                        "trace_id": trace_id,
                        "data_type": "NUMERIC"
                    }
                    if comment:
                        score_params["comment"] = comment

                    self.langfuse.create_score(**score_params)
                else:
                    current_trace_id = self.langfuse.get_current_trace_id()
                    if current_trace_id:
                        score_params = {
                            "name": name,
                            "value": value,
                            "trace_id": current_trace_id,
                            "data_type": "NUMERIC"
                        }
                        if comment:
                            score_params["comment"] = comment

                        self.langfuse.create_score(**score_params)
                    else:
                        print("No current trace to score")

            except Exception as e:
                print(f"Error adding score: {e}")


@dataclass
class LangfuseConfig:
    tracer: trace.Tracer
    langfuse: Langfuse
    langfuse_tracer: LangfuseTracer

def configure_langfuse():
    LANGFUSE_PROJECT_ID = os.getenv("LANGFUSE_PROJECT_ID")
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
        try:
            resource = Resource.create({"service.name": "hip_rag_agent"})
            tracer_provider = TracerProvider(
                resource=resource,
                sampler=TraceIdRatioBased(1.0)
            )

            trace.set_tracer_provider(tracer_provider)

            LANGFUSE_AUTH = base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()
            os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{LANGFUSE_HOST}/api/public/otel"
            os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

            print("OpenTelemetry configured successfully for Langfuse")
        except Exception as e:
            print(f"Warning: Failed to configure OpenTelemetry: {e}")
            print("Continuing with basic configuration...")

    nest_asyncio.apply()

    try:
        logfire.configure(
            service_name='hip_rag_agent',
            send_to_logfire=False,
            scrubbing=logfire.ScrubbingOptions(callback=scrubbing_callback)
        )
        print("Logfire configured successfully")
    except Exception as e:
        print(f"Warning: Failed to configure Logfire: {e}")

    tracer = trace.get_tracer("hip_rag_agent")

    try:
        langfuse_client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
            # CHANGED: tracing_enabled is still valid in v4, no change needed here
            # NOTE: if you want v3-style "export all spans" behavior, add:
            # should_export_span=lambda span: True
            tracing_enabled=True,
            debug=False
        )
        print("Langfuse client initialized successfully")
    except Exception as e:
        print(f"Error initializing Langfuse client: {e}")
        langfuse_client = None

    if langfuse_client:
        langfuse_tracer = LangfuseTracer(langfuse=langfuse_client)
    else:
        langfuse_tracer = None

    return LangfuseConfig(
        tracer=tracer,
        langfuse=langfuse_client,
        langfuse_tracer=langfuse_tracer
    )


# Decorator for easy trace creation
def trace_function(name: Optional[str] = None,
                  user_id: Optional[str] = None,
                  session_id: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None):
    """Decorator to automatically trace a function with Langfuse v4."""
    def decorator(func):
        @observe(name=name or func.__name__)
        def wrapper(*args, **kwargs):
            # CHANGED: update_current_trace() → propagate_attributes() context manager
            if user_id or session_id or metadata:
                propagate_kwargs = {}
                if user_id:
                    propagate_kwargs['user_id'] = user_id
                if session_id:
                    propagate_kwargs['session_id'] = session_id
                if metadata:
                    # NOTE: metadata must now be dict[str, str]
                    propagate_kwargs['metadata'] = {k: str(v) for k, v in metadata.items()}
                with propagate_attributes(**propagate_kwargs):
                    return func(*args, **kwargs)
            return func(*args, **kwargs)
        return wrapper
    return decorator
