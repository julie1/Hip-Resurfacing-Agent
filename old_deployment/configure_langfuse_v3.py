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
from langfuse import Langfuse, get_client, observe
from datetime import datetime

load_dotenv()

def scrubbing_callback(match: logfire.ScrubMatch):
    """Preserve the Langfuse session ID."""
    if (
        match.path == ("attributes", "langfuse.session.id")
        and match.pattern_match.group(0) == "session"
    ):
        # Return the original value to prevent redaction.
        return match.value

@dataclass
class LangfuseTracer:
    """Helper class to create traces in Langfuse using the v3 API."""
    langfuse: Langfuse

    def create_trace(self, name: str, user_id: Optional[str] = None,
                   session_id: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   input_data: Optional[Dict[str, Any]] = None,
                   external_id: Optional[str] = None) -> str:
        """Create a new trace and return the trace ID."""

        # Generate deterministic trace ID if external_id is provided
        if external_id:
            trace_id = Langfuse.create_trace_id(seed=external_id)
        else:
            trace_id = None  # Let Langfuse generate one

        try:
            # Use context manager approach for v3
            with self.langfuse.start_as_current_span(
                name=name,
                langfuse_trace_id=trace_id
            ) as span:
                # Update trace attributes
                span.update_trace(
                    user_id=user_id,
                    session_id=session_id,
                    metadata=metadata or {},
                    input=input_data
                )

                # Get the actual trace ID
                actual_trace_id = self.langfuse.get_current_trace_id()
                return actual_trace_id

        except Exception as e:
            print(f"Error creating trace: {e}")
            # Return a fallback trace ID in proper format (32 char hex)
            return uuid.uuid4().hex

    def create_span(self, name: str, input_data: Optional[Dict[str, Any]] = None) -> str:
        """Create a span within current trace context and return the span ID."""
        try:
            with self.langfuse.start_as_current_span(name=name) as span:
                if input_data:
                    span.update(input=input_data)

                # Return the span's trace ID for consistency
                return self.langfuse.get_current_trace_id()

        except Exception as e:
            print(f"Error creating span: {e}")
            return uuid.uuid4().hex

    def create_generation(self, name: str,
                         input_data: Optional[Dict[str, Any]] = None,
                         model: Optional[str] = None) -> str:
        """Create a generation within current trace context."""
        try:
            with self.langfuse.start_as_current_generation(
                name=name,
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
            update_params = {}
            if output is not None:
                update_params['output'] = output
            if metadata is not None:
                update_params['metadata'] = metadata
            if tags is not None:
                update_params['tags'] = tags

            self.langfuse.update_current_trace(**update_params)

        except Exception as e:
            print(f"Error updating current trace: {e}")



    def add_score(self, name: str, value: float,
                     comment: Optional[str] = None,
                     trace_id: Optional[str] = None):
            """Add a score to a trace using v3 API."""
            try:
                if trace_id:
                    # Score a specific trace using v3 API
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
                    # Score the current trace context using v3 API
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

# Configure Langfuse for agent observability
def configure_langfuse():
    LANGFUSE_PROJECT_ID = os.getenv("LANGFUSE_PROJECT_ID")
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    # Initialize proper OpenTelemetry TracerProvider with sampler
    if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
        try:
            # Create a proper TracerProvider with sampler
            resource = Resource.create({"service.name": "hip_rag_agent"})
            tracer_provider = TracerProvider(
                resource=resource,
                sampler=TraceIdRatioBased(1.0)  # Sample all traces
            )

            # Set the tracer provider
            trace.set_tracer_provider(tracer_provider)

            # Configure OpenTelemetry exporter for Langfuse
            LANGFUSE_AUTH = base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()
            os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{LANGFUSE_HOST}/api/public/otel"
            os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

            print("OpenTelemetry configured successfully for Langfuse")
        except Exception as e:
            print(f"Warning: Failed to configure OpenTelemetry: {e}")
            print("Continuing with basic configuration...")

    # Configure Logfire to work with Langfuse
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

    # Get tracer - use the properly configured one if available
    tracer = trace.get_tracer("hip_rag_agent")

    # Initialize Langfuse client with v3 parameters
    try:
        langfuse_client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
            tracing_enabled=True,  # v3: renamed from 'enabled'
            debug=False  # Set to True for debugging
        )
        print("Langfuse client initialized successfully")
    except Exception as e:
        print(f"Error initializing Langfuse client: {e}")
        # Create a dummy client to prevent crashes
        langfuse_client = None

    # Create the helper tracer
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
    """Decorator to automatically trace a function with Langfuse v3."""
    def decorator(func):
        @observe(name=name or func.__name__)
        def wrapper(*args, **kwargs):
            # Get the global client and update trace attributes
            client = get_client()
            if client and (user_id or session_id or metadata):
                client.update_current_trace(
                    user_id=user_id,
                    session_id=session_id,
                    metadata=metadata
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator
