from __future__ import annotations
import os
import webbrowser
import urllib.parse
import base64
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
# CHANGED: updated import to reference the renamed config file
from configure_langfuse_v4 import LangfuseConfig, LangfuseTracer
# CHANGED: added propagate_attributes import
from langfuse import propagate_attributes

class LangfuseDashboard:
    """
    Helper class to generate and open Langfuse dashboard links with automatic authentication
    """
    def __init__(self, config: LangfuseConfig):
        self.config = config
        self.base_url = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        self.public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        self.secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        self.project_id = os.getenv("LANGFUSE_PROJECT_ID", "")
        if not self.project_id and self.public_key:
            parts = self.public_key.split("_")
            if len(parts) >= 3 and parts[0] == "prj":
                self.project_id = parts[1]
                print(f"Extracted project ID: {self.project_id}")
        if not self.project_id:
            print("WARNING: Could not determine Langfuse project ID. URLs may not work correctly.")

    def get_auth_token(self) -> str:
        if self.public_key and self.secret_key:
            auth_string = f"{self.public_key}:{self.secret_key}"
            return base64.b64encode(auth_string.encode()).decode()
        return ""

    def get_traces_url(self, time_range: str = "1h", user_id: Optional[str] = None,
                      session_id: Optional[str] = None, trace_id: Optional[str] = None,
                      name: Optional[str] = None, tags: Optional[List[str]] = None) -> str:
        if not self.project_id:
            print("ERROR: Project ID not set. Cannot generate Langfuse URL.")
            return f"{self.base_url}/login"
        if trace_id:
            return f"{self.base_url}/project/{self.project_id}/traces/{trace_id}"
        params = {}
        if time_range:
            now = datetime.now()
            if time_range.endswith("h"):
                start_time = now - timedelta(hours=int(time_range[:-1]))
            elif time_range.endswith("d"):
                start_time = now - timedelta(days=int(time_range[:-1]))
            else:
                start_time = now - timedelta(hours=1)
            params["dateRange"] = f"{start_time.isoformat()}Z,{now.isoformat()}Z"
        filters = []
        if user_id: filters.append(f"userId:{user_id}")
        if session_id: filters.append(f"sessionId:{session_id}")
        if name: filters.append(f"name:{name}")
        if tags:
            for tag in tags: filters.append(f"tags:{tag}")
        if filters:
            params["filter"] = " AND ".join(filters)
        base_traces_url = f"{self.base_url}/project/{self.project_id}/traces"
        auth_token = self.get_auth_token()
        if auth_token:
            params["auth_token"] = auth_token
        return f"{base_traces_url}?{urllib.parse.urlencode(params)}"

    def open_dashboard(self, time_range: str = "1h", user_id: Optional[str] = None,
                      session_id: Optional[str] = None, trace_id: Optional[str] = None,
                      name: Optional[str] = None, tags: Optional[List[str]] = None) -> None:
        url = self.get_traces_url(time_range=time_range, user_id=user_id,
                                   session_id=session_id, trace_id=trace_id, name=name, tags=tags)
        if self.public_key and self.secret_key:
            print(f"Opening Langfuse dashboard: {url}")
            try:
                webbrowser.open(url)
                print(f"If the dashboard doesn't open automatically, go to {self.base_url} and log in.")
            except Exception as e:
                print(f"Error opening browser: {e}\nManual URL: {url}")
        else:
            print(f"Cannot authenticate automatically. Opening login page: {url}")
            webbrowser.open(url)

    def add_score_to_trace(self, trace_id: str, name: str, value: float,
                           comment: Optional[str] = None) -> None:
        try:
            processed_comment = comment if comment and comment.strip() else None
            print(f"Adding score to trace {trace_id}: {name}={value}")
            score_params = {"name": name, "value": value, "trace_id": trace_id, "data_type": "NUMERIC"}
            if processed_comment:
                score_params["comment"] = processed_comment
            self.config.langfuse.create_score(**score_params)
            print(f"Added score '{name}': {value} to trace {trace_id}")
        except Exception as e:
            print(f"Error adding score to trace {trace_id}: {e}")
            import traceback; traceback.print_exc()


class FeedbackCollector:
    """
    Helper class to collect and record user feedback in Langfuse v4
    """
    def __init__(self, config: LangfuseConfig, dashboard: LangfuseDashboard):
        self.config = config
        self.dashboard = dashboard
        self.current_traces = {}

    def start_interaction(self, session_id: str, user_id: Optional[str] = None,
                         name: str = "user_interaction", metadata: Optional[Dict[str, Any]] = None,
                         input_data: Optional[Dict[str, Any]] = None, tags: Optional[List[str]] = None,
                         external_id: Optional[str] = None) -> str:
        try:
            if external_id:
                trace_id = self.config.langfuse.create_trace_id(seed=external_id)

            # CHANGED: start_as_current_span -> start_as_current_observation
            with self.config.langfuse.start_as_current_observation(name=name) as span:

                # CHANGED: span.update_trace(**update_params) decomposed into:
                #   propagate_attributes() for user_id/session_id/metadata/tags
                #   span.set_trace_io() for input
                propagate_kwargs = {}
                if user_id: propagate_kwargs['user_id'] = user_id
                if session_id: propagate_kwargs['session_id'] = session_id
                if metadata:
                    # NOTE: v4 requires metadata to be dict[str, str]
                    propagate_kwargs['metadata'] = {k: str(v) for k, v in metadata.items()}
                if tags: propagate_kwargs['tags'] = tags

                if propagate_kwargs:
                    with propagate_attributes(**propagate_kwargs):
                        pass

                if input_data:
                    span.set_trace_io(input=input_data)

                actual_trace_id = self.config.langfuse.get_current_trace_id()
                self.current_traces[session_id] = actual_trace_id
                print(f"Created new trace {actual_trace_id} for session {session_id}")
                return actual_trace_id

        except Exception as e:
            print(f"Error starting interaction trace: {e}")
            import traceback; traceback.print_exc()
            fallback_id = uuid.uuid4().hex
            self.current_traces[session_id] = fallback_id
            return fallback_id

    def register_trace(self, session_id: str, trace_id: str) -> None:
        if not session_id:
            print("Warning: Empty session_id provided to register_trace"); return
        if not trace_id:
            print("Warning: Empty trace_id provided to register_trace"); return
        self.current_traces[session_id] = trace_id
        print(f"Registered trace {trace_id} for session {session_id}")

    def update_current_interaction(self, session_id: str, output: Optional[Dict[str, Any]] = None,
                                   metadata: Optional[Dict[str, Any]] = None,
                                   tags: Optional[List[str]] = None) -> None:
        trace_id = self.current_traces.get(session_id)
        if not trace_id:
            print(f"No active trace found for session {session_id}"); return
        try:
            # Delegates to LangfuseTracer.update_current_trace() in configure_langfuse_v4.py
            self.config.langfuse_tracer.update_current_trace(output=output, metadata=metadata, tags=tags)
            print(f"Updated trace {trace_id} for session {session_id}")
        except Exception as e:
            print(f"Error updating trace {trace_id}: {e}")

    def record_feedback(self, session_id: str, score_name: str, score_value: float,
                       comment: Optional[str] = None, trace_id: Optional[str] = None) -> None:
        target_trace_id = trace_id or self.current_traces.get(session_id)
        if not target_trace_id:
            print(f"No trace ID found for session {session_id}"); return
        processed_comment = comment if comment and comment.strip() else None
        print(f"Recording feedback for trace {target_trace_id}: {score_name}={score_value}")
        try:
            score_params = {"name": score_name, "value": score_value,
                           "trace_id": target_trace_id, "data_type": "NUMERIC"}
            if processed_comment:
                score_params["comment"] = processed_comment
            print(f"Score parameters: {score_params}")
            self.config.langfuse.create_score(**score_params)
            print(f"Successfully added score '{score_name}': {score_value} to trace {target_trace_id}")
        except Exception as e:
            print(f"Error adding score to trace {target_trace_id}: {e}")
            print(f"Project ID: {self.dashboard.project_id}")
            import traceback; traceback.print_exc()
