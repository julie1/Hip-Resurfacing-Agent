from __future__ import annotations
import os
import webbrowser
import urllib.parse
import base64
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from configure_langfuse_v3 import LangfuseConfig, LangfuseTracer

class LangfuseDashboard:
    """
    Helper class to generate and open Langfuse dashboard links with automatic authentication
    """
    def __init__(self, config: LangfuseConfig):
        self.config = config
        self.base_url = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        # Get credentials
        self.public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        self.secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")

        # Extract project ID more reliably
        self.project_id = os.getenv("LANGFUSE_PROJECT_ID", "")
        if not self.project_id and self.public_key:
            # Public keys are typically in format prj_[project_id]_pub_[random]
            parts = self.public_key.split("_")
            if len(parts) >= 3 and parts[0] == "prj":
                self.project_id = parts[1]
                print(f"Extracted project ID: {self.project_id}")

        if not self.project_id:
            print("WARNING: Could not determine Langfuse project ID. URLs may not work correctly.")

    def get_auth_token(self) -> str:
        """Generate authentication token for direct dashboard access"""
        if self.public_key and self.secret_key:
            auth_string = f"{self.public_key}:{self.secret_key}"
            return base64.b64encode(auth_string.encode()).decode()
        return ""

    def get_traces_url(self,
                      time_range: str = "1h",
                      user_id: Optional[str] = None,
                      session_id: Optional[str] = None,
                      trace_id: Optional[str] = None,
                      name: Optional[str] = None,
                      tags: Optional[List[str]] = None) -> str:
        """
        Generate URL to view traces in Langfuse dashboard

        Args:
            time_range: Time range (e.g., "1h", "24h", "7d")
            user_id: Filter by user ID
            session_id: Filter by session ID
            trace_id: Specific trace ID to view
            name: Filter by trace name
            tags: Filter by tags (v3 feature)

        Returns:
            URL to Langfuse dashboard
        """
        if not self.project_id:
            print("ERROR: Project ID not set. Cannot generate Langfuse URL.")
            return f"{self.base_url}/login"  # Fallback to login page

        if trace_id:
            # Direct link to specific trace
            return f"{self.base_url}/project/{self.project_id}/traces/{trace_id}"

        # Build query parameters
        params = {}

        # Calculate time range
        if time_range:
            now = datetime.now()
            if time_range.endswith("h"):
                hours = int(time_range[:-1])
                start_time = now - timedelta(hours=hours)
            elif time_range.endswith("d"):
                days = int(time_range[:-1])
                start_time = now - timedelta(days=days)
            else:
                # Default to 1 hour
                start_time = now - timedelta(hours=1)

            # Format according to Langfuse expectations - ISO format with Z for UTC
            params["dateRange"] = f"{start_time.isoformat()}Z,{now.isoformat()}Z"

        # Add filters
        filters = []
        if user_id:
            filters.append(f"userId:{user_id}")
        if session_id:
            filters.append(f"sessionId:{session_id}")
        if name:
            filters.append(f"name:{name}")
        if tags:
            for tag in tags:
                filters.append(f"tags:{tag}")

        if filters:
            params["filter"] = " AND ".join(filters)

        # Build URL
        query_string = urllib.parse.urlencode(params)
        base_traces_url = f"{self.base_url}/project/{self.project_id}/traces"

        # Add auth token if available
        auth_token = self.get_auth_token()
        if auth_token:
            # Add token as URL parameter - this is a workaround approach
            # Proper implementation would use HTTP headers, but browser security may prevent this
            params["auth_token"] = auth_token
            query_string = urllib.parse.urlencode(params)

        return f"{base_traces_url}?{query_string}"

    def open_dashboard(self,
                      time_range: str = "1h",
                      user_id: Optional[str] = None,
                      session_id: Optional[str] = None,
                      trace_id: Optional[str] = None,
                      name: Optional[str] = None,
                      tags: Optional[List[str]] = None) -> None:
        """Open the Langfuse dashboard in a browser with automatic authentication"""
        url = self.get_traces_url(
            time_range=time_range,
            user_id=user_id,
            session_id=session_id,
            trace_id=trace_id,
            name=name,
            tags=tags
        )

        # If we have credentials, try to create an authenticated session first
        if self.public_key and self.secret_key:
            # Print URL for debugging
            print(f"Opening Langfuse dashboard: {url}")

            try:
                # Try to open the browser with the URL
                webbrowser.open(url)

                # Also print out manual access info in case automatic opening fails
                print(f"If the dashboard doesn't open automatically, you can:")
                print(f"1. Go to {self.base_url}")
                print(f"2. Log in with your Langfuse credentials")
                print(f"3. Navigate to project '{self.project_id}'")
            except Exception as e:
                print(f"Error opening browser: {e}")
                print(f"Manual URL: {url}")
        else:
            print(f"Cannot authenticate automatically. Opening login page: {url}")
            webbrowser.open(url)


    def add_score_to_trace(self,
                              trace_id: str,
                              name: str,
                              value: float,
                              comment: Optional[str] = None) -> None:
            """
            Add a score to a trace using v3 API

            Args:
                trace_id: ID of the trace to score
                name: Name of the score (e.g., "relevance", "accuracy")
                value: Score value (typically 0-1 or 0-100)
                comment: Optional comment explaining the score
            """
            try:
                # Process the comment to ensure it's not None or empty
                processed_comment = comment if comment and comment.strip() else None

                # Debug info
                print(f"Adding score to trace {trace_id}: {name}={value}")
                if processed_comment:
                    print(f"With comment: {processed_comment[:50]}..." if len(processed_comment) > 50 else f"With comment: {processed_comment}")

                # Create parameter dictionary for v3 API
                score_params = {
                    "name": name,
                    "value": value,
                    "trace_id": trace_id,
                    "data_type": "NUMERIC"  # v3 requires explicit data_type
                }

                # Only add comment if it's not empty
                if processed_comment:
                    score_params["comment"] = processed_comment

                # Use v3 API method - create_score instead of score
                self.config.langfuse.create_score(**score_params)

                print(f"Added score '{name}': {value} to trace {trace_id}")
            except Exception as e:
                print(f"Error adding score to trace {trace_id}: {e}")
                import traceback
                traceback.print_exc()



class FeedbackCollector:
    """
    Helper class to collect and record user feedback in Langfuse v3
    """
    def __init__(self, config: LangfuseConfig, dashboard: LangfuseDashboard):
        self.config = config
        self.dashboard = dashboard
        self.current_traces = {}  # Store active trace IDs by session

    def start_interaction(self,
                         session_id: str,
                         user_id: Optional[str] = None,
                         name: str = "user_interaction",
                         metadata: Optional[Dict[str, Any]] = None,
                         input_data: Optional[Dict[str, Any]] = None,
                         tags: Optional[List[str]] = None,
                         external_id: Optional[str] = None) -> str:
        """
        Start a new traced interaction using v3 context managers

        Args:
            session_id: Session identifier
            user_id: User identifier
            name: Trace name
            metadata: Additional metadata
            input_data: Input data for the trace
            tags: Tags for the trace
            external_id: External ID for deterministic trace ID generation

        Returns:
            Trace ID
        """
        try:
            # Generate deterministic trace ID if external_id provided
            if external_id:
                trace_id = self.config.langfuse.create_trace_id(seed=external_id)
            else:
                trace_id = None

            # Start trace using context manager
            with self.config.langfuse.start_as_current_span(
                name=name
            ) as span:
                # Update trace attributes
                update_params = {}
                if user_id:
                    update_params['user_id'] = user_id
                if session_id:
                    update_params['session_id'] = session_id
                if metadata:
                    update_params['metadata'] = metadata
                if input_data:
                    update_params['input'] = input_data
                if tags:
                    update_params['tags'] = tags

                if update_params:
                    span.update_trace(**update_params)

                # Get the actual trace ID
                actual_trace_id = self.config.langfuse.get_current_trace_id()

                # Store the trace ID for this session
                self.current_traces[session_id] = actual_trace_id

                print(f"Created new trace {actual_trace_id} for session {session_id}")
                return actual_trace_id

        except Exception as e:
            print(f"Error starting interaction trace: {e}")
            import traceback
            traceback.print_exc()
            # Return a properly formatted fallback trace ID
            fallback_id = uuid.uuid4().hex
            self.current_traces[session_id] = fallback_id
            return fallback_id

    def register_trace(self, session_id: str, trace_id: str) -> None:
        """
        Register an existing trace ID for a specific session.
        Use this when a trace was created elsewhere and you want to associate it with a session
        for subsequent feedback collection.

        Args:
            session_id: The session identifier to associate the trace with
            trace_id: The existing trace ID to register
        """
        if not session_id:
            print("Warning: Empty session_id provided to register_trace")
            return

        if not trace_id:
            print("Warning: Empty trace_id provided to register_trace")
            return

        # Store the trace ID for this session
        self.current_traces[session_id] = trace_id
        print(f"Registered trace {trace_id} for session {session_id}")

    def update_current_interaction(self,
                                 session_id: str,
                                 output: Optional[Dict[str, Any]] = None,
                                 metadata: Optional[Dict[str, Any]] = None,
                                 tags: Optional[List[str]] = None) -> None:
        """
        Update the current interaction trace with output, metadata, or tags

        Args:
            session_id: Session identifier
            output: Output data to add to the trace
            metadata: Additional metadata to add
            tags: Tags to add to the trace
        """
        trace_id = self.current_traces.get(session_id)
        if not trace_id:
            print(f"No active trace found for session {session_id}")
            return

        try:
            # Use the tracer helper to update the current trace
            self.config.langfuse_tracer.update_current_trace(
                output=output,
                metadata=metadata,
                tags=tags
            )
            print(f"Updated trace {trace_id} for session {session_id}")
        except Exception as e:
            print(f"Error updating trace {trace_id}: {e}")

    def record_feedback(self,
                       session_id: str,
                       score_name: str,
                       score_value: float,
                       comment: Optional[str] = None,
                       trace_id: Optional[str] = None) -> None:
        """
        Record user feedback as a score in Langfuse v3

        Args:
            session_id: Session ID to identify the interaction
            score_name: Type of feedback (e.g. "relevance", "helpfulness")
            score_value: Numeric score (1-5, 0-100, etc.)
            comment: Optional comment explaining the score
            trace_id: Optional explicit trace ID (will use current session trace if not provided)
        """
        # Get trace ID - either provided or from current session
        target_trace_id = trace_id or self.current_traces.get(session_id)

        if not target_trace_id:
            print(f"No trace ID found for session {session_id}")
            return

        # Process the comment to ensure it's not None or empty
        processed_comment = comment if comment and comment.strip() else None

        # Log what we're about to do to help with debugging
        print(f"Recording feedback for trace {target_trace_id}: {score_name}={score_value}")
        if processed_comment:
            print(f"With comment: {processed_comment[:50]}..." if len(processed_comment) > 50 else f"With comment: {processed_comment}")

        # Add score using v3 API
        try:
            # Use v3 create_score method instead of score
            score_params = {
                "name": score_name,
                "value": score_value,
                "trace_id": target_trace_id,
                "data_type": "NUMERIC"  # v3 requires explicit data_type
            }

            # Only add comment if it exists and is not empty
            if processed_comment:
                score_params["comment"] = processed_comment

            # Debug the exact parameters being sent
            print(f"Score parameters: {score_params}")

            # Send the score to Langfuse using v3 API
            self.config.langfuse.create_score(**score_params)

            print(f"Successfully added score '{score_name}': {score_value} to trace {target_trace_id}")
        except Exception as e:
            print(f"Error adding score to trace {target_trace_id}: {e}")
            print(f"Project ID: {self.dashboard.project_id}")
            import traceback
            traceback.print_exc()

    
