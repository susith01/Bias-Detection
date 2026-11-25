"""Supabase PostgreSQL & Storage integration for FairLens AI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from config import CONFIG

logger = logging.getLogger(__name__)

try:
    from supabase import Client, create_client
except ImportError:  # pragma: no cover
    Client = Any  # type: ignore
    create_client = None  # type: ignore


class SupabaseManager:
    """Reusable database & file storage wrapper with safe error handling."""

    def __init__(self, url: Optional[str] = None, key: Optional[str] = None) -> None:
        self.url = (url if url is not None else CONFIG.supabase_url).strip()
        self.key = (key if key is not None else CONFIG.supabase_key).strip()
        self._client: Optional[Any] = None
        self.last_error: Optional[str] = None

    @property
    def is_configured(self) -> bool:
        return bool(self.url and self.key and create_client is not None)

    def get_client(self) -> Optional[Any]:
        if not self.is_configured:
            return None
        if self._client is None:
            try:
                self._client = create_client(self.url, self.key)
            except Exception as exc:
                self.last_error = f"Supabase client initialization failed: {exc}"
                logger.warning(self.last_error)
                return None
        return self._client

    def insert_record(self, table: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Insert a record into Supabase or return a local fallback payload on offline mode."""
        client = self.get_client()
        record = {"id": payload.get("id") or str(uuid4()), **payload}
        if client is None:
            return {"status": "offline_local", "data": record, "error": self.last_error}
        try:
            response = client.table(table).insert(record).execute()
            rows = getattr(response, "data", None) or [record]
            return {"status": "persisted", "data": rows[0], "error": None}
        except Exception as exc:
            self.last_error = f"Insert into '{table}' failed: {exc}"
            logger.error(self.last_error)
            return {"status": "error", "data": record, "error": self.last_error}

    def fetch_records(
        self,
        table: str,
        limit: int = 100,
        order_by: str = "created_at",
        desc: bool = True,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Query records safely with optional equality filters."""
        client = self.get_client()
        if client is None:
            return []
        try:
            query = client.table(table).select("*")
            if filters:
                for col, val in filters.items():
                    query = query.eq(col, val)
            query = query.order(order_by, desc=desc).limit(limit)
            response = query.execute()
            return list(getattr(response, "data", None) or [])
        except Exception as exc:
            self.last_error = f"Fetch from '{table}' failed: {exc}"
            logger.error(self.last_error)
            return []

    def upload_file(self, bucket: str, local_path: Path, remote_path: str) -> Dict[str, Any]:
        """Upload dataset, Joblib model, or PDF report to Supabase Storage."""
        client = self.get_client()
        if client is None:
            return {"status": "offline_local", "path": str(local_path), "error": self.last_error}
        try:
            with Path(local_path).open("rb") as fh:
                content = fh.read()
            client.storage.from_(bucket).upload(
                path=remote_path,
                file=content,
                file_options={"upsert": "true"},
            )
            return {"status": "uploaded", "path": f"{bucket}/{remote_path}", "error": None}
        except Exception as exc:
            self.last_error = f"Storage upload to '{bucket}/{remote_path}' failed: {exc}"
            logger.error(self.last_error)
            return {"status": "error", "path": str(local_path), "error": self.last_error}

    def save_dataset_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return self.insert_record("datasets", metadata)

    def save_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        return self.insert_record("analyses", analysis)

    def save_model_metadata(self, model_meta: Dict[str, Any]) -> Dict[str, Any]:
        return self.insert_record("models", model_meta)

    def save_fairness_result(self, fairness_meta: Dict[str, Any]) -> Dict[str, Any]:
        return self.insert_record("fairness_results", fairness_meta)

    def save_mitigation_result(self, mitigation_meta: Dict[str, Any]) -> Dict[str, Any]:
        return self.insert_record("mitigation_results", mitigation_meta)

    def save_prediction(self, prediction_meta: Dict[str, Any]) -> Dict[str, Any]:
        return self.insert_record("predictions", prediction_meta)

    def save_incoming_record(self, incoming_meta: Dict[str, Any]) -> Dict[str, Any]:
        return self.insert_record("incoming_records", incoming_meta)

    def save_monitoring_batch(self, batch_meta: Dict[str, Any]) -> Dict[str, Any]:
        return self.insert_record("monitoring_batches", batch_meta)

    def save_report_metadata(self, report_meta: Dict[str, Any]) -> Dict[str, Any]:
        return self.insert_record("reports", report_meta)


def get_supabase_manager() -> SupabaseManager:
    """Factory returning a configured SupabaseManager instance."""
    return SupabaseManager()
