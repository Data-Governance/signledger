"""Core ledger implementation for SignLedger."""

import asyncio
import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable, Union, Iterator
from collections import OrderedDict
import logging

from pydantic import BaseModel, Field, field_validator

from .exceptions import IntegrityError, ValidationError, StorageError
from ..crypto.hashing import HashChain
from ..backends.base import StorageBackend, InMemoryBackend

logger = logging.getLogger(__name__)


class Entry(BaseModel):
    """Immutable ledger entry."""

    model_config = {"frozen": True}  # Make entries immutable after creation

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: Dict[str, Any]
    hash: Optional[str] = None
    previous_hash: Optional[str] = None
    signature: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    nonce: int = 0
    # Monotonic, strictly-increasing per-row sequence assigned by the storage
    # backend (Postgres GENERATED ALWAYS AS IDENTITY). It is NOT supplied by the
    # caller on append and is NOT part of the signed hash input (see to_dict --
    # seq_no is deliberately excluded so that adding this column does not change
    # any previously computed hash and all existing entries stay verifiable).
    # It exists purely for deterministic ordering and tail-truncation detection.
    # None until the row has been persisted and read back.
    seq_no: Optional[int] = None

    @field_validator("data")
    @classmethod
    def validate_data(cls, v):
        """Ensure data is not empty."""
        if not v:
            raise ValueError("Entry data cannot be empty")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary.

        NOTE: seq_no is intentionally NOT included here. to_dict() feeds the
        hash chain (HashChain.calculate_hash hashes to_dict() minus hash and
        signature). Keeping seq_no out of this dict is the deliberate, reversible
        ARCH decision that lets us add the monotonic sequence column WITHOUT
        invalidating any entry hashed before the column existed. Tail-truncation
        is detected via the seq stream / checkpoint, not via the hash chain.
        """
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "hash": self.hash,
            "previous_hash": self.previous_hash,
            "signature": self.signature,
            "metadata": self.metadata,
            "nonce": self.nonce,
        }
    
    def to_json(self) -> str:
        """Convert entry to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entry":
        """Create entry from dictionary."""
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class LedgerStats(BaseModel):
    """Ledger statistics."""
    
    total_entries: int = 0
    first_entry_time: Optional[datetime] = None
    last_entry_time: Optional[datetime] = None
    total_size_bytes: int = 0
    hash_algorithm: str = "sha256"
    integrity_verified: bool = True
    last_verification_time: Optional[datetime] = None


class Ledger:
    """Main ledger class for SignLedger."""
    
    def __init__(
        self,
        backend: Optional[StorageBackend] = None,
        hash_algorithm: str = "sha256",
        enable_signatures: bool = False,
        auto_verify: bool = True,
        verify_interval: int = 3600,  # seconds
        max_entries_memory: int = 1000,
    ):
        self.backend = backend or InMemoryBackend()
        self.hash_chain = HashChain(algorithm=hash_algorithm)
        self.enable_signatures = enable_signatures
        self.auto_verify = auto_verify
        self.verify_interval = verify_interval
        self.max_entries_memory = max_entries_memory
        
        # Thread safety
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        
        # Cache recent entries
        self._cache = OrderedDict()
        self._cache_size = 0
        
        # Verification thread
        self._verify_thread: Optional[threading.Thread] = None
        self._stop_verify = threading.Event()
        
        # Subscribers
        self._subscribers: List[Callable[[Entry], None]] = []
        
        # Initialize
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the ledger."""
        # Start verification thread if enabled
        if self.auto_verify:
            self._start_verification_thread()
        
        # Load latest entries into cache
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load recent entries into cache."""
        try:
            entries = list(self.backend.get_entries(limit=self.max_entries_memory))
            for entry in reversed(entries):  # Add in chronological order
                self._add_to_cache(entry)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    
    def _add_to_cache(self, entry: Entry) -> None:
        """Add entry to cache with size limit."""
        self._cache[entry.id] = entry
        self._cache_size += len(entry.to_json())
        
        # Remove oldest entries if cache is too large
        while len(self._cache) > self.max_entries_memory:
            removed_id, removed_entry = self._cache.popitem(last=False)
            self._cache_size -= len(removed_entry.to_json())
    
    def _start_verification_thread(self) -> None:
        """Start background verification thread."""
        if self._verify_thread and self._verify_thread.is_alive():
            return
        
        self._stop_verify.clear()
        self._verify_thread = threading.Thread(
            target=self._verification_loop,
            daemon=True
        )
        self._verify_thread.start()
    
    def _verification_loop(self) -> None:
        """Background verification loop."""
        while not self._stop_verify.wait(self.verify_interval):
            try:
                logger.info("Running scheduled integrity verification...")
                self.verify_integrity()
            except Exception as e:
                logger.error(f"Scheduled verification failed: {e}")
    
    def append(
        self,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        sign: bool = False,
        signer: Optional[Callable[[str], str]] = None,
    ) -> Entry:
        """Append a new entry to the ledger.
        
        Args:
            data: Entry data to append
            metadata: Optional metadata
            sign: Whether to sign the entry
            signer: Optional signing function
            
        Returns:
            The created entry
            
        Raises:
            ValidationError: If entry validation fails
            StorageError: If storage operation fails
        """
        with self._write_lock:
            build_failures: List[BaseException] = []

            def build_entry(previous_hash: Optional[str]) -> Entry:
                try:
                    return _build(previous_hash)
                except BaseException as exc:
                    build_failures.append(exc)
                    raise

            def _build(previous_hash: Optional[str]) -> Entry:
                # Create entry
                entry = Entry(
                    data=data,
                    metadata=metadata or {},
                    previous_hash=previous_hash,
                )

                # Calculate hash
                entry_dict = entry.to_dict()
                entry_dict["hash"] = self.hash_chain.calculate_hash(entry)

                # Sign if requested or if signatures are enabled
                if sign or self.enable_signatures:
                    if not signer:
                        raise ValidationError("Signer function required for signing")
                    entry_dict["signature"] = signer(entry_dict["hash"])

                return Entry.from_dict(entry_dict)

            # Read the tip, build and store in one backend step. The backend
            # makes this atomic across processes where it can (PostgreSQL:
            # advisory lock + one transaction); self._write_lock only
            # serialises threads of this process.
            try:
                final_entry = self.backend.append_chained(build_entry)
            except Exception as e:
                if build_failures and e is build_failures[-1]:
                    # Hashing or signing failed, not storage: surface as before.
                    raise
                raise StorageError(f"Failed to append entry: {e}", "append", self.backend.name)

            # Update cache
            with self._lock:
                self._add_to_cache(final_entry)
            
            # Notify subscribers
            self._notify_subscribers(final_entry)
            
            return final_entry
    
    def get_entry(self, entry_id: str) -> Optional[Entry]:
        """Get entry by ID.
        
        Args:
            entry_id: Entry ID
            
        Returns:
            Entry if found, None otherwise
        """
        # Check cache first
        with self._lock:
            if entry_id in self._cache:
                return self._cache[entry_id]
        
        # Fetch from backend
        try:
            return self.backend.get_entry(entry_id)
        except Exception as e:
            logger.error(f"Failed to get entry {entry_id}: {e}")
            return None
    
    def get_entries(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> Iterator[Entry]:
        """Get entries within time range.
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum number of entries
            offset: Number of entries to skip
            
        Yields:
            Matching entries
        """
        return self.backend.get_entries(
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            offset=offset,
        )
    
    def query(
        self,
        filter_func: Callable[[Entry], bool],
        limit: Optional[int] = None,
    ) -> List[Entry]:
        """Query entries with custom filter.
        
        Args:
            filter_func: Function to filter entries
            limit: Maximum number of results
            
        Returns:
            List of matching entries
        """
        results = []
        for entry in self.get_entries():
            if filter_func(entry):
                results.append(entry)
                if limit and len(results) >= limit:
                    break
        return results
    
    def verify_integrity(
        self,
        start_entry: Optional[str] = None,
        end_entry: Optional[str] = None,
    ) -> bool:
        """Verify ledger integrity.
        
        Args:
            start_entry: Start from this entry ID (None = beginning)
            end_entry: End at this entry ID (None = end)
            
        Returns:
            True if integrity is valid
            
        Raises:
            IntegrityError: If integrity check fails
        """
        logger.info("Starting integrity verification...")

        previous_hash = None
        previous_seq: Optional[int] = None
        entry_count = 0

        # Get entry iterator (backend yields in strict seq order).
        entries = self.backend.get_entries()

        # Skip to start entry if specified
        if start_entry:
            for entry in entries:
                if entry.id == start_entry:
                    break
                previous_hash = entry.hash
                previous_seq = getattr(entry, "seq_no", None)

        # Verify each entry
        for entry in entries:
            entry_count += 1

            # Verify previous hash link
            if entry.previous_hash != previous_hash:
                raise IntegrityError(
                    f"Hash chain broken at entry {entry.id}",
                    entry_id=entry.id,
                    expected_hash=previous_hash,
                    actual_hash=entry.previous_hash,
                )

            # Verify entry hash
            calculated_hash = self.hash_chain.calculate_hash(entry)
            if entry.hash != calculated_hash:
                raise IntegrityError(
                    f"Invalid hash for entry {entry.id}",
                    entry_id=entry.id,
                    expected_hash=calculated_hash,
                    actual_hash=entry.hash,
                )

            # Verify the monotonic sequence is strictly increasing. A duplicate
            # or out-of-order seq means rows were reordered or re-inserted; a
            # backwards step is impossible under GENERATED ALWAYS AS IDENTITY, so
            # observing one is itself evidence of tampering.
            current_seq = getattr(entry, "seq_no", None)
            if current_seq is not None and previous_seq is not None:
                if current_seq <= previous_seq:
                    raise IntegrityError(
                        f"Sequence not strictly increasing at entry {entry.id} "
                        f"(seq {current_seq} <= previous {previous_seq})",
                        entry_id=entry.id,
                    )
            if current_seq is not None:
                previous_seq = current_seq

            # Update for next iteration
            previous_hash = entry.hash

            # Stop at end entry if specified
            if end_entry and entry.id == end_entry:
                break

        # Whole-ledger scan only: detect tail-truncation. The hash chain cannot
        # see deletion of the NEWEST rows (the surviving tail is still internally
        # consistent), so we cross-check the seq stream and the latest persisted
        # checkpoint. Skipped when a bounded sub-range was requested.
        if not start_entry and not end_entry:
            self._verify_no_tail_truncation(entry_count)

        logger.info(f"Integrity verification complete. Verified {entry_count} entries.")
        return True

    def _verify_no_tail_truncation(self, verified_count: int) -> None:
        """Detect deletion of the newest rows using the seq stream + checkpoint.

        Two independent signals, used when the backend exposes them:

        1. Seq stream: for a contiguous append-only ledger,
           (max_seq - min_seq + 1) must equal the row count. A gap means rows
           were removed from the MIDDLE or the TAIL.

        2. Persisted checkpoint: if the latest stored checkpoint recorded a
           higher total_entries or max_seq than we now observe, the newest rows
           were deleted (tail-truncation) -- which a pure seq-contiguity check
           cannot catch if the truncated tail leaves a still-contiguous prefix.

        Raises IntegrityError on either signal. No-ops for backends (e.g.
        InMemoryBackend) that do not expose these helpers.
        """
        summary = None
        if hasattr(self.backend, "get_seq_summary"):
            try:
                summary = self.backend.get_seq_summary()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"Could not read seq summary: {exc}")

        if summary and summary.get("count"):
            count = summary["count"]
            min_seq = summary["min_seq"]
            max_seq = summary["max_seq"]
            if min_seq is not None and max_seq is not None:
                expected_span = max_seq - min_seq + 1
                if expected_span != count:
                    raise IntegrityError(
                        "Sequence gap detected: rows are missing "
                        f"(span {expected_span} != count {count}); "
                        "the ledger was truncated or rows were deleted.",
                    )

        if hasattr(self.backend, "get_latest_checkpoint"):
            try:
                checkpoint = self.backend.get_latest_checkpoint()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"Could not read latest checkpoint: {exc}")
                checkpoint = None

            if checkpoint:
                stored_total = checkpoint.get("total_entries")
                stored_max_seq = checkpoint.get("max_seq")
                observed_total = (
                    summary["count"] if summary else self.backend.count_entries()
                )
                if stored_total is not None and observed_total < stored_total:
                    raise IntegrityError(
                        "Tail-truncation detected: current entry count "
                        f"{observed_total} is lower than the last signed "
                        f"checkpoint total {stored_total}.",
                    )
                observed_max_seq = summary["max_seq"] if summary else None
                if (
                    stored_max_seq is not None
                    and observed_max_seq is not None
                    and observed_max_seq < stored_max_seq
                ):
                    raise IntegrityError(
                        "Tail-truncation detected: current max sequence "
                        f"{observed_max_seq} is lower than the last signed "
                        f"checkpoint max sequence {stored_max_seq}.",
                    )
    
    def verify_entry(self, entry_id: str) -> bool:
        """Verify a single entry.
        
        Args:
            entry_id: Entry ID to verify
            
        Returns:
            True if entry is valid
        """
        entry = self.get_entry(entry_id)
        if not entry:
            return False
        
        # Verify hash
        calculated_hash = self.hash_chain.calculate_hash(entry)
        if entry.hash != calculated_hash:
            return False
        
        # Verify chain if not first entry
        if entry.previous_hash:
            # Find previous entry
            found_previous = False
            for e in self.get_entries(end_time=entry.timestamp):
                if e.hash == entry.previous_hash:
                    found_previous = True
                    break
            
            if not found_previous:
                return False
        
        return True
    
    def get_stats(self) -> LedgerStats:
        """Get ledger statistics.
        
        Returns:
            Ledger statistics
        """
        stats = LedgerStats(hash_algorithm=self.hash_chain.algorithm)
        
        try:
            # Get total entries
            stats.total_entries = self.backend.count_entries()
            
            # Get time range
            first_entry = self.backend.get_oldest_entry()
            if first_entry:
                stats.first_entry_time = first_entry.timestamp
            
            last_entry = self.backend.get_latest_entry()
            if last_entry:
                stats.last_entry_time = last_entry.timestamp
            
            # Estimate size
            stats.total_size_bytes = self.backend.get_size()
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
        
        return stats
    
    def subscribe(self, callback: Callable[[Entry], None]) -> None:
        """Subscribe to new entries.
        
        Args:
            callback: Function called with new entries
        """
        self._subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable[[Entry], None]) -> None:
        """Unsubscribe from new entries.
        
        Args:
            callback: Callback to remove
        """
        if callback in self._subscribers:
            self._subscribers.remove(callback)
    
    def _notify_subscribers(self, entry: Entry) -> None:
        """Notify subscribers of new entry."""
        for callback in self._subscribers:
            try:
                callback(entry)
            except Exception as e:
                logger.error(f"Subscriber callback failed: {e}")
    
    def close(self) -> None:
        """Close the ledger and cleanup resources."""
        # Stop verification thread
        if self._verify_thread:
            self._stop_verify.set()
            self._verify_thread.join(timeout=5)
        
        # Close backend
        self.backend.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
    
    # Async methods for frameworks like FastAPI
    
    async def append_async(
        self,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        sign: bool = False,
        signer: Optional[Callable[[str], str]] = None,
    ) -> Entry:
        """Async version of append."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.append,
            data,
            metadata,
            sign,
            signer,
        )
    
    async def get_entry_async(self, entry_id: str) -> Optional[Entry]:
        """Async version of get_entry."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_entry, entry_id)
    
    async def verify_integrity_async(
        self,
        start_entry: Optional[str] = None,
        end_entry: Optional[str] = None,
    ) -> bool:
        """Async version of verify_integrity."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.verify_integrity,
            start_entry,
            end_entry,
        )