"""Monotonic operator-timeout, VMM renewal and stream-stall state."""
from __future__ import annotations

from dataclasses import dataclass, field
import threading


@dataclass(frozen=True)
class OperatorTimeoutConfig:
    """User-facing capture limit; independent of the firmware stream lease."""

    enabled: bool = True
    duration_s: float = 300.0
    action: str = "Ask"

    def validate(self) -> None:
        if self.enabled and self.duration_s <= 0:
            raise ValueError("Capture timeout duration must be greater than zero.")
        if self.action not in ("Ask", "Continue same duration", "Continue indefinitely", "Stop"):
            raise ValueError(f"Unsupported capture-timeout action: {self.action}")


@dataclass
class LiveSessionState:
    """Thread-safe scheduler state for a live acquisition session.

    Times are monotonic-clock values rather than wall-clock timestamps, so a
    system clock adjustment cannot prematurely expire or extend a capture.
    ``due_actions`` only reports work; the GUI/worker remains responsible for
    carrying out the serial command or prompting the operator.
    """

    operator: OperatorTimeoutConfig
    vmm_start_timeout_ms: int = 300_000
    vmm_renew_fraction: float = 0.8
    max_recovery_attempts: int = 3
    started_monotonic: float = 0.0
    operator_deadline: float | None = None
    next_vmm_renewal_monotonic: float = 0.0
    last_sample_block_monotonic: float = 0.0
    next_recovery_monotonic: float = 0.0
    stream_renewals: int = 0
    timeout_extensions: int = 0
    recovery_attempts: int = 0
    operator_indefinite: bool = False
    timeout_prompt_pending: bool = False
    stalled: bool = False
    recovery_failed: bool = False
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def start(self, now: float) -> None:
        """Reset all deadlines and counters for a session beginning at ``now``."""
        self.operator.validate()
        if self.vmm_start_timeout_ms <= 0:
            raise ValueError("VMM START timeout must be finite and greater than zero.")
        if not 0.1 <= self.vmm_renew_fraction < 1.0:
            raise ValueError("VMM renewal fraction must be between 0.1 and 1.0.")
        with self._lock:
            self.started_monotonic = now
            self.last_sample_block_monotonic = now
            self.next_recovery_monotonic = now
            self.next_vmm_renewal_monotonic = now + self._renew_interval_s()
            self.operator_indefinite = (not self.operator.enabled
                                        or self.operator.action == "Continue indefinitely")
            self.operator_deadline = (None if self.operator_indefinite
                                      else now + self.operator.duration_s)
            self.stream_renewals = 0
            self.timeout_extensions = 0
            self.recovery_attempts = 0
            self.timeout_prompt_pending = False
            self.stalled = False
            self.recovery_failed = False

    def _renew_interval_s(self) -> float:
        """Renew before the finite firmware lease reaches its hard deadline."""
        return self.vmm_start_timeout_ms / 1000.0 * self.vmm_renew_fraction

    def record_sample(self, now: float) -> bool:
        """Record stream progress and report whether this ends a stalled state."""
        with self._lock:
            recovered = self.stalled
            self.last_sample_block_monotonic = now
            self.stalled = False
            self.recovery_failed = False
            return recovered

    def due_actions(self, now: float, stall_after_s: float) -> list[str]:
        """Return scheduler actions that became due at ``now``.

        A due condition is advanced or latched before returning, which makes
        repeated polling idempotent until the next deadline.
        """
        actions: list[str] = []
        with self._lock:
            if now >= self.next_vmm_renewal_monotonic:
                actions.append("renew_vmm")
                # Reserve the next slot now to avoid repeatedly queueing renewal.
                self.next_vmm_renewal_monotonic = now + self._renew_interval_s()

            if (self.operator_deadline is not None and now >= self.operator_deadline
                    and not self.timeout_prompt_pending):
                if self.operator.action == "Ask":
                    self.timeout_prompt_pending = True
                    actions.append("ask_operator")
                elif self.operator.action == "Continue same duration":
                    self.operator_deadline = now + self.operator.duration_s
                    self.timeout_extensions += 1
                    actions.append("operator_extended")
                elif self.operator.action == "Stop":
                    self.operator_deadline = None
                    actions.append("stop")

            age = now - self.last_sample_block_monotonic
            if age >= stall_after_s and now >= self.next_recovery_monotonic:
                self.stalled = True
                if self.recovery_attempts < self.max_recovery_attempts:
                    self.recovery_attempts += 1
                    self.next_recovery_monotonic = now + stall_after_s
                    actions.append("recover_stream")
                elif not self.recovery_failed:
                    self.recovery_failed = True
                    actions.append("recovery_failed")
        return actions

    def mark_vmm_renewal(self, now: float) -> None:
        """Record a successfully issued lease renewal and restart its timer."""
        with self._lock:
            self.stream_renewals += 1
            self.next_vmm_renewal_monotonic = now + self._renew_interval_s()

    def continue_same_duration(self, now: float) -> None:
        """Resolve an operator prompt by starting another equal capture period."""
        with self._lock:
            self.timeout_prompt_pending = False
            self.operator_deadline = now + self.operator.duration_s
            self.timeout_extensions += 1

    def continue_indefinitely(self) -> None:
        """Resolve an operator prompt by disabling future operator deadlines."""
        with self._lock:
            self.timeout_prompt_pending = False
            self.operator_indefinite = True
            self.operator_deadline = None

    def cancel_prompt_for_stop(self) -> None:
        """Resolve an operator prompt when capture is being stopped."""
        with self._lock:
            self.timeout_prompt_pending = False
            self.operator_deadline = None

    def snapshot(self, now: float) -> dict[str, float | int | bool | None]:
        """Return a consistent diagnostics snapshot without exposing mutable state."""
        with self._lock:
            return {
                "elapsed_s": max(0.0, now - self.started_monotonic),
                "operator_remaining_s": (None if self.operator_deadline is None
                                          else max(0.0, self.operator_deadline - now)),
                "operator_indefinite": self.operator_indefinite,
                "vmm_remaining_s": max(0.0, self.next_vmm_renewal_monotonic - now),
                "last_block_age_s": max(0.0, now - self.last_sample_block_monotonic),
                "stream_renewals": self.stream_renewals,
                "timeout_extensions": self.timeout_extensions,
                "recovery_attempts": self.recovery_attempts,
                "stalled": self.stalled,
                "recovery_failed": self.recovery_failed,
            }
