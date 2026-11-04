# manager/supervisor.py
"""
Per-slot supervisor (phase 5, Task 5, Q21).

The supervisor owns no state of its own. supervise(slot, interval) is a
ticker that asks the slot's consumer for a probe every interval while the
slot is idle, unhealthy or failed. The consumer -- the single writer of the
slot's state -- classifies each probe it runs (SlotState.probe returns
PROBE_HEALTHY, PROBE_LOADING or PROBE_FAILED) and hands the result to
judge, which does the bookkeeping on the slot's Supervision (the
consecutive-failure count, the restart counters, the capped backoff) and
says whether a restart is due. When it is, the consumer restarts the slot's
unit under the slot's swap lock -- or, with supervisor_restart off (the
default), logs one "would restart" warning and counts the restart as
suppressed.

A backend answering 503 is loading its model, not failing: judge exempts a
PROBE_LOADING result while now - grace_started < swap_timeout (Review Focus
3). The grace starts when the slot is made (the manager's own start is a
grace), restarts when the slot leaves idle and when a restart returns; a
failed slot is counted but never restarted -- its recovery is a /swap or a
healthy probe.
"""
from dataclasses import dataclass, field

import asyncio

# The three outcomes SlotState.probe reports and judge understands.
PROBE_HEALTHY = "healthy"
PROBE_LOADING = "loading"
PROBE_FAILED = "failed"


@dataclass
class Supervision:
    """The supervisor's per-slot bookkeeping (Task 5, Q21).

    Written only by the slot's own queue consumer: judge() after every
    probe the consumer runs, and the restart it triggers. The supervisor
    task itself writes nothing (it only asks for probes). restarts counts
    restarts that actually ran; restarts_suppressed counts the due
    restarts the operator has not opted into (supervisor_restart off). The
    two private fields pace the backoff: _last_due is the time.monotonic()
    of the last due restart, _delay the current backoff delay (None means
    the base, config.supervisor_backoff). A healthy probe puts _delay back
    to the base but keeps _last_due, so a due restart always comes at
    least the current delay after the previous one.
    """

    consecutive_failures: int = 0
    restarts: int = 0
    restarts_suppressed: int = 0
    _last_due: float | None = field(default=None, repr=False)
    _delay: float | None = field(default=None, repr=False)


def judge(sup: Supervision, result: str, *, stored_state: str,
          grace_started: float, config, now: float) -> bool:
    """Judge one probe result on the slot's Supervision.

    Pure with respect to the outside world: it reads no clock (now is
    passed in) and touches only `sup` -- never the slot's state. Returns
    True when a restart is due.

    A healthy probe resets the count and puts the backoff delay back to
    its base, but it does not forget when the last due restart was: every
    due restart still comes at least the current delay after the previous
    one, a healthy probe in between or not. A loading probe (HTTP 503)
    while now - grace_started < config.swap_timeout changes nothing: the
    backend is still loading, not failing, and it neither counts nor
    resets. Every other result (a connection failure, any other status, a
    malformed or model-less 200, or a probe that raised and arrived here
    as PROBE_FAILED) counts one failure.

    A restart is due when the count is at least config.supervisor_failures,
    the slot's stored state is `unhealthy` -- a `failed` slot is counted but
    never restarted (its recovery is a /swap or a healthy probe), and no
    other state is ever due a restart -- and the backoff allows it: the
    first restart is always
    allowed, and after that each due restart must come at least the current
    delay after the previous one. The delay starts at
    config.supervisor_backoff, doubles after each due restart, and is capped
    at config.supervisor_backoff_max; a healthy probe puts it back to its
    base, but the next due restart still comes at least that delay after
    the last one. A due restart resets the count -- in observe-only mode
    too, so the backoff paces the `would restart` lines.
    """
    if result == PROBE_HEALTHY:
        sup.consecutive_failures = 0
        # Put the delay back to its base, but keep _last_due: the next due
        # restart still comes at least the (base) delay after the last one.
        sup._delay = None
        return False
    if result == PROBE_LOADING and now - grace_started < config.swap_timeout:
        # Still loading after a restart, a move out of idle, or the
        # manager's own start: neither counts nor resets (Review Focus 3).
        return False
    sup.consecutive_failures += 1
    if (sup.consecutive_failures < config.supervisor_failures
            or stored_state != "unhealthy"):
        return False
    if sup._last_due is not None:
        # Inside the backoff delay since the last due restart. _delay can
        # be None here (back at the base after a healthy probe), which
        # keeps _last_due, so read the base in that case.
        delay = (sup._delay if sup._delay is not None
                 else config.supervisor_backoff)
        if now - sup._last_due < delay:
            return False
    # Due. Record the time, reset the count, and set the delay for the
    # next due restart: the base after a reset, otherwise doubled and
    # capped.
    sup._last_due = now
    if sup._delay is None:
        sup._delay = config.supervisor_backoff
    else:
        sup._delay = min(sup._delay * 2, config.supervisor_backoff_max)
    sup.consecutive_failures = 0
    return True


async def supervise(slot, interval: float) -> None:
    """Ask the slot's consumer for a probe every `interval` seconds, for as
    long as the task lives.

    Probes only the states the supervisor can help with: idle (routine
    health), unhealthy and failed (the states that need watching). A slot
    that is serving or swapping is left to its own work -- the derived
    state excludes `serving` here. It writes nothing: request_probe only
    arms the handshake, and the probe, its judgment and any restart all run
    on the slot's own queue consumer, the single writer of the slot's state
    and supervision. Cancellation ends the loop.
    """
    while True:
        await asyncio.sleep(interval)
        if slot.state in ("idle", "unhealthy", "failed"):
            slot.request_probe()


def supervisor_status(sup: Supervision, config) -> dict:
    """The slot's `supervisor` block for /status: exactly these four keys,
    and nothing else."""
    return {
        "restart_enabled": bool(config.supervisor_restart),
        "restarts": sup.restarts,
        "restarts_suppressed": sup.restarts_suppressed,
        "consecutive_failures": sup.consecutive_failures,
    }
