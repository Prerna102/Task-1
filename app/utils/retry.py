import asyncio
import random


async def retry_async(
    operation,
    attempts=3,
    base_delay=1,
):
    """Retries an asynchronous operation."""

    # Tries the operation several times.
    for attempt in range(attempts):

        try:
            # Executes the asynchronous operation
            return await operation()

        except Exception:

            # Re-raises the error after the final attempt.
            if attempt == attempts - 1:
                raise

            # Calculates exponential backoff.
            delay = (
                base_delay * (2 ** attempt)
                + random.uniform(0, 0.5)
            )

            # Waits without blocking the event loop.
            await asyncio.sleep(delay)

            