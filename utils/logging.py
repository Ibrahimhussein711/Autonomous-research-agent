"""
Tiny print-based logging helpers so every agent produces consistent,
readable terminal output without duplicating formatting code.
"""

WIDTH = 60


def section(title: str) -> None:
    print("\n" + "=" * WIDTH)
    print(title)
    print("=" * WIDTH)


def subsection(title: str) -> None:
    print("\n" + title)
    print("-" * WIDTH)


def info(message: str) -> None:
    print(f"\n{message}")


def success(message: str) -> None:
    print(f"\n✅ {message}")


def warn(message: str) -> None:
    print(f"\n⚠️  {message}")


def error(message: str) -> None:
    print(f"\n❌ {message}")


def retrying(attempt: int, err: Exception, kind: str, delay: float) -> None:
    print(
        f"\n🔁 Attempt {attempt} failed ({kind}): {err}\n"
        f"   Retrying in {delay:.1f}s..."
    )
