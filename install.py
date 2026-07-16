#!/usr/bin/env python3
"""Install this checkout as a Hermes user plugin.

Normal users should prefer Hermes' native Git installer::

    hermes plugins install kortexa-ai/hermes-auxiliary-brain --enable

This script is the convenient path for contributors working from a checkout.
It intentionally uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PLUGIN_NAME = "auxiliary-brain"
RUNTIME_PATHS = (
    "plugin.yaml",
    "__init__.py",
    "auxiliary_brain",
    "SKILL.md",
    "after-install.md",
)


class InstallError(RuntimeError):
    """A recoverable installation problem with a user-facing message."""


def default_hermes_home() -> Path:
    """Match Hermes' platform default when HERMES_HOME is not set."""
    configured = os.environ.get("HERMES_HOME", "").strip()
    if configured:
        return Path(configured).expanduser()
    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
        return base / "hermes"
    return Path.home() / ".hermes"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install this checkout into the active Hermes plugin directory."
    )
    parser.add_argument(
        "--hermes-home",
        type=Path,
        default=None,
        help="Target Hermes home (default: HERMES_HOME or the platform default)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing auxiliary-brain plugin installation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the operations without changing files or config",
    )
    parser.add_argument(
        "--no-enable",
        action="store_true",
        help="Copy the plugin without enabling it in Hermes",
    )
    return parser.parse_args(argv)


def runtime_sources(source_root: Path) -> list[Path]:
    """Return the runtime files present in the checkout, validating essentials."""
    required = (source_root / "plugin.yaml", source_root / "__init__.py")
    missing = [path.name for path in required if not path.is_file()]
    package = source_root / "auxiliary_brain"
    if not package.is_dir():
        missing.append(package.name)
    if missing:
        joined = ", ".join(missing)
        raise InstallError(f"Checkout is incomplete; missing: {joined}")
    return [source_root / name for name in RUNTIME_PATHS if (source_root / name).exists()]


def copy_runtime(source_root: Path, destination: Path, *, force: bool, dry_run: bool) -> None:
    sources = runtime_sources(source_root)
    if destination.exists() and not force:
        raise InstallError(
            f"{destination} already exists. Use --force to replace it, "
            f"or use `hermes plugins update {PLUGIN_NAME}` for a Git installation."
        )

    print(f"Plugin destination: {destination}")
    for source in sources:
        print(f"  copy {source.name}")
    if dry_run:
        return

    plugins_dir = destination.parent
    plugins_dir.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{PLUGIN_NAME}-", dir=plugins_dir))
    try:
        for source in sources:
            target = stage / source.name
            if source.is_dir():
                shutil.copytree(
                    source,
                    target,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
            else:
                shutil.copy2(source, target)
        if destination.exists():
            shutil.rmtree(destination)
        stage.replace(destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def hermes_command() -> list[str] | None:
    """Find a Hermes CLI without assuming how the user installed it."""
    executable = shutil.which("hermes")
    if executable:
        return [executable]

    # A contributor may be running this with the Hermes virtualenv's Python.
    try:
        import importlib.util

        if importlib.util.find_spec("hermes_cli.main") is not None:
            return [sys.executable, "-m", "hermes_cli.main"]
    except (ImportError, ModuleNotFoundError, ValueError):
        pass
    return None


def enable_plugin(hermes_home: Path, *, dry_run: bool) -> bool:
    command = hermes_command()
    display = "hermes plugins enable auxiliary-brain --no-allow-tool-override"
    print(f"Enable command: {display}")
    if dry_run:
        return True
    if command is None:
        print(
            "Warning: the Hermes CLI was not found, so the plugin was copied but not enabled.\n"
            f"Run `{display}` with HERMES_HOME={hermes_home} after installing Hermes.",
            file=sys.stderr,
        )
        return False

    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)
    result = subprocess.run(
        [
            *command,
            "plugins",
            "enable",
            PLUGIN_NAME,
            "--no-allow-tool-override",
        ],
        env=env,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(
            "Warning: the plugin was copied, but Hermes could not enable it.\n"
            f"Run `{display}` manually with HERMES_HOME={hermes_home}.",
            file=sys.stderr,
        )
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_root = Path(__file__).resolve().parent
    hermes_home = (args.hermes_home or default_hermes_home()).expanduser().resolve()
    destination = hermes_home / "plugins" / PLUGIN_NAME

    try:
        copy_runtime(source_root, destination, force=args.force, dry_run=args.dry_run)
    except (InstallError, OSError) as exc:
        print(f"Install failed: {exc}", file=sys.stderr)
        return 1

    enabled = args.no_enable or enable_plugin(hermes_home, dry_run=args.dry_run)
    if args.dry_run:
        print("Dry run complete; no files or config were changed.")
        return 0

    print(f"Installed Hermes Auxiliary Brain in {destination}")
    if args.no_enable:
        print(f"Enable it later with: hermes plugins enable {PLUGIN_NAME}")
    elif enabled:
        print("Plugin enabled. Start a new Hermes session, then run: hermes brain setup --auto")
    return 0 if enabled else 2


if __name__ == "__main__":
    raise SystemExit(main())
