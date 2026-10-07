"""Device configuration shared by direct entrypoints and the Spark launcher."""
import os
from pathlib import Path
import tomllib

_config = Path(__file__).resolve().parents[3] / 'config.local.toml'
_local = tomllib.loads(_config.read_text()).get('thor', {}) if _config.exists() else {}
HOST = os.environ.get('COSMOS_THOR_HOST', _local.get('host', 'thor'))
WORKSPACE = os.environ.get('COSMOS_THOR_WORKSPACE', _local.get('workspace', str(Path.home() / 'cosmos_ws')))
CODE = os.environ.get('COSMOS_THOR_CODE', WORKSPACE + '/deploy/current/thor')
PYTHON = os.environ.get('COSMOS_THOR_PYTHON', WORKSPACE + '/.venv/bin/python')
URI = os.environ.get('COSMOS_THOR_URI', _local.get('uri', 'ws://thor:8000'))
