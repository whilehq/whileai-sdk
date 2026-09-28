"""The 02 procedure, plus a read_file description that makes large reads the default.

The stock description says reads default to 100 lines; the pilot agent then
walked files in slices. A prompt suffix may not beat the tool's own text, so
this candidate changes the tool text too.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from deepagents import HarnessProfile
from deepagents.middleware.filesystem import READ_FILE_TOOL_DESCRIPTION

_spec = spec_from_file_location("_c02", Path(__file__).with_name("02_review_procedure.py"))
_c02 = module_from_spec(_spec)
_spec.loader.exec_module(_c02)

READ = READ_FILE_TOOL_DESCRIPTION.replace(
    "By default, it reads up to 100 lines starting from the beginning of the file",
    "Pass limit=300 or more: reading a file once in a large window is cheaper than many small reads",
)

PROFILE = HarnessProfile(
    excluded_tools=_c02.PROFILE.excluded_tools,
    system_prompt_suffix=_c02.SUFFIX,
    tool_description_overrides={"read_file": READ},
)
