"""The reader's side of docs/reference/what-to-run. See _common.py."""

from _common import *  # noqa: F403

# This page is written to the one import (rule 1): its blocks say
# ``import whileai as wai`` and reach the engine one dot down, so the fixture
# binds the same alias the reader would (#780). ``_common.py`` binds ``wai``
# to ``whileai.simulations`` for the pages that still spell it that way, and
# the star import above brings that binding in, so it is restored here.
wai = __import__("whileai")
