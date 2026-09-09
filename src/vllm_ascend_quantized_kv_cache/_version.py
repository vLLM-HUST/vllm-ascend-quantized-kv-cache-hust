# SPDX-License-Identifier: Apache-2.0
"""Single source of truth for the distribution version.

The Extension Manager guide requires ``extension_version`` in the static
manifest to stay identical to the distribution version. Keep exactly one
copy of the version here and import it everywhere else.
"""

__version__ = "0.2.0.dev0"
