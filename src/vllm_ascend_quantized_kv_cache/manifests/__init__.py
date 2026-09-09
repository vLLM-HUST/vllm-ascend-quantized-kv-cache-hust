# SPDX-License-Identifier: Apache-2.0
"""Manifest package for the vLLM-HUST Extension Manager.

The ``vllm_hust.extension_bundles`` entry point in ``pyproject.toml`` points
at this module. The manager reads the static ``vllm-hust-extension-v0.2.json``
descriptor shipped here as package data; discovery must never import the
implementation modules, so this package intentionally stays empty.
"""
