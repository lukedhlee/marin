# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501  (the vendored template must stay byte-identical, long lines included)

"""Grug Datakit 09-21 chat format for Snowball SFT stages that start from that checkpoint.

The 09-21 checkpoint (``grug-datakit-sft-20260921``) was trained with a different chat template and
tokenizer from Stage-3. Its ``training_chat_template.jinja`` (sha256 6f55d2ce..., byte-identical to
marin main's ``marin.datakit.chat_template.MARIN_CHAT_TEMPLATE``) is vendored below verbatim and checked
by hash at import. Its tokenizer differs from the Stage-3 one only in ids 128005 / 128011, which are the
special tokens ``<tool_call>`` / ``</tool_call>``.

Rows for these stages carry ``conversations`` = list of ``{role, content, reasoning}`` and a row column
``enable_thinking`` (bool). ``SnowballDatakitChatFormat`` maps them onto what the template reads: a
non-empty ``reasoning`` becomes ``reasoning_content`` (rendered as ``<|start_think|>...<|end_think|>``
inside the ``{% generation %}`` block), and ``enable_thinking`` becomes the per-row template kwarg that
selects the ``Reasoning: /think`` or ``Reasoning: /nothink`` system header. The trained span per
assistant turn is then exactly ``[<|start_think|>{reasoning}<|end_think|>]{content}<|eot_id|>``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from levanter.data._preprocessor import BatchProcessor
from levanter.data.text.formats import ChatLmDatasetFormat, ChatProcessor, LmDatasetFormatBase
from levanter.tokenizers import MarinTokenizer

GRUG_DATAKIT_0921_TOKENIZER_SHA256 = "6a61ca2105a54e9984169b409581f31cc50330d1b47104af13c221b05e462e45"
GRUG_DATAKIT_0921_TRAINING_TEMPLATE_SHA256 = "6f55d2ce5974bbcc5d1636c564cddddc945465edecc56e3a391486268af4d930"
# The serving template shipped as chat_template.jinja differs in one line (absent enable_thinking -> /think).
GRUG_DATAKIT_0921_SERVING_TEMPLATE_SHA256 = "b3f20bc21498407f2815dd1a6c26150192926ccf04f6324ca40a1209bff7d1de"
# ids the 09-21 tokenizer turns into special tokens (reserved_special_token_2 / _3 in the Stage-3 tokenizer)
GRUG_DATAKIT_0921_SPECIAL_IDS = {"<tool_call>": 128005, "</tool_call>": 128011}

GRUG_DATAKIT_0921_TRAINING_TEMPLATE = """{{ bos_token }}
{%- if enable_thinking is defined -%}
  {%- if enable_thinking is sameas true -%}
    {%- set _reasoning_mode = "/think" -%}
  {%- elif enable_thinking is sameas false -%}
    {%- set _reasoning_mode = "/nothink" -%}
  {%- else -%}
    {%- set _reasoning_mode = enable_thinking -%}
  {%- endif -%}
{%- else -%}
  {%- set _reasoning_mode = none -%}
{%- endif -%}
{%- set _custom_instructions = custom_instructions | default(None, true) -%}
{%- set _xml_tools_list = xml_tools | default([], true) -%}
{%- if tools is defined and tools -%}
  {%- set _xml_tools_list = tools -%}
{%- endif -%}
{%- set _python_tools = python_tools | default([], true) -%}
{%- set _has_aux_header = (_reasoning_mode is not none) or _custom_instructions or (_xml_tools_list) or (_python_tools) -%}
{%- if _has_aux_header -%}
<|start_header_id|>system<|end_header_id|>
{%- if _reasoning_mode is not none -%}
Reasoning: {{ _reasoning_mode }}
{%- endif %}
{%- if _custom_instructions %}
{{ _custom_instructions | trim }}
{%- endif %}
{% if _xml_tools_list or _python_tools %}
{{ "
### Tools
" }}
You may call one or more functions to assist with the user query.
{% if _xml_tools_list %}
You are provided with function signatures within <tools> </tools> tags:

<tools>
{% for tool in _xml_tools_list %}
{{ tool if tool is string else tool | tojson }}{{ "
" }}
{% endfor %}
</tools>

For each function call, pass a json object with function name and arguments within <tool_call> </tool_call> tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

{% endif %}
{% if _python_tools %}
When you send a message containing Python code between <|python_tag|> and <|eom_id|> tags, it will be executed in a stateful Jupyter notebook environment, and you will then be given the output.

You can use the following tools in your python code like regular functions:
<tools>
{% for tool in _python_tools %}
{{ tool if tool is string else tool | tojson }}{{ "
" }}
{% endfor %}
</tools>
{% endif %}
{% endif %}
<|eot_id|>
{%- endif -%}
{%- macro text(content) -%}
  {%- if content is string -%}
    {{- content -}}
  {%- elif content is mapping -%}
    {{- content.get('text', '') -}}
  {%- elif content is iterable -%}
    {%- for chunk in content if chunk.get('type') == 'text' -%}
      {{- chunk.text -}}
    {%- endfor -%}
  {%- endif -%}
{%- endmacro -%}

{%- set tool_names = namespace(by_id={}) -%}
{%- macro tool_calls(calls) -%}
  {%- for call in calls -%}
    {%- set function = call.function -%}
    {%- if call.get('id') -%}
      {%- set tool_names.by_id = dict(tool_names.by_id, **{call.id: function.name}) -%}
    {%- endif -%}
    {{- '<tool_call>
{"name": ' -}}
    {{- function.name | tojson -}}
    {{- ', "arguments": ' -}}
    {{- function.arguments if function.arguments is string else function.arguments | tojson -}}
    {{- '}
</tool_call>' -}}
  {%- endfor -%}
{%- endmacro -%}

{%- for message in messages -%}
  {%- set content = message.get('content') -%}
  {%- set text_content = content is string
      or (content is mapping and content.get('text') is string)
      or (content is sequence and content is not string and content is not mapping
          and content | selectattr('type', 'equalto', 'text') | list | length == content | length) -%}
  {{- '<|start_header_id|>' ~ message.role ~ '<|end_header_id|>
' -}}
  {%- if message.role == 'assistant' -%}
    {% generation %}
    {%- if message.get('reasoning_content') -%}
      {{- '<|start_think|>' ~ message.reasoning_content ~ '<|end_think|>' -}}
    {%- endif -%}
    {{- text(content) | trim -}}
    {{- tool_calls(message.get('tool_calls') or []) -}}
    {{- '<|eot_id|>' -}}
    {% endgeneration %}
  {%- elif message.role == 'tool' -%}
    {{- '<tool_response' -}}
    {%- set name = message.get('name') or tool_names.by_id.get(message.get('tool_call_id')) -%}
    {%- if name -%}{{- ' name="' ~ name ~ '"' -}}{%- endif -%}
    {{- '>' -}}
    {{- text(content) if text_content else content | tojson if content is not none else '' -}}
    {{- '</tool_response><|eot_id|>
' -}}
  {%- elif message.role == 'ipython' -%}
    {{- {"output": text(content) if text_content else content} | tojson -}}
    {{- '<|eot_id|>
' -}}
  {%- else -%}
    {{- (text(content) | trim) ~ '<|eot_id|>
' -}}
  {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
  {{- '<|start_header_id|>assistant<|end_header_id|>
' -}}
{%- endif -%}"""

_template_sha = hashlib.sha256(GRUG_DATAKIT_0921_TRAINING_TEMPLATE.encode("utf-8")).hexdigest()
if _template_sha != GRUG_DATAKIT_0921_TRAINING_TEMPLATE_SHA256:
    raise RuntimeError("The vendored Grug Datakit 09-21 training template does not match its pinned sha256.")


def datakit_row_to_chat(
    row: dict[str, Any],
    *,
    messages_field: str,
    reasoning_key: str = "reasoning",
    enable_thinking_field: str = "enable_thinking",
    kwargs_field: str = "chat_template_kwargs",
) -> dict[str, Any]:
    """One parquet row -> the row ChatProcessor renders: reasoning_content + per-row enable_thinking."""
    thinking = row.get(enable_thinking_field)
    if not isinstance(thinking, (bool, np.bool_)):
        raise ValueError(f"row {row.get('id')!r} has {enable_thinking_field}={thinking!r}; expected a bool.")
    thinking = bool(thinking)
    messages = []
    for message in row[messages_field]:
        m = {k: v for k, v in dict(message).items() if k != reasoning_key and v is not None}
        reasoning = message.get(reasoning_key)
        if isinstance(reasoning, str) and reasoning.strip():
            m["reasoning_content"] = reasoning
        messages.append(m)
    return {**row, messages_field: messages, kwargs_field: {"enable_thinking": thinking}}


class DatakitRowAdapter(BatchProcessor[dict, dict]):
    """ChatProcessor over rows rewritten by ``datakit_row_to_chat`` (1:1, so id threading is unchanged)."""

    def __init__(self, inner: ChatProcessor, *, reasoning_key: str, enable_thinking_field: str):
        self.inner = inner
        self.reasoning_key = reasoning_key
        self.enable_thinking_field = enable_thinking_field

    def __call__(self, batch: Sequence[dict]) -> Sequence[dict]:
        rows = [
            datakit_row_to_chat(
                row,
                messages_field=self.inner.messages_field,
                reasoning_key=self.reasoning_key,
                enable_thinking_field=self.enable_thinking_field,
                kwargs_field=self.inner.chat_template_kwargs_field or "chat_template_kwargs",
            )
            for row in batch
        ]
        return self.inner(rows)

    @property
    def output_exemplar(self):
        return self.inner.output_exemplar

    @property
    def num_cpus(self) -> int:
        return self.inner.num_cpus

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **self.inner.metadata,
            "row_adapter": "datakit_reasoning_enable_thinking_v1",
            "reasoning_key": self.reasoning_key,
            "enable_thinking_field": self.enable_thinking_field,
        }


@LmDatasetFormatBase.register_subclass("snowball_datakit_chat")
@dataclass(frozen=True)
class SnowballDatakitChatFormat(ChatLmDatasetFormat):
    """ChatLmDatasetFormat plus the Datakit row mapping (reasoning -> reasoning_content, per-row enable_thinking)."""

    reasoning_key: str = "reasoning"
    enable_thinking_field: str = "enable_thinking"

    def build_preprocessor(
        self, tokenizer: MarinTokenizer, *, enforce_eos: bool = True, enforce_bos: bool = True
    ) -> BatchProcessor[dict, dict]:
        inner = super().build_preprocessor(tokenizer, enforce_eos=enforce_eos, enforce_bos=enforce_bos)
        assert isinstance(inner, ChatProcessor)
        return DatakitRowAdapter(
            inner, reasoning_key=self.reasoning_key, enable_thinking_field=self.enable_thinking_field
        )
