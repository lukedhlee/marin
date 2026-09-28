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


class PrerenderedRowProcessor(BatchProcessor[dict, dict]):
    """Rows that arrive already rendered and masked: ``ids`` / ``loss`` pass through as ``input_ids`` /
    ``assistant_masks``, the two fields ChatProcessor writes, so the cache and every step after it (ChatDataset's
    packing, its shift of the mask onto the predicting position, the loss) are the chat path's, unchanged.

    Nothing is re-tokenized. Each row is checked instead: equal lengths, loss values 0/1, ids inside the vocabulary,
    BOS first and untrained, at least one trained token, and at most ``max_tokens`` ids (a longer row would be cut by
    the packer's "left" slice without a word, dropping its trained tail).
    """

    def __init__(self, *, ids_field: str, loss_field: str, max_tokens: int, vocab_size: int, bos_id: int):
        self.ids_field = ids_field
        self.loss_field = loss_field
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size
        self.bos_id = bos_id

    def __call__(self, batch: Sequence[dict]) -> Sequence[dict]:
        out = []
        for row in batch:
            ids = np.asarray(row[self.ids_field], dtype=np.int64)
            loss = np.asarray(row[self.loss_field], dtype=np.int64)
            name = row.get("id")
            if ids.ndim != 1 or ids.shape != loss.shape:
                raise ValueError(f"row {name!r}: ids {ids.shape} and loss {loss.shape} differ")
            if not 0 < ids.size <= self.max_tokens:
                raise ValueError(f"row {name!r} has {ids.size} tokens; rows must be 1..{self.max_tokens}")
            if ids.min() < 0 or ids.max() >= self.vocab_size:
                raise ValueError(f"row {name!r} has ids outside [0, {self.vocab_size})")
            if not np.isin(loss, (0, 1)).all():
                raise ValueError(f"row {name!r} has loss values other than 0/1")
            if ids[0] != self.bos_id or loss[0] != 0:
                raise ValueError(f"row {name!r} must start with an untrained BOS ({self.bos_id})")
            if not loss.any():
                raise ValueError(f"row {name!r} trains no token")
            out.append({"input_ids": ids.astype(np.int32), "assistant_masks": loss.astype(np.int32)})
        return out

    @property
    def output_exemplar(self):
        return {"input_ids": np.zeros((0,), dtype=np.int32), "assistant_masks": np.zeros((0,), dtype=np.int32)}

    @property
    def num_cpus(self) -> int:
        return 1

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "row_adapter": "prerendered_ids_loss_v1",
            "ids_field": self.ids_field,
            "loss_field": self.loss_field,
            "max_tokens": self.max_tokens,
            "vocab_size": self.vocab_size,
            "bos_id": self.bos_id,
        }


@LmDatasetFormatBase.register_subclass("snowball_prerendered_chat")
@dataclass(frozen=True)
class SnowballPrerenderedChatFormat(ChatLmDatasetFormat):
    """Rows rendered and masked upstream (relay SFT: OpenThoughts-Agent data/relay/sft/render.py with 09-21's own
    chat_template.jinja; loss on the teacher's turns only, 09-21's turns in context but masked). A ChatLmDatasetFormat
    so the cache layout and ChatDataset (packing, mask shift) are the chat path's; only the preprocessor differs, and
    it copies ``ids`` / ``loss`` instead of rendering. ``mask_user_turns`` must stay True: False would drop the mask
    and train every token."""

    ids_field: str = "ids"
    loss_field: str = "loss"
    max_tokens: int = 65536

    def build_preprocessor(
        self, tokenizer: MarinTokenizer, *, enforce_eos: bool = True, enforce_bos: bool = True
    ) -> BatchProcessor[dict, dict]:
        del enforce_eos, enforce_bos  # the rows carry their own BOS and end-of-turn tokens
        if not self.mask_user_turns:
            raise ValueError("SnowballPrerenderedChatFormat needs mask_user_turns=True (the row's loss is the mask)")
        return PrerenderedRowProcessor(
            ids_field=self.ids_field,
            loss_field=self.loss_field,
            max_tokens=self.max_tokens,
            vocab_size=len(tokenizer),
            bos_id=tokenizer.bos_token_id,
        )


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
