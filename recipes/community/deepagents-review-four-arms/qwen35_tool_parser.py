"""vLLM tool-parser plugin wrapping Prime's training-time Qwen3.5 parser.

vLLM's built-in qwen3_xml parser regex-matches decoded text and misses
valid tool calls the model was trained to emit (measured: 44/508 dbviews
rollouts died with a well-formed <tool_call> left unparsed in content,
~8.7% episode mortality; quant code is shorter and mostly survived).
PrimeIntellect-ai/renderers parses on token ids with the exact grammar
the RL renderer trained against - this plugin delegates to it.

Register with:
    --tool-parser-plugin /root/prime_qwen35_tool_parser.py
    --tool-call-parser prime_qwen35

Streaming requests fall back to non-streaming extraction semantics (the
sub-agent loop and eval harness are non-streaming; interactive chat
streams text only, tools are not offered on that path).
"""

import json
from collections.abc import Sequence

from renderers import Qwen35Renderer, Qwen35RendererConfig, ToolSpec

# vLLM 0.26.0 split the old monolithic vllm.entrypoints.openai.protocol
# module: request schema lives under chat_completion.protocol, the tool-call
# result/delta types under engine.protocol.
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.tool_parsers.abstract_tool_parser import ToolParser, ToolParserManager


@ToolParserManager.register_module("prime_qwen35")
class PrimeQwen35ToolParser(ToolParser):
    # vLLM 0.26 constructs tool parsers per request as
    # ToolParser(tokenizer, tools); older vLLM passed only the tokenizer.
    # Accept and forward the optional tools arg so both call styles work.
    def __init__(self, tokenizer, tools=None):
        super().__init__(tokenizer, tools)
        self._renderer = Qwen35Renderer(tokenizer, Qwen35RendererConfig(enable_thinking=False))

    def _specs(self, request: ChatCompletionRequest) -> list[ToolSpec] | None:
        if not request.tools:
            return None
        return [ToolSpec(**t.model_dump()) for t in request.tools]

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        token_ids = self.model_tokenizer.encode(model_output, add_special_tokens=False)
        parsed = self._renderer.parse_response(token_ids, tools=self._specs(request))
        if not parsed.tool_calls:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        calls = []
        for tc in parsed.tool_calls:
            if tc.name is None:
                continue
            args = tc.arguments
            if not isinstance(args, str):
                args = json.dumps(args or {}, ensure_ascii=False)
            calls.append(ToolCall(function=FunctionCall(name=tc.name, arguments=args)))
        return ExtractedToolCallInformation(
            tools_called=bool(calls),
            tool_calls=calls,
            content=parsed.content or None,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        return DeltaMessage(content=delta_text)
