# Qwen3.6 multi-turn SFT rendering

`to_qwen36.py` converts one normalized multi-turn trajectory into one supervised
fine-tuning (SFT) sample:

```text
one trajectory record -> one input_ids sequence + one labels sequence
```

It does **not** create a separate sample for each assistant turn. System messages,
user messages, assistant responses, tool calls, and tool results remain in their
original order and are concatenated into a single sequence. Every assistant turn
can contribute to the loss.

## Hypothetical input

The following abbreviated record contains two assistant turns. The first assistant
thinks and calls a weather tool; the tool returns an observation; the second
assistant thinks about the result and answers the user.

```python
record = {
    "provenance": {
        # Selects the exact tool definitions supplied to the chat template.
        "tool_ontology": "openhands_open_swe",
    },
    "trajectory": {
        "events": [
            {
                "role": "system",
                "blocks": [
                    {"kind": "text", "text": "You are a helpful assistant."}
                ],
            },
            {
                "role": "user",
                "blocks": [
                    {"kind": "text", "text": "What is the weather in Paris?"}
                ],
            },
            {
                "role": "assistant",
                "blocks": [
                    {
                        "kind": "reasoning",
                        "text": "I should query the weather tool.",
                        "trainable": True,
                    },
                    {
                        "kind": "tool_call",
                        "tool_name": "weather",
                        "arguments_parsed": {"city": "Paris"},
                    },
                ],
            },
            {
                "role": "tool",
                "blocks": [
                    {
                        "kind": "observation",
                        "tool_name": "weather",
                        "text": '{"temperature_c": 21, "condition": "sunny"}',
                    }
                ],
            },
            {
                "role": "assistant",
                "blocks": [
                    {
                        "kind": "reasoning",
                        "text": "The tool reports 21 C and sunny conditions.",
                        "trainable": True,
                    },
                    {
                        "kind": "text",
                        "text": "It is 21°C and sunny in Paris.",
                    },
                ],
            },
        ]
    },
}
```

This example is illustrative: the named tool must exist in the selected ontology
in a real record.

## Conceptual rendered sample

The real text is produced by Qwen's chat template. Omitting the template-generated
tool documentation, it is conceptually:

```text
<|im_start|>system
[tool definitions and system instructions]<|im_end|>
<|im_start|>user
What is the weather in Paris?<|im_end|>
<|im_start|>assistant
<think>
I should query the weather tool.
</think>

<tool_call>
<function=weather>
<parameter=city>
Paris
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"temperature_c": 21, "condition": "sunny"}
</tool_response><|im_end|>
<|im_start|>assistant
<think>
The tool reports 21 C and sunny conditions.
</think>

It is 21°C and sunny in Paris.<|im_end|>
```

Notice that the tool observation is rendered as a `user` turn wrapped in
`<tool_response>`. Consecutive tool observations are merged into one such turn.

## Input tokens versus loss labels

Every rendered token is retained in `input_ids`, because later assistant turns
need the complete conversation as context. `labels` controls which tokens produce
training loss:

| Conversation portion | In `input_ids` | Trained in `labels` |
| --- | --- | --- |
| System prompt and tool definitions | Yes | No (`-100`) |
| User messages | Yes | No (`-100`) |
| Tool results / observations | Yes | No (`-100`) |
| Assistant generation prefix | Yes | No (`-100`) |
| Assistant reasoning | Yes | According to `reasoning.trainable` |
| Assistant answer text | Yes | Yes |
| Assistant tool calls and arguments | Yes | Yes |
| Assistant `<\|im_end\|>` | Yes | Yes by default |

For the hypothetical trajectory, the loss is therefore applied to both assistant
turns:

```text
MASK:  system + user question
MASK:  <|im_start|>assistant\n<think>\n
TRAIN: first reasoning + </think> + weather tool call + <|im_end|>
MASK:  tool result
MASK:  <|im_start|>assistant\n<think>\n
TRAIN: second reasoning + </think> + final answer + <|im_end|>
```

The first assistant turn teaches the model how to reason and emit a correctly
formatted tool call. The tool result remains visible as context but produces no
loss. The second assistant turn teaches the model how to interpret that result and
produce the final response.

## Thinking-token handling

Qwen3.6 supports thinking and non-thinking serving modes. The renderer selects one
mode for the entire record.

### Thinking record

If any event contains non-empty reasoning, the default `PER_RECORD` policy treats
the trajectory as a thinking record:

```text
MASK:  <|im_start|>assistant\n<think>\n
TRAIN: reasoning text + \n</think>\n\n
```

The opening scaffold is masked because it is supplied at inference time. The
reasoning and closing `</think>` are trained because the model must generate them.
If an individual assistant turn has no reasoning inside an otherwise-thinking
record, the model is still trained to close the empty think block.

### Non-thinking record

If the entire record has no reasoning, the default policy treats it as
non-thinking:

```text
MASK:  <|im_start|>assistant\n<think>\n\n</think>\n\n
TRAIN: assistant answer or tool call + <|im_end|>
```

The complete empty think block is masked because `enable_thinking=False` supplies
that block during inference.

## Important properties

- One record becomes one sample; there is no per-turn splitting.
- All prior turns are retained as context for later assistant turns.
- All assistant turns can be supervised in the same sample.
- Tool calls are model output and are trained.
- Tool results are environment input and are masked.
- Over-length records are rejected rather than automatically split or truncated.
- The renderer verifies that its segmented text exactly matches Qwen's chat
  template before producing tokens.
