# RouteHub providers

Prefix the model with its provider. The key is read from the first set variable in the key column unless `api_key=` is passed, and the base URL from the base-URL column unless `api_base=` or `base_url=` is passed.

| Provider | Model string | API key variables | Base URL variables |
|---|---|---|---|
| OpenAI | `gpt-5.4`, `openai/gpt-4o` | `OPENAI_API_KEY` | `OPENAI_BASE_URL`, `OPENAI_API_BASE` |
| Anthropic (native API) | `claude-sonnet-4-6`, `anthropic/claude-opus-4-7` | `ANTHROPIC_API_KEY` | `ANTHROPIC_API_BASE`, `ANTHROPIC_BASE_URL` |
| Google Gemini | `gemini/gemini-3-flash-preview` | `GEMINI_API_KEY`, `GOOGLE_API_KEY` | `GEMINI_API_BASE` |
| Azure OpenAI | `azure/<deployment>` | `AZURE_API_KEY`, `AZURE_OPENAI_API_KEY` | `AZURE_API_BASE`, `AZURE_OPENAI_ENDPOINT` |
| Groq | `groq/llama-3.3-70b-versatile` | `GROQ_API_KEY` | `GROQ_API_BASE` |
| xAI | `xai/grok-4` | `XAI_API_KEY` | `XAI_API_BASE` |
| DeepSeek | `deepseek/deepseek-chat` | `DEEPSEEK_API_KEY` | `DEEPSEEK_API_BASE` |
| OpenRouter | `openrouter/anthropic/claude-sonnet-4.6` | `OPENROUTER_API_KEY`, `OR_API_KEY` | `OPENROUTER_API_BASE` |
| Together AI | `together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo` | `TOGETHERAI_API_KEY`, `TOGETHER_API_KEY`, `TOGETHER_AI_TOKEN` | `TOGETHERAI_API_BASE` |
| Mistral | `mistral/mistral-large-latest` | `MISTRAL_API_KEY` | `MISTRAL_API_BASE` |
| Cohere | `cohere/...` | `COHERE_API_KEY`, `CO_API_KEY` | none |
| Fireworks | `fireworks_ai/accounts/fireworks/models/...` | `FIREWORKS_API_KEY`, `FIREWORKS_AI_API_KEY` | `FIREWORKS_API_BASE` |
| Perplexity | `perplexity/sonar` | `PERPLEXITYAI_API_KEY`, `PERPLEXITY_API_KEY` | `PERPLEXITY_API_BASE` |
| Cerebras | `cerebras/...` | `CEREBRAS_API_KEY` | `CEREBRAS_API_BASE` |
| DeepInfra | `deepinfra/...` | `DEEPINFRA_API_KEY` | `DEEPINFRA_API_BASE` |
| SambaNova | `sambanova/...` | `SAMBANOVA_API_KEY` | `SAMBANOVA_API_BASE` |
| NVIDIA NIM | `nvidia_nim/...` | `NVIDIA_NIM_API_KEY` | `NVIDIA_NIM_API_BASE` |
| Moonshot | `moonshot/...` | `MOONSHOT_API_KEY` | `MOONSHOT_API_BASE` |
| Alibaba DashScope | `dashscope/...` | `DASHSCOPE_API_KEY` | `DASHSCOPE_API_BASE` |
| Hugging Face | `huggingface/...` | `HF_TOKEN`, `HUGGINGFACE_API_KEY` | none |
| Ollama | `ollama/llama3`, `ollama_chat/llama3` | none | `OLLAMA_API_BASE` |
| LM Studio | `lm_studio/...` | optional, `LM_STUDIO_API_KEY` | `LM_STUDIO_API_BASE` |
| vLLM | `hosted_vllm/...` | optional, `HOSTED_VLLM_API_KEY` | `HOSTED_VLLM_API_BASE` |
| Any OpenAI-compatible server | `openai_like/<model>`, or any name with `api_base=` | optional, `OPENAI_LIKE_API_KEY` | `OPENAI_LIKE_API_BASE` |

Notes:

- Azure also needs an API version: `api_version=`, or `AZURE_API_VERSION` or `OPENAI_API_VERSION`.
- Ollama defaults to `http://localhost:11434/v1` and LM Studio to `http://localhost:1234/v1`. `/v1` is appended to `OLLAMA_API_BASE` when missing.
- vLLM and `openai_like` have no default base URL: set the variable or pass `api_base=`.
- Local servers get a placeholder key when none is set (`ollama`, `lm-studio`, or `EMPTY` for vLLM and `openai_like`).
- `custom_llm_provider=` sets the provider for a model string without a prefix.

## Bare model names

These prefixes need no provider:

| Prefix | Provider |
|---|---|
| `gpt-`, `o1`, `o3`, `o4`, `chatgpt-`, `codex-`, `text-embedding-`, `davinci`, `babbage`, `omni-moderation` | OpenAI |
| `claude-` | Anthropic |
| `gemini-`, `gemma-` | Gemini |
| `grok-` | xAI |
| `deepseek-` | DeepSeek |
| `mistral-`, `codestral`, `magistral-`, `ministral-`, `pixtral-`, `devstral-` | Mistral |
| `command-` | Cohere |

Any other bare name raises `BadRequestError`, unless `api_base=` is passed, in which case it goes to that server as `openai_like`.

## litellm provider aliases

These litellm prefixes are accepted and mapped: `together` to `together_ai`, `fireworks` to `fireworks_ai`, `nvidia` to `nvidia_nim`, `google` and `google_ai_studio` to `gemini`, `x-ai` to `xai`, `vllm` to `hosted_vllm`, `custom_openai` to `openai_like`, `cohere_chat` to `cohere`, `text-completion-openai` to `openai`.
