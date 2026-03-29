defmodule Lux.LLM.OpenRouter do
  @moduledoc """
  OpenRouter LLM implementation — unified API for hundreds of LLM models.

  OpenRouter provides a single endpoint to access models from OpenAI, Anthropic,
  Google, Meta, Mistral, and many others. It automatically handles fallbacks
  and selects the most cost-effective options.

  Supports passing Beams, Prisms, and Lenses as tools.

  ## Configuration

      config :lux, :api_keys,
        openrouter: "sk-or-v1-..."

      config :lux, :open_router_models,
        default: "anthropic/claude-3.5-sonnet"

  ## Usage

      Lux.LLM.OpenRouter.call("Hello!", [], %{
        model: "openai/gpt-4o",
        api_key: "sk-or-v1-..."
      })
  """

  @behaviour Lux.LLM

  alias Lux.Beam
  alias Lux.Lens
  alias Lux.LLM.ProviderRegistry
  alias Lux.LLM.ResponseSignal
  alias Lux.Prism

  require Beam
  require Lens
  require Logger

  @endpoint "https://openrouter.ai/api/v1/chat/completions"
  @default_model "anthropic/claude-3.5-sonnet"

  defmodule Config do
    @moduledoc false

    @type t :: %__MODULE__{
            endpoint: String.t(),
            model: String.t(),
            api_key: String.t(),
            temperature: float(),
            max_tokens: integer() | nil,
            tools: boolean(),
            tool_choice: atom() | String.t(),
            stream: boolean(),
            site_url: String.t() | nil,
            site_name: String.t() | nil,
            route: String.t() | nil,
            retries: integer(),
            messages: [map()]
          }

    defstruct endpoint: "https://openrouter.ai/api/v1/chat/completions",
              model: "anthropic/claude-3.5-sonnet",
              api_key: nil,
              temperature: 0.7,
              max_tokens: nil,
              tools: true,
              tool_choice: :auto,
              stream: false,
              site_url: nil,
              site_name: nil,
              route: nil,
              retries: 3,
              messages: []
  end

  @doc """
  Register OpenRouter as a provider in the Lux.LLM.ProviderRegistry.

  ## Examples

      iex> Lux.LLM.OpenRouter.register()
      :ok

  """
  @spec register() :: :ok | {:error, term()}
  def register do
    :ok =
      Lux.LLM.ProviderRegistry.register(:openrouter, %{
        module: __MODULE__,
        models: [
          "anthropic/claude-3.5-sonnet",
          "openai/gpt-4o",
          "openai/gpt-4o-mini",
          "google/gemini-pro",
          "meta-llama/llama-3-70b-instruct",
          "mistralai/mistral-7b-instruct"
        ],
        cost_per_1k_tokens: %{
          "anthropic/claude-3.5-sonnet" => %{input: 0.003, output: 0.015},
          "openai/gpt-4o" => %{input: 0.005, output: 0.015},
          "openai/gpt-4o-mini" => %{input: 0.00015, output: 0.0006},
          "google/gemini-pro" => %{input: 0.00125, output: 0.005},
          "meta-llama/llama-3-70b-instruct" => %{input: 0.0009, output: 0.0009},
          "mistralai/mistral-7b-instruct" => %{input: 0.0002, output: 0.0006}
        },
        max_tokens: 128_000,
        features: [:streaming, :tools, :json_mode],
        priority: 3,
        enabled: true
      })
  end

  @impl true
  @spec call(String.t(), [Lux.Prism.t() | Lux.Beam.t() | Lux.Lens.t()], map()) ::
          {:ok, map()} | {:error, term()}
  def call(prompt, tools, config) do
    cfg = resolve_config(config)

    messages = cfg.messages ++ build_messages(prompt)
    tools_config = if cfg.tools, do: build_tools_config(tools), else: []

    body =
      %{
        model: cfg.model,
        messages: messages,
        temperature: cfg.temperature
      }
      |> maybe_put(:max_tokens, cfg.max_tokens)
      |> maybe_put_tools(tools_config, cfg.tool_choice)

    headers = build_headers(cfg)

    request =
      [url: @endpoint, json: body, headers: headers]
      |> Keyword.merge(Application.get_env(:lux, __MODULE__, []))

    result = request |> Req.new() |> Req.post() |> handle_response(cfg)

    case result do
      {:ok, _signal} = success ->
        success

      {:error, reason} when cfg.retries > 0 ->
        Logger.warning("OpenRouter request failed: #{inspect(reason)}, retrying...")
        call(prompt, tools, %{cfg | retries: cfg.retries - 1})

      {:error, _reason} = error ->
        error
    end
  end

  # ─── Message Building ──────────────────────────────────────────────────────

  defp build_messages(prompt) when is_binary(prompt), do: [%{role: "user", content: prompt}]
  defp build_messages(messages) when is_list(messages), do: messages

  # ─── Tools Conversion ─────────────────────────────────────────────────────

  defp build_tools_config([]), do: []
  defp build_tools_config(tools), do: Enum.map(tools, &tool_to_function/1)

  defp tool_to_function({:python, path}) do
    path |> Prism.view() |> tool_to_function()
  end

  def tool_to_function(tool_module) when is_atom(tool_module) and not is_nil(tool_module) do
    cond do
      Lux.prism?(tool_module) -> tool_module.view() |> tool_to_function()
      Lux.beam?(tool_module)  -> tool_module.view() |> tool_to_function()
      Lux.lens?(tool_module)  -> tool_module.view() |> tool_to_function()
      true -> raise "Unsupported tool type: #{inspect(tool_module)}"
    end
  end

  def tool_to_function(%Beam{module_name: name, description: desc, input_schema: schema}) do
    %{
      type: "function",
      function: %{
        name: String.replace(name, ".", "_"),
        description: desc || "",
        parameters: schema
      }
    }
  end

  def tool_to_function(%Prism{module_name: name, description: desc, input_schema: schema}) do
    %{
      type: "function",
      function: %{
        name: String.replace(name, ".", "_"),
        description: desc || "",
        parameters: schema
      }
    }
  end

  def tool_to_function(%Lens{name: name, description: desc, schema: schema}) do
    %{
      type: "function",
      function: %{
        name: name || "unnamed_lens",
        description: desc || "",
        parameters: schema
      }
    }
  end

  # ─── Request Helpers ──────────────────────────────────────────────────────

  defp maybe_put(body, _key, nil), do: body
  defp maybe_put(body, key, value), do: Map.put(body, key, value)

  defp maybe_put_tools(body, [], _choice), do: body

  defp maybe_put_tools(body, tools, choice) do
    body
    |> Map.put(:tools, tools)
    |> Map.put(:tool_choice, format_tool_choice(choice))
  end

  defp format_tool_choice(:none), do: "none"
  defp format_tool_choice(:auto), do: "auto"

  defp format_tool_choice(name) when is_binary(name) do
    %{"type" => "function", "function" => %{"name" => String.replace(name, ".", "_")}}
  end

  defp format_tool_choice(_), do: "auto"

  defp build_headers(cfg) do
    headers = [
      {"Authorization", "Bearer #{Lux.Config.resolve(cfg.api_key)}"},
      {"Content-Type", "application/json"}
    ]

    headers
    |> maybe_append({"HTTP-Referer", cfg.site_url})
    |> maybe_append({"X-OpenRouter-Title", cfg.site_name})
    |> maybe_append({"X-OpenRouter-Route", cfg.route})
  end

  defp maybe_append(headers, {_k, nil}), do: headers
  defp maybe_append(headers, {_k, ""}), do: headers
  defp maybe_append(headers, {k, v}), do: headers ++ [{k, v}]

  # ─── Response Handling ────────────────────────────────────────────────────

  defp handle_response({:ok, %{status: 200, body: body}}, _cfg) do
    with %{"choices" => [choice | _]} <- body,
         %{"message" => message, "finish_reason" => finish_reason} <- choice,
         {:ok, content} <- parse_content(message["content"]),
         {:ok, tool_calls_results} <- execute_tool_calls(message["tool_calls"]) do
      payload = %{
        content: content,
        model: body["model"] || "unknown",
        finish_reason: finish_reason,
        tool_calls: message["tool_calls"],
        tool_calls_results: tool_calls_results
      }

      metadata = %{
        id: body["id"],
        created: body["created"],
        usage: body["usage"],
        system_fingerprint: Map.get(body, "system_fingerprint")
      }

      %{schema_id: ResponseSignal, payload: payload, metadata: metadata}
      |> Lux.Signal.new()
      |> ResponseSignal.validate()
    else
      nil -> {:error, "OpenRouter: unexpected response structure"}
      {:error, _} = err -> err
    end
  end

  defp handle_response({:ok, %{status: 401}}, _cfg), do: {:error, :invalid_api_key}

  defp handle_response({:ok, %{status: 403, body: %{"error" => %{"message" => msg}}}, _cfg) do
    {:error, {:forbidden, msg}}
  end

  defp handle_response({:ok, %{status: 429}}, _cfg), do: {:error, :rate_limited}

  defp handle_response({:ok, %{status: status, body: %{"error" => %{"message" => msg}}}, _cfg) do
    {:error, {status, msg}}
  end

  defp handle_response({:ok, %{status: status, body: %{"error" => msg}}}, _cfg)
      when is_binary(msg) do
    {:error, {status, msg}}
  end

  defp handle_response({:error, %{reason: reason}}, _cfg) do
    Logger.error("OpenRouter network error: #{inspect(reason)}")
    {:error, "OpenRouter request failed: #{inspect(reason)}"}
  end

  # ─── Content Parsing ─────────────────────────────────────────────────────

  defp parse_content(nil), do: {:ok, nil}

  defp parse_content(content) when is_binary(content) do
    case Jason.decode(content) do
      {:ok, decoded} -> {:ok, decoded}
      {:error, _} -> {:ok, content}
    end
  end

  defp parse_content(other), do: {:ok, other}

  # ─── Tool Execution ──────────────────────────────────────────────────────

  defp execute_tool_calls(nil), do: {:ok, nil}

  defp execute_tool_calls(tool_calls) when is_list(tool_calls) do
    results =
      tool_calls
      |> Enum.map(&execute_tool_call/1)
      |> Enum.reduce({:ok, []}, fn
        {:ok, result, _log}, {:ok, acc} -> {:ok, [result | acc]}
        {:ok, result}, {:ok, acc} -> {:ok, [result | acc]}
        {:error, _} = err, _ -> err
      end)

    results
  end

  defp execute_tool_call(%{"function" => %{"name" => name, "arguments" => args}}) do
    args_decoded = Jason.decode!(args)
    execute_tool(name, args_decoded, nil)
  end

  defp execute_tool(name, args, ctx \\ nil)

  defp execute_tool(name, args, ctx) when is_binary(name) do
    name
    |> String.replace("_", ".")
    |> List.wrap()
    |> Module.concat()
    |> Code.ensure_loaded()
    |> case do
      {:module, module_name} -> execute_tool(module_name, args, ctx)
      {:error, :nofile} -> {:error, "Tool module not found: #{name}"}
      {:error, reason} -> {:error, "Failed to load #{name}: #{inspect(reason)}"}
    end
  end

  defp execute_tool(module_name, args, ctx) when is_atom(module_name) do
    cond do
      Lux.prism?(module_name) -> module_name.handler(args, ctx)
      Lux.beam?(module_name) -> module_name.run(args, ctx)
      Lux.lens?(module_name) -> module_name.focus(args)
      true -> {:error, "Module #{module_name} is not a valid Beam, Prism, or Lens"}
    end
  end

  # ─── Config Resolution ──────────────────────────────────────────────────

  defp resolve_config(config) do
    defaults = %{
      model: default_model(),
      api_key: api_key()
    }

    struct(Config, Map.merge(defaults, config))
  end

  defp default_model do
    case Application.get_env(:lux, :open_router_models) do
      nil -> @default_model
      cfg -> cfg[:default] || @default_model
    end
  end

  defp api_key do
    case Application.get_env(:lux, :api_keys) do
      nil -> nil
      keys -> keys[:openrouter]
    end
  end
end
