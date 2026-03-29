defmodule Lux.LLM.OpenRouter do
  @moduledoc """
  OpenRouter LLM implementation that supports passing Beams, Prisms, and Lenses as tools.
  """

  @behaviour Lux.LLM

  alias Lux.Beam
  alias Lux.Lens
  alias Lux.LLM.ResponseSignal
  alias Lux.Prism

  require Beam
  require Lens
  require Logger

  @endpoint "https://openrouter.ai/api/v1/chat/completions"
  @default_model "anthropic/claude-3.5-sonnet"

  defmodule Config do
    @moduledoc """
    Configuration module for OpenRouter.
    """
    @type t :: %__MODULE__{
            endpoint: String.t(),
            model: String.t(),
            api_key: String.t(),
            temperature: float(),
            max_tokens: integer() | nil,
            tools: boolean(),
            tool_choice: atom() | String.t() | nil,
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

  @impl true
  def call(prompt, tools, config) do
    cfg =
      struct(
        Config,
        Map.merge(
          %{
            model: Application.get_env(:lux, :open_router_models)[:default],
            api_key: Application.get_env(:lux, :api_keys)[:openrouter]
          },
          config
        )
      )

    messages = cfg.messages ++ build_messages(prompt)
    tools_config = if cfg.tools, do: build_tools_config(tools), else: []

    body =
      %{
        model: Lux.Config.resolve(cfg.model),
        messages: messages,
        temperature: cfg.temperature
      }
      |> maybe_add_max_tokens(cfg.max_tokens)
      |> maybe_add_tools(tools_config, cfg.tool_choice)
      |> maybe_add_stream(cfg.stream)

    headers =
      [
        {"Authorization", "Bearer #{Lux.Config.resolve(cfg.api_key)}"},
        {"Content-Type", "application/json"}
      ]
      |> maybe_add_header("HTTP-Referer", cfg.site_url)
      |> maybe_add_header("X-OpenRouter-Title", cfg.site_name)
      |> maybe_add_header("X-OpenRouter-Route", cfg.route)

    [
      url: @endpoint,
      json: body,
      headers: headers
    ]
    |> Keyword.merge(Application.get_env(:lux, __MODULE__, []))
    |> Req.new()
    |> Req.post()
    |> case do
      {:ok, %{status: 200} = response} ->
        handle_response(response, cfg)

      {:ok, %{status: 401}} ->
        {:error, :invalid_api_key}

      {:ok, %{status: 403, body: %{"error" => msg}}} ->
        {:error, {:forbidden, msg}}

      {:ok, %{status: 429}} ->
        {:error, :rate_limited}

      {:ok, %{status: status, body: %{"error" => msg}}} ->
        {:error, {status, msg}}

      {:error, error} ->
        handle_error(error, cfg)
    end
  end

  @doc """
  Register OpenRouter as a provider in the Lux.LLM.ProviderRegistry.

  ## Examples

      iex> Lux.LLM.OpenRouter.register()
      :ok
  """
  @spec register() :: :ok | {:error, term()}
  def register do
    :ok = Lux.LLM.ProviderRegistry.register(:openrouter, %{
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
      priority: 5,
      enabled: true
    })
  end

  defp build_messages(prompt) do
    [%{role: "user", content: prompt}]
  end

  defp build_tools_config([]), do: []
  defp build_tools_config(tools), do: Enum.map(tools, &tool_to_function/1)

  defp maybe_add_max_tokens(body, nil), do: body
  defp maybe_add_max_tokens(body, max_tokens), do: Map.put(body, :max_tokens, max_tokens)

  defp maybe_add_tools(body, [], _choice), do: body
  defp maybe_add_tools(body, tools, choice) do
    body
    |> Map.put(:tools, tools)
    |> Map.put(:tool_choice, format_tool_choice(choice))
  end

  defp maybe_add_stream(body, false), do: body
  defp maybe_add_stream(body, true), do: Map.put(body, :stream, true)

  defp format_tool_choice(:none), do: "none"
  defp format_tool_choice(:auto), do: "auto"
  defp format_tool_choice(name) when is_binary(name) do
    %{"type" => "function", "function" => %{"name" => String.replace(name, ".", "_")}}
  end
  defp format_tool_choice(_), do: "auto"

  defp maybe_add_header(headers, _key, nil), do: headers
  defp maybe_add_header(headers, _key, ""), do: headers
  defp maybe_add_header(headers, key, value), do: headers ++ [{key, value}]

  defp tool_to_function({:python, path}) do
    path
    |> Prism.view()
    |> tool_to_function()
  end

  def tool_to_function(tool_module) when is_atom(tool_module) and not is_nil(tool_module) do
    cond do
      Lux.prism?(tool_module) ->
        tool_to_function(tool_module.view())

      Lux.beam?(tool_module) ->
        tool_to_function(tool_module.view())

      Lux.lens?(tool_module) ->
        tool_to_function(tool_module.view())

      true ->
        raise "Unsupported tool type: #{inspect(tool_module)}"
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

  defp handle_response(%{body: body}, _cfg) do
    with %{"choices" => [choice | _]} <- body,
         %{"message" => message} <- choice,
         {:ok, content} <- parse_content(message["content"]),
         {:ok, tool_calls_results} <- execute_tool_calls(message["tool_calls"]) do
      payload = %{
        content: content,
        model: body["model"] || "unknown",
        finish_reason: choice["finish_reason"],
        tool_calls: message["tool_calls"],
        tool_calls_results: tool_calls_results
      }

      metadata = %{
        id: body["id"],
        created: body["created"],
        usage: body["usage"],
        system_fingerprint: body["system_fingerprint"]
      }

      %{
        schema_id: ResponseSignal,
        payload: payload,
        metadata: metadata
      }
      |> Lux.Signal.new()
      |> ResponseSignal.validate()
    end
  end

  def parse_content(content) when is_binary(content) do
    case Jason.decode(content) do
      {:ok, structured_output} -> {:ok, structured_output}
      {:error, _} -> {:ok, content}
    end
  end

  def parse_content(nil), do: {:ok, nil}
  def parse_content(other), do: {:ok, other}

  def execute_tool_calls(nil), do: {:ok, nil}

  def execute_tool_calls(tool_calls) when is_list(tool_calls) do
    tool_calls
    |> Enum.map(&execute_tool_call/1)
    |> Enum.reduce({:ok, []}, fn
      {:ok, result, _log}, {:ok, results} -> {:ok, [result | results]}
      {:ok, result}, {:ok, results} -> {:ok, [result | results]}
      error, _ -> error
    end)
  end

  def execute_tool_call(%{"function" => %{"name" => tool_name, "arguments" => args}}) do
    args = Jason.decode!(args)
    execute_tool(tool_name, args, nil)
  end

  def execute_tool(tool_name, args, ctx \\ nil)

  def execute_tool(tool_name, args, ctx) when is_binary(tool_name) do
    tool_name
    |> String.replace("_", ".")
    |> List.wrap()
    |> Module.concat()
    |> Code.ensure_loaded()
    |> case do
      {:module, module_name} ->
        execute_tool(module_name, args, ctx)

      {:error, :nofile} ->
        {:error,
         "Failed to load tool module #{tool_name}: It doesn't seems to be implemented or reachable"}

      {:error, error} ->
        {:error, "Failed to load tool module #{tool_name}: #{inspect(error)}"}
    end
  end

  def execute_tool(module_name, args, ctx) when is_atom(module_name) do
    cond do
      Lux.prism?(module_name) -> module_name.handler(args, ctx)
      Lux.beam?(module_name)  -> module_name.run(args, ctx)
      true -> {:error, "Tool #{module_name} does not have a registered handler or run function"}
    end
  end

  defp handle_error(error, cfg) do
    Logger.error("OpenRouter API error: #{inspect(error)}")

    if cfg.retries > 0 do
      Logger.warning("Retrying... #{cfg.retries} attempts left")
      call("retry", [], %{cfg | retries: cfg.retries - 1})
    else
      {:error, "OpenRouter API error: #{inspect(error)}"}
    end
  end
end
