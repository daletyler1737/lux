defmodule Lux.LLM.OpenRouterTest do
  use ExUnit.Case, async: true

  alias Lux.LLM.OpenRouter

  describe "register/1" do
    test "registers openrouter as a provider" do
      assert :ok = OpenRouter.register()

      providers = Lux.LLM.ProviderRegistry.list_providers()
      assert :openrouter in providers
    end
  end

  describe "call/3" do
    test "accepts prompt, tools, and config" do
      assert function_exported?(OpenRouter, :call, 3)
    end

    test "returns error tuple on invalid API key" do
      # Without a real API key, we expect an error response
      result = OpenRouter.call("hello", [], %{api_key: "invalid"})
      assert match?({:error, _}, result)
    end
  end

  describe "Config struct" do
    test "has correct defaults" do
      cfg = %OpenRouter.Config{}
      assert cfg.endpoint == "https://openrouter.ai/api/v1/chat/completions"
      assert cfg.model == "anthropic/claude-3.5-sonnet"
      assert cfg.temperature == 0.7
      assert cfg.tools == true
      assert cfg.stream == false
      assert cfg.retries == 3
    end

    test "can be overridden via config map" do
      cfg = struct(OpenRouter.Config, %{model: "openai/gpt-4o", temperature: 0.9})
      assert cfg.model == "openai/gpt-4o"
      assert cfg.temperature == 0.9
      assert cfg.endpoint == "https://openrouter.ai/api/v1/chat/completions"
    end
  end

  describe "integration with ProviderRegistry" do
    test "openrouter provider is registered with correct module" do
      OpenRouter.register()

      config = Lux.LLM.ProviderRegistry.get_config(:openrouter)
      assert config.module == Lux.LLM.OpenRouter
    end

    test "openrouter provider has zero or low default cost" do
      OpenRouter.register()

      config = Lux.LLM.ProviderRegistry.get_config(:openrouter)
      assert is_map(config.cost_per_1k_tokens)
      assert config.max_tokens > 0
      assert :streaming in config.features || :tools in config.features
    end
  end
end
