"""Integration tests — full workflow from tool definition to JSON Schema export."""

from typing import Any

from quartermaster_tools import (
    AbstractTool,
    Chain,
    Handler,
    ToolDescriptor,
    ToolParameter,
    ToolParameterOption,
    ToolRegistry,
    ToolResult,
)


class WeatherTool(AbstractTool):
    """Example tool: fetch weather for a city."""

    def name(self) -> str:
        return "get_weather"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="city",
                description="City name",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="units",
                description="Temperature units",
                type="string",
                default="celsius",
                options=[
                    ToolParameterOption(label="Celsius", value="celsius"),
                    ToolParameterOption(label="Fahrenheit", value="fahrenheit"),
                ],
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Get current weather",
            long_description="Fetches the current weather for a specified city",
            version=self.version(),
            parameters=self.parameters(),
        )

    def run(self, **kwargs: Any) -> ToolResult:
        city = kwargs.get("city", "")
        units = kwargs.get("units", "celsius")
        # Simulated response
        temp = 22 if units == "celsius" else 72
        return ToolResult(
            success=True,
            data={"city": city, "temperature": temp, "units": units, "condition": "sunny"},
        )


class NormalizeCityHandler(Handler):
    def handle(self, data: dict[str, Any]) -> dict[str, Any]:
        if "city" in data:
            data["city"] = data["city"].strip().title()
        return data


class ValidateInputHandler(Handler):
    def handle(self, data: dict[str, Any]) -> dict[str, Any]:
        if not data.get("city"):
            raise ValueError("City is required")
        return data


class AddTimestampHandler(Handler):
    def handle(self, data: dict[str, Any]) -> dict[str, Any]:
        data["timestamp"] = "2026-04-10T12:00:00Z"
        return data


class TestEndToEndWorkflow:
    """Full workflow: create tools, register, chain handlers, execute, export schema."""

    def test_tool_creation_and_execution(self):
        tool = WeatherTool()
        result = tool.run(city="London", units="celsius")
        assert result.success
        assert result.data["city"] == "London"
        assert result.data["temperature"] == 22

    def test_tool_safe_run_validates(self):
        tool = WeatherTool()
        # Missing required 'city'
        result = tool.safe_run(units="fahrenheit")
        assert not result.success
        assert "city" in result.error

    def test_registry_workflow(self):
        reg = ToolRegistry()
        reg._plugins_loaded = True

        # Register
        reg.register(WeatherTool())
        assert "get_weather" in reg
        assert len(reg) == 1

        # Lookup
        tool = reg.get("get_weather")
        result = tool.run(city="Paris")
        assert result.success
        assert result.data["city"] == "Paris"

        # List
        descriptors = reg.list_tools()
        assert len(descriptors) == 1
        assert descriptors[0].name == "get_weather"

    def test_chain_preprocessing(self):
        chain = (
            Chain()
            .add_handler(NormalizeCityHandler())
            .add_handler(ValidateInputHandler())
            .add_handler(AddTimestampHandler())
        )
        result = chain.run({"city": "  london  ", "units": "celsius"})
        assert result["city"] == "London"
        assert "timestamp" in result

    def test_chain_then_tool_execution(self):
        """Preprocess input with chain, then execute tool."""
        chain = Chain().add_handler(NormalizeCityHandler()).add_handler(ValidateInputHandler())
        preprocessed = chain.run({"city": "  new york  "})

        tool = WeatherTool()
        result = tool.run(**preprocessed)
        assert result.success
        assert result.data["city"] == "New York"

    def test_json_schema_export_openai(self):
        reg = ToolRegistry()
        reg._plugins_loaded = True
        reg.register(WeatherTool())

        tools = reg.to_openai_tools()
        assert len(tools) == 1

        func = tools[0]
        assert func["type"] == "function"
        assert func["function"]["name"] == "get_weather"

        params = func["function"]["parameters"]
        assert params["type"] == "object"
        assert "city" in params["properties"]
        assert params["properties"]["city"]["type"] == "string"
        assert params["required"] == ["city"]

        # Units should have enum
        assert "enum" in params["properties"]["units"]
        assert "celsius" in params["properties"]["units"]["enum"]

    def test_json_schema_export_anthropic(self):
        reg = ToolRegistry()
        reg._plugins_loaded = True
        reg.register(WeatherTool())

        tools = reg.to_anthropic_tools()
        assert len(tools) == 1

        tool = tools[0]
        assert tool["name"] == "get_weather"
        assert tool["description"] == "Get current weather"
        assert tool["input_schema"]["type"] == "object"
        assert "city" in tool["input_schema"]["properties"]

    def test_json_schema_export_mcp(self):
        reg = ToolRegistry()
        reg._plugins_loaded = True
        reg.register(WeatherTool())

        tools = reg.to_mcp_tools()
        assert len(tools) == 1

        tool = tools[0]
        assert tool["name"] == "get_weather"
        assert "inputSchema" in tool
        assert tool["inputSchema"]["type"] == "object"

    def test_version_management(self):
        """Register multiple versions and look them up."""
        reg = ToolRegistry()
        reg._plugins_loaded = True
        reg.register(WeatherTool())

        class WeatherToolV2(WeatherTool):
            def version(self) -> str:
                return "2.0.0"

            def run(self, **kwargs: Any) -> ToolResult:
                result = super().run(**kwargs)
                result.data["forecast"] = "partly cloudy tomorrow"
                return result

        reg.register(WeatherToolV2())

        # Latest should be v2
        latest = reg.get("get_weather")
        assert latest.version() == "2.0.0"

        # Can still get v1
        v1 = reg.get("get_weather", "1.0.0")
        assert v1.version() == "1.0.0"

        # V2 result has forecast
        result = latest.run(city="Tokyo")
        assert "forecast" in result.data

    def test_full_pipeline(self):
        """Complete pipeline: register tools, preprocess, execute, validate output."""
        # Setup registry
        reg = ToolRegistry()
        reg._plugins_loaded = True
        reg.register(WeatherTool())

        # Setup preprocessing chain
        preprocess = Chain().add_handler(NormalizeCityHandler()).add_handler(ValidateInputHandler())

        # Simulate agent workflow
        user_input = {"city": "  berlin  ", "units": "fahrenheit"}

        # 1. Preprocess
        processed = preprocess.run(user_input)
        assert processed["city"] == "Berlin"

        # 2. Get tool from registry
        tool = reg.get("get_weather")

        # 3. Validate params
        errors = tool.validate_params(**processed)
        assert errors == []

        # 4. Execute
        result = tool.run(**processed)
        assert result.success
        assert result.data["temperature"] == 72
        assert result.data["units"] == "fahrenheit"

        # 5. Export schema (for LLM integration)
        schemas = reg.to_json_schema()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "get_weather"
