/**
 * MCP Stdio Client - Executes MCP server commands and calls tools via stdio transport
 *
 * Environment variables:
 * - MCP_COMMAND: The command to start the MCP server (e.g., "npx @modelcontextprotocol/server-slack")
 * - MCP_OPERATION: The operation to perform: "list_tools" or "call_tool" (default: "call_tool")
 * - MCP_TOOL_NAME: The name of the tool to call (required for call_tool operation)
 * - MCP_TOOL_ARGUMENTS: JSON string of arguments to pass to the tool
 * - MCP_ENV_*: Additional environment variables to pass to the MCP server (prefix stripped)
 */

const { spawn } = require("child_process");
const readline = require("readline");

const MCP_PROTOCOL_VERSION = "2024-11-05";
const INITIALIZE_TIMEOUT = 60000; // 60 seconds for slow package resolution
const TOOL_CALL_TIMEOUT = 120000; // 2 minutes

class McpStdioClient {
  constructor(command, serverEnv = {}) {
    this.command = command;
    this.serverEnv = serverEnv;
    this.process = null;
    this.messageId = 0;
    this.pendingRequests = new Map();
    this.initialized = false;
    this.serverStderrLines = [];
    this.processExited = false;
    this.exitCode = null;
  }

  async start() {
    return new Promise((resolve, reject) => {
      const env = {
        ...process.env,
        ...this.serverEnv,
      };

      delete env.MCP_COMMAND;
      delete env.MCP_TOOL_NAME;
      delete env.MCP_TOOL_ARGUMENTS;

      this.process = spawn(this.command, {
        stdio: ["pipe", "pipe", "pipe"],
        env,
        shell: true,
      });

      this.process.on("error", (err) => {
        reject(new Error(`Failed to start MCP server: ${err.message}`));
      });

      this.process.on("exit", (code, signal) => {
        this.processExited = true;
        this.exitCode = code;

        const stderrTail = this.serverStderrLines.slice(-20).join("\n");
        const msg = `MCP server exited with code ${code}${signal ? `, signal ${signal}` : ""}.\nServer output:\n${stderrTail}`;

        for (const [id, { reject: rej }] of this.pendingRequests) {
          rej(new Error(msg));
        }
        this.pendingRequests.clear();

        if (!this.initialized) {
          reject(new Error(msg));
        }
      });

      const rl = readline.createInterface({
        input: this.process.stdout,
        crlfDelay: Infinity,
      });

      rl.on("line", (line) => {
        try {
          const message = JSON.parse(line);
          this.handleMessage(message);
        } catch (e) {
          console.error("[MCP Server]", line);
        }
      });

      this.process.stderr.on("data", (data) => {
        const text = data.toString().trimEnd();
        this.serverStderrLines.push(text);
        console.error("[MCP Server]", text);
      });

      setTimeout(() => {
        if (this.process && !this.process.killed && !this.processExited) {
          resolve();
        }
      }, 500);
    });
  }

  handleMessage(message) {
    if (message.id !== undefined && this.pendingRequests.has(message.id)) {
      const { resolve, reject } = this.pendingRequests.get(message.id);
      this.pendingRequests.delete(message.id);

      if (message.error) {
        reject(
          new Error(message.error.message || JSON.stringify(message.error)),
        );
      } else {
        resolve(message.result);
      }
    }
  }

  sendRequest(method, params = {}) {
    return new Promise((resolve, reject) => {
      const id = ++this.messageId;
      const request = {
        jsonrpc: "2.0",
        id,
        method,
        params,
      };

      this.pendingRequests.set(id, { resolve, reject });
      this.process.stdin.write(JSON.stringify(request) + "\n");
    });
  }

  sendNotification(method, params = {}) {
    const notification = {
      jsonrpc: "2.0",
      method,
      params,
    };
    this.process.stdin.write(JSON.stringify(notification) + "\n");
  }

  async initialize() {
    const result = await Promise.race([
      this.sendRequest("initialize", {
        protocolVersion: MCP_PROTOCOL_VERSION,
        capabilities: {},
        clientInfo: {
          name: "quartermaster-code-runner",
          version: "1.0.0",
        },
      }),
      new Promise((_, reject) =>
        setTimeout(
          () => reject(new Error("Initialize timeout")),
          INITIALIZE_TIMEOUT,
        ),
      ),
    ]);

    // Send initialized notification
    this.sendNotification("notifications/initialized");
    this.initialized = true;

    return result;
  }

  async listTools() {
    const result = await this.sendRequest("tools/list", {});
    return result.tools || [];
  }

  async callTool(name, arguments_) {
    const result = await Promise.race([
      this.sendRequest("tools/call", {
        name,
        arguments: arguments_,
      }),
      new Promise((_, reject) =>
        setTimeout(
          () => reject(new Error("Tool call timeout")),
          TOOL_CALL_TIMEOUT,
        ),
      ),
    ]);

    // Parse content from result
    const content = result.content || [];
    const textParts = [];

    for (const item of content) {
      if (item.type === "text") {
        textParts.push(item.text || "");
      } else if (item.type === "image") {
        textParts.push(`[Image: ${item.mimeType || "unknown"}]`);
      } else if (item.type === "resource") {
        textParts.push(`[Resource: ${item.uri || "unknown"}]`);
      }
    }

    return textParts.join("\n");
  }

  async close() {
    if (this.process && !this.process.killed) {
      this.process.kill();
    }
  }
}

async function main() {
  const command = process.env.MCP_COMMAND;
  const operation = process.env.MCP_OPERATION || "call_tool";
  const toolName = process.env.MCP_TOOL_NAME;
  const toolArgumentsJson = process.env.MCP_TOOL_ARGUMENTS || "{}";

  if (!command) {
    console.error("Error: MCP_COMMAND environment variable is required");
    process.exit(1);
  }

  if (operation === "call_tool" && !toolName) {
    console.error(
      "Error: MCP_TOOL_NAME environment variable is required for call_tool operation",
    );
    process.exit(1);
  }

  let toolArguments;
  try {
    toolArguments = JSON.parse(toolArgumentsJson);
  } catch (e) {
    console.error("Error: MCP_TOOL_ARGUMENTS must be valid JSON");
    process.exit(1);
  }

  // Extract MCP_ENV_* variables for the server
  const serverEnv = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (key.startsWith("MCP_ENV_")) {
      serverEnv[key.substring(8)] = value;
    }
  }

  const client = new McpStdioClient(command, serverEnv);

  try {
    console.error(`Starting MCP server: ${command}`);
    await client.start();

    console.error("Initializing MCP connection...");
    await client.initialize();

    if (operation === "list_tools") {
      console.error("Listing tools...");
      const tools = await client.listTools();
      console.log(JSON.stringify(tools));
    } else {
      console.error(`Calling tool: ${toolName}`);
      const result = await client.callTool(toolName, toolArguments);
      console.log(result);
    }

    await client.close();
    process.exit(0);
  } catch (error) {
    console.error("MCP Error:", error.message);
    if (client.serverStderrLines.length > 0) {
      console.error("--- MCP Server stderr ---");
      for (const line of client.serverStderrLines) {
        console.error(line);
      }
      console.error("--- End server stderr ---");
    }
    await client.close();
    process.exit(1);
  }
}

main();
