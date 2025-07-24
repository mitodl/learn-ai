const express = require("express")
const { spawn } = require("child_process")
const app = express()

app.use(express.json())
app.use((_, res, next) => {
  res.header("Access-Control-Allow-Origin", "*")
  res.header("Access-Control-Allow-Headers", "Content-Type")
  res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
  next()
})

let mcpProcess = null
let isInitialized = false
const mcpResponses = new Map()

function initializeMCP() {
  return new Promise((resolve, reject) => {
    if (isInitialized) {
      resolve()
      return
    }

    // Validate required environment variables
    if (!process.env.CANVAS_DOMAIN || !process.env.CANVAS_API_TOKEN) {
      reject(
        new Error(
          "Missing required environment variables: CANVAS_DOMAIN and CANVAS_API_TOKEN",
        ),
      )
      return
    }

    console.log("Starting Canvas MCP server...")
    mcpProcess = spawn("npx", ["-y", "canvas-mcp-server"], {
      stdio: ["pipe", "pipe", "pipe"],
      env: {
        ...process.env,
        CANVAS_DOMAIN: process.env.CANVAS_DOMAIN,
        CANVAS_API_TOKEN: process.env.CANVAS_API_TOKEN,
      },
    })

    mcpProcess.stderr.on("data", (data) => {
      console.error("MCP STDERR:", data.toString())
    })

    mcpProcess.stdout.on("data", (data) => {
      const lines = data
        .toString()
        .split("\n")
        .filter((line) => line.trim())
      for (const line of lines) {
        try {
          const parsed = JSON.parse(line)
          console.log("MCP Response:", parsed)
          mcpResponses.set(parsed.id, parsed)
        } catch (e) {
          console.log("MCP Output:", line)
        }
      }
    })

    // Initialize MCP protocol
    const initMessage = {
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: {
        protocolVersion: "2024-11-05",
        capabilities: {},
        clientInfo: { name: "canvas-bridge", version: "1.0.0" },
      },
    }

    console.log("Sending init message:", initMessage)
    mcpProcess.stdin.write(`${JSON.stringify(initMessage)}\n`)

    // Wait for init response
    const checkInit = () => {
      if (mcpResponses.has(1)) {
        const response = mcpResponses.get(1)
        if (response.result) {
          console.log("MCP initialized successfully")
          isInitialized = true
          resolve()
        } else {
          reject(
            new Error(`MCP initialization failed: ${JSON.stringify(response)}`),
          )
        }
      } else {
        setTimeout(checkInit, 100)
      }
    }

    setTimeout(checkInit, 100)

    setTimeout(() => {
      if (!isInitialized) {
        reject(new Error("MCP initialization timeout"))
      }
    }, 10000)
  })
}

// MCP Streamable HTTP endpoints (must come before wildcard routes)
app.post("/mcp", handleMCPRequest)
app.post("/api/canvas/mcp", handleMCPRequest)
app.post("/api/canvas/mcp/", handleMCPRequest)

app.get("/api/canvas/", async (_, res) => {
  try {
    if (!isInitialized) {
      await initializeMCP()
    }

    const messageId = Date.now()
    const message = {
      jsonrpc: "2.0",
      id: messageId,
      method: "tools/list",
      params: {},
    }

    console.log("Listing available tools...")
    mcpProcess.stdin.write(`${JSON.stringify(message)}\n`)

    // Wait for response
    const responsePromise = new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error("Tools list timeout"))
      }, 10000)

      const checkResponse = () => {
        if (mcpResponses.has(messageId)) {
          clearTimeout(timeout)
          const response = mcpResponses.get(messageId)
          mcpResponses.delete(messageId)
          resolve(response)
        } else {
          setTimeout(checkResponse, 100)
        }
      }

      setTimeout(checkResponse, 100)
    })

    const result = await responsePromise
    res.json(result)
  } catch (error) {
    console.error("Error listing tools:", error)
    res.status(500).json({ error: error.message })
  }
})

app.post("/api/canvas/:tool", async (req, res) => {
  try {
    if (!isInitialized) {
      await initializeMCP()
    }

    const toolName = req.params.tool
    const params = req.body || {}
    const messageId = Date.now()

    const message = {
      jsonrpc: "2.0",
      id: messageId,
      method: "tools/call",
      params: {
        name: toolName,
        arguments: params,
      },
    }

    console.log("Calling tool:", toolName, "with params:", params)

    mcpProcess.stdin.write(`${JSON.stringify(message)}\n`)

    // Wait for response
    const responsePromise = new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error("Tool call timeout"))
      }, 30000)

      const checkResponse = () => {
        if (mcpResponses.has(messageId)) {
          clearTimeout(timeout)
          const response = mcpResponses.get(messageId)
          mcpResponses.delete(messageId)
          resolve(response)
        } else {
          setTimeout(checkResponse, 100)
        }
      }

      setTimeout(checkResponse, 100)
    })

    const result = await responsePromise
    res.json(result)
  } catch (error) {
    console.error("Error calling tool:", error)
    res.status(500).json({ error: error.message })
  }
})

async function handleMCPRequest(req, res) {
  try {
    if (!isInitialized) {
      await initializeMCP()
    }

    const message = req.body
    console.log("MCP Request:", message)

    // Handle different MCP methods
    if (message.method === "notifications/initialized") {
      // This is a notification, no response needed
      res.status(200).send()
    } else if (message.method === "initialize") {
      // Return server capabilities
      const response = {
        jsonrpc: "2.0",
        id: message.id,
        result: {
          protocolVersion: "2024-11-05",
          capabilities: {
            tools: {},
          },
          serverInfo: {
            name: "canvas-mcp-bridge",
            version: "1.0.0",
          },
        },
      }
      res.json(response)
    } else if (message.method === "tools/list") {
      // Forward to MCP server
      const messageId = Date.now()
      const mcpMessage = {
        jsonrpc: "2.0",
        id: messageId,
        method: "tools/list",
        params: {},
      }

      mcpProcess.stdin.write(`${JSON.stringify(mcpMessage)}\n`)

      const responsePromise = new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error("Tools list timeout"))
        }, 10000)

        const checkResponse = () => {
          if (mcpResponses.has(messageId)) {
            clearTimeout(timeout)
            const response = mcpResponses.get(messageId)
            mcpResponses.delete(messageId)
            resolve(response)
          } else {
            setTimeout(checkResponse, 100)
          }
        }

        setTimeout(checkResponse, 100)
      })

      const result = await responsePromise

      // Return with original request ID
      const response = {
        jsonrpc: "2.0",
        id: message.id,
        result: result.result,
      }
      res.json(response)
    } else if (message.method === "tools/call") {
      // Forward tool call to MCP server
      const messageId = Date.now()
      const mcpMessage = {
        jsonrpc: "2.0",
        id: messageId,
        method: "tools/call",
        params: message.params,
      }

      console.log("Forwarding tool call:", mcpMessage)
      mcpProcess.stdin.write(`${JSON.stringify(mcpMessage)}\n`)

      const responsePromise = new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error("Tool call timeout"))
        }, 30000)

        const checkResponse = () => {
          if (mcpResponses.has(messageId)) {
            clearTimeout(timeout)
            const response = mcpResponses.get(messageId)
            mcpResponses.delete(messageId)
            resolve(response)
          } else {
            setTimeout(checkResponse, 100)
          }
        }

        setTimeout(checkResponse, 100)
      })

      const result = await responsePromise

      // Return with original request ID
      const response = {
        jsonrpc: "2.0",
        id: message.id,
        result: result.result,
        error: result.error,
      }
      res.json(response)
    } else {
      // Unknown method
      const response = {
        jsonrpc: "2.0",
        id: message.id,
        error: {
          code: -32601,
          message: "Method not found",
        },
      }
      res.json(response)
    }
  } catch (error) {
    console.error("MCP endpoint error:", error)
    const response = {
      jsonrpc: "2.0",
      id: req.body.id,
      error: {
        code: -32603,
        message: error.message,
      },
    }
    res.status(500).json(response)
  }
}

app.get("/health", (_, res) => {
  res.json({ status: "ok", initialized: isInitialized })
})

const PORT = process.env.PORT || 3000
app.listen(PORT, "0.0.0.0", () => {
  console.log(`Canvas MCP Bridge listening on port ${PORT}`)
  console.log(`MCP streamable_http endpoint: http://0.0.0.0:${PORT}/mcp`)
})
