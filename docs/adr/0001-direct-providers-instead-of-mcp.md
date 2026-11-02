# ADR 0001: Call providers directly instead of through MCP servers

**Status:** accepted. **Context:** v1 exposed weather/search as MCP tools. MCP 2.x changed its server API
(`FastMCP` -> `MCPServer`), and for in-process tools it adds a subprocess, a transport and a failure mode with no
benefit: we need typed results, per-call timeouts, circuit breakers and caching on every tool.
**Decision:** providers are plain async classes behind `Protocol`s (`providers/base.py`) wrapped by one `Toolbelt`.
**Consequences:** simpler, faster, testable with fakes; an MCP façade can be added later to *expose* these tools to
other agents without changing the core.
