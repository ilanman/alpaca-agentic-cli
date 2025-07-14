# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- LangChain integration with enhanced tool calling framework
- New `LangChainTradingAgent` class with improved capabilities
- External data source integration (yfinance, DuckDuckGo search, Wikipedia)
- Enhanced main entry point with simplified interface
- Tool comparison and testing functionality
- Comprehensive CHANGELOG system for tracking changes
- MCP tool wrapper for LangChain compatibility
- Streaming response support in LangChain agent
- Enhanced memory management with conversation buffer
- Better error handling and retry logic

### Changed
- Upgraded agent to use LangChain’s event streaming API for true streaming of LLM and tool output.
- CLI now only shows assistant output, with no extra labels or prefixes, for a cleaner experience.
- Improved error handling and user feedback for failed trades and tool errors.
- Clarified trade status messages (e.g., “Trade request made. Awaiting response...”).
- Cleaned up code: removed unused debugging, test artifacts, and streaming flag logic.

### Removed
- Output labeling such as `[TOOL OUTPUT]` and `[ASSISTANT]` from CLI.
- Manual streaming flag and related CLI/test code.

### Fixed
- N/A

### Security
- N/A

## [Unreleased]
- Streaming output is now always enabled; only the assistant's output is shown in the CLI (no [TOOL OUTPUT] or [ASSISTANT] labels, no raw tool output).
- Trade confirmation step added: users must confirm trades before execution, with a clear prompt and live price lookup.
- Improved error handling and output clarity for failed trades and tool responses.
- Codebase cleanup: removed unused code, debugging statements, and test artifacts; refactored for maintainability.
- Added basic unit tests for MCPToolWrapper, covering parameter validation and trade confirmation logic.

## [0.1.0] - 2024-07-13

### Added
- Initial release
- Alpaca MCP server integration
- OpenAI-powered trading agent
- Basic CLI interface
- Environment variable configuration
- Token usage tracking
- Disposable query support

---

## Release Notes Guidelines

### Version Format
- **Major.Minor.Patch** (e.g., 1.2.3)
- **Major**: Breaking changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, backward compatible

### Change Categories
- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security-related changes

### Commit Messages
Use conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation
- `style:` for formatting changes
- `refactor:` for code refactoring
- `test:` for adding tests
- `chore:` for maintenance tasks

### Examples
```
feat: add LangChain integration for enhanced tool calling
fix: resolve MCP server connection timeout issues
docs: update README with installation instructions
refactor: migrate from custom agent to LangChain framework
``` 