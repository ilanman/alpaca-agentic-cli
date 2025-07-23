# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Fixed import errors related to project restructuring
- Reverted project structure to original layout to resolve module path issues
- Improved stability and error handling in agent initialization

### Changed
- Updated import paths in agent code to match restored project structure
- Cleaned up temporary restructuring changes

### Fixed
- Resolved ModuleNotFoundError for 'src.chat.mcp_client' by restoring original directory layout

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