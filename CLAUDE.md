# Before Every Response
- NEVER describe how code works without reading it first (Read/Grep). If you didn't use a tool to check, say "I haven't verified this" or check first.
- NEVER use nested/inline imports. ALL imports go at the top of the file. Check your edits for this before submitting.
- NEVER create circular dependencies between modules. If a type hint would cause a circular import, fix the module structure instead of using string annotations.
- Never remove breakpoints or uncomment code that was left commented out.
- No decorative comment banners (e.g., `# ────────`). Use docstrings and clear function names instead.
- Add type hints to all function signatures.

# Rules
- Do what's asked, nothing more/less.
- NEVER add comments about what code used to be or what was moved/removed.
- Follow instructions precisely. If asked to implement but not integrate, don't integrate.
- NEVER use unittest mocks — only mocker fixture.
- Always write vectorized numpy — no Python loops over arrays.
- Keep notebooks simple — short function calls only, all logic in modules.
- No patchwork — design complete algorithms from first principles.
- No fat VM solutions — hard constraint.
- Never create git commits — user commits themselves.
- Never modify user's code without asking first.
- Test code end-to-end before presenting.
- Terse responses — no trailing summaries.
- Debugging: always write debugging code in its own module, and save all data/info/logs needed for thorough analysis without requiring multiple test runs.

# Project Context
Read `pychunkedgraph/debug/stitch_test/SESSION.md` for full stitch redesign context.
