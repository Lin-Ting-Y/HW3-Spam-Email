# Project Context
```markdown
# Project Context

## Purpose
This workspace contains the OpenSpec-driven spec and change-management artifacts for an AI/IoT student project (HW3) focused on integrating LLM-based assistants with IoT and backend services. The repo uses OpenSpec to propose, review, and track changes to capabilities and to keep the "source of truth" for behavioral requirements.

## Tech Stack
- Python 3.12 (recommended on Windows) for ML training/inference (scikit-learn, pandas, numpy, joblib)
- Streamlit for interactive UI and deployment
- Node.js (tested on Node 16+) and npm for OpenSpec CLI tooling
- OpenSpec CLI (`@fission-ai/openspec`) for spec-driven workflow
- Optional: TypeScript/Jest for any Node-based utilities; pytest for Python tests

## Project Conventions

### Code Style
- Use Prettier for formatting and ESLint (recommended) for linting.
- Keep files small and single-purpose. Name functions and modules with clear verb-noun semantics.

### Architecture Patterns
- Modular service boundaries per capability (see `openspec/specs/`): one capability = one focused area of behavior.
- Prefer simple, well-understood patterns (Streamlit apps for interactive inference, single-process workers) unless performance or scale require otherwise.

### Testing Strategy
- Python: unit tests with pytest for data pipeline and model inference; optional notebooks for exploration.
- Node: Jest if building any Node utilities.
- Specs drive acceptance-level scenarios; tests should cover the scenarios in `#### Scenario:` blocks.

### Git Workflow
- Branch from `main` for feature work: `feature/<change-id>` or `changes/<change-id>`.
- Use conventional commits: `feat:`, `fix:`, `chore:`, `docs:`; reference change-id in PR descriptions.
- Open a PR that contains the `openspec/changes/<change-id>/` directory and implementation; request review and approval before merging.

## Domain Context
- This project integrates LLM models (e.g., Claude Sonnet) with IoT or backend services. Model selection and rollout are treated as capabilities managed by OpenSpec.

## Important Constraints
- Development environment: Windows/PowerShell is supported (CI may run Linux).
- Some changes (model enablement) require external provider access or admin console privileges; these cannot be completed purely by repo changes.

## External Dependencies
- PyPI registry for Python packages (scikit-learn, streamlit, etc.)
- npm registry for OpenSpec CLI
- Optional: Anthropic/Claude or other hosted LLM providers if used by other changes

## Assumptions
- Primary implementation languages: Python (ML + Streamlit UI) and Node (tooling via OpenSpec CLI). If you change stacks, update this document accordingly.
- Hosted model enablement (if any) requires provider-side configuration beyond this repo.

```
