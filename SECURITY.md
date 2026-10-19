# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 1.x | ✅ |

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Use GitHub’s private vulnerability reporting:

1. Go to the repository’s **Security** tab.
2. Click **Report a vulnerability**.
3. Include reproduction steps, affected version, and impact.

If private reporting is unavailable, contact the maintainer through the GitHub
profile [`NOOBHEKER`](https://github.com/NOOBHEKER) and ask for a private
channel.

You can expect an acknowledgement within 7 days. Please allow time for a fix
before public disclosure.

## Scope

Relevant issues include:

- Code execution or path traversal through uploaded CSV files or HGVSp input.
- Leakage of API keys or local file contents.
- Dependency vulnerabilities with a practical exploit path.
- Unsafe deserialization. Note: `data/*.pkl` and `*.pt` are pickle/torch
  artifacts. Only load artifacts from this repository or sources you trust.

## Out of scope

- Incorrect biological predictions. These are research outputs, not a security
  boundary. Report them as regular issues with evidence.
- Vulnerabilities in third-party services (UniProt, NCBI, Hugging Face, NVIDIA).
- Denial of service from intentionally huge local files.

## Secrets

Never commit `NVIDIA_API_KEY`, Hugging Face tokens, or `.env` files. The app
reads `NVIDIA_API_KEY` from the environment only. If you accidentally commit a
secret, rotate it immediately and open a private report.

## Research-use disclaimer

This project is not a medical device and has no clinical security or safety
certification. Do not deploy it in a clinical decision path.
