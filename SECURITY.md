# API Security Protocol

This document outlines the security measures implemented to protect API keys and sensitive data within the Beauty Insights Suite application.

## 1. Secret Management

All API keys, including the Google YouTube Data API Key and the Web3.Storage API Token, are managed as secrets and are **never** hardcoded in the source code.

- **Local Development**: Secrets are stored in a `.env` file in the project root. This file is explicitly excluded from version control via `.gitignore`.
- **Deployment**: Secrets are managed using Streamlit's built-in secrets management (`.streamlit/secrets.toml`) or the hosting provider's environment variable settings. The `secrets.toml` file is also excluded from version control.

## 2. Backend-Only API Calls

All API calls to external services that require authentication (e.g., Google, Web3.Storage) are performed exclusively on the backend (server-side within the Streamlit application).

- **Proxying**: The frontend UI does not make direct calls to these services. Instead, it sends requests to the Streamlit backend, which then uses the securely stored API keys to make the external calls.
- **No Exposure**: This ensures that API keys are never exposed to the client-side (browser), preventing them from being intercepted or extracted.

## 3. Key Rotation

Any previously exposed keys have been invalidated and rotated. A process for regular key rotation should be established for long-term maintenance.

## 4. How to Use

1.  **Local**: Copy `.env.example` to `.env` and add your keys.
2.  **Deployment**: Add the secrets to your Streamlit Cloud account under the app's settings.
