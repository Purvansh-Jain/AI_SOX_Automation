# Cloud Streamlit App (self-contained)

This folder contains a full, self-contained Streamlit app ready for local runs and Streamlit Community Cloud.

Key properties:
- No `.env` required. Uses Streamlit secrets (local: `.streamlit/secrets.toml`; Cloud: Settings → Secrets).
- Self-contained: tools, models, utils, prompt are all under `cloud-app/`.
- Uses LangGraph prebuilt agent (fallback to `create_react_agent` if needed) and Azure OpenAI via `st.secrets`.

## Local run

1) Create secrets file `.streamlit/secrets.toml` under `cloud-app/` (we created a template for you):

```
AZURE_OPENAI_API_KEY = "<your-key>"
AZURE_OPENAI_ENDPOINT = "https://<your-endpoint>.openai.azure.com/"

# Optional overrides if your Azure resource uses different names/versions
# AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini"   # or your deployment name
# AZURE_OPENAI_API_VERSION = "2025-01-01-preview"
```

2) Install dependencies and run:

```powershell
cd cloud-app
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

App will open at http://localhost:8501.

## Deploy on Streamlit Cloud

- App file path: `cloud-app/app.py`
- Python dependencies: point to `cloud-app/requirements.txt` in App Settings → Advanced → Python packages file path
- Set Secrets in App → Settings → Secrets:

```
AZURE_OPENAI_API_KEY = "<your-key>"
AZURE_OPENAI_ENDPOINT = "https://<your-endpoint>.openai.azure.com/"

# Optional
# AZURE_OPENAI_DEPLOYMENT = "gpt-4.1"
# AZURE_OPENAI_API_VERSION = "2025-01-01-preview"
```

## Notes
- If you enable embeddings later, add `sentence-transformers` and `torch` to `cloud-app/requirements.txt` (slower builds).
- Azure deployment name is set to `gpt-4.1` and API `2025-01-01-preview`. Ensure your Azure resource has a matching deployment name/version.
