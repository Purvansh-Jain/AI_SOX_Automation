# SOX Automation Cloud App 🌐

**Production-ready Streamlit Cloud deployment for SOX compliance automation**

## 📋 Overview

The Cloud App is the **production version** of the SOX Automation Assistant, optimized for deployment on Streamlit Cloud. It provides AI-powered SOX compliance analysis using Azure OpenAI with minimal dependencies and cloud-native configuration.

### Key Features

- ✅ **AI-Powered Control Matching** - Uses Azure OpenAI GPT-4 for semantic control analysis
- ✅ **Intelligent Column Mapping** - Automatic mapping of uploaded file columns to required schema
- ✅ **Executive Summary Generation** - AI-generated compliance reports
- ✅ **Anomaly Detection** - Identifies scope mismatches and control gaps
- ✅ **Cloud-Native** - Designed for Streamlit Cloud with secrets management
- ✅ **Minimal Dependencies** - Lightweight, API-only approach (no local ML models)

## 🏗️ Architecture

```
cloud-app/
├── app.py                          # Main Streamlit application
├── tools.py                        # AI agent tools and control matching logic
├── models.py                       # Data models and state management
├── utils.py                        # Utility functions
├── intelligent_column_mapper.py    # Heuristic-based column mapper
├── prompt.txt                      # AI agent system prompt
├── requirements.txt                # Python dependencies
├── .streamlit/
│   └── secrets.toml               # Streamlit Cloud secrets (not in git)
└── Files/                         # Sample data files
    ├── RCM 2.csv
    ├── RCM 2.xlsx
    ├── TB 1 (1).csv
    └── TB 1 (1).xlsx
```

## 🚀 Deployment to Streamlit Cloud

### Prerequisites

- Azure OpenAI account with GPT-4 deployment
- Streamlit Cloud account
- GitHub repository

### Step 1: Configure Secrets

In your Streamlit Cloud app settings, add these secrets:

```toml
# .streamlit/secrets.toml (in Streamlit Cloud dashboard)

AZURE_OPENAI_API_KEY = "your-azure-openai-api-key"
AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com"
AZURE_OPENAI_DEPLOYMENT = "gpt-4.1"  # Optional, defaults to "gpt-4.1"
AZURE_OPENAI_API_VERSION = "2025-01-01-preview"  # Optional
```

### Step 2: Deploy

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `cloud-app/app.py` as the main file
5. Configure secrets in the app settings
6. Deploy!

### Step 3: Test

Upload your RCM and Trial Balance files, then interact with the AI assistant to:
- Map file columns automatically
- Run SOX automation
- Generate executive summaries
- Detect anomalies and control gaps

## 🔧 Local Development (Optional)

If you want to test locally before deploying:

```bash
# Navigate to cloud-app directory
cd cloud-app

# Install dependencies
pip install -r requirements.txt

# Create .streamlit/secrets.toml with your credentials
mkdir .streamlit
# Add your secrets to .streamlit/secrets.toml

# Run the app
streamlit run app.py
```

## 📦 Dependencies

The cloud app uses **minimal dependencies** for faster deployment:

- **Streamlit** - Web framework
- **Pandas/NumPy** - Data processing
- **OpenPyXL/XlsxWriter** - Excel file handling
- **Matplotlib/Plotly** - Visualizations
- **LangChain** - AI agent framework
- **Azure OpenAI** - GPT-4 API access

**No local ML models required** - Everything runs through Azure OpenAI API.

## 🤖 How It Works

### Control Matching Strategy

The cloud app uses **GPT-4 semantic matching** for control analysis:

1. **GPT-4 Analysis** - Sends control descriptions to GPT-4 for relevance scoring
2. **Fallback Synonyms** - Uses keyword matching if API is unavailable
3. **Hybrid Results** - Combines both approaches for comprehensive coverage

### Column Mapping

Uses **deterministic heuristics**:
- Normalizes column names (remove spaces, lowercase)
- Matches common synonyms (e.g., "BU" → "Entity", "Brand" → "Entity")
- Fuzzy matching for similar names

## 🔐 Security

- ✅ Secrets managed via Streamlit Cloud (never in code)
- ✅ No `.env` files - cloud-native configuration
- ✅ API keys never logged or exposed
- ✅ File uploads processed in memory

## 🆚 Cloud App vs AI App

| Feature | Cloud App | AI App |
|---------|-----------|--------|
| **Deployment** | Streamlit Cloud | Local development |
| **Configuration** | `st.secrets` | `.env` file |
| **Control Matching** | GPT-4 API | Local embeddings + GPT-4 |
| **Column Mapping** | Heuristics | GPT-4 powered |
| **ML Models** | None (API only) | sentence-transformers |
| **Dependencies** | ~20 packages | ~180 packages |
| **Use Case** | Production | Development/Testing |

## 🐛 Troubleshooting

### "Azure OpenAI secrets are missing"
- Ensure secrets are configured in Streamlit Cloud dashboard
- Check secret names match exactly: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`

### "File not found: prompt.txt"
- Verify `prompt.txt` is in the `cloud-app/` directory
- Check file path in `app.py` line 90

### "Control matching failed"
- Verify Azure OpenAI deployment name is correct
- Check API key has permissions for GPT-4 deployment
- Review rate limits on your Azure OpenAI resource

## 📊 Expected Output

After running automation, you'll get:
- **Excel Report** - `Final_Automation_Report.xlsx` with multiple sheets
- **PDF Report** - `Automation_Report.pdf` with visualizations
- **Executive Summary** - AI-generated compliance overview
- **Control Mappings** - Matched controls by account type
- **Anomaly Report** - Flagged issues and recommendations

## 📞 Support

For issues or questions:
1. Check the [main project README](../README.md)
2. Review error logs in Streamlit Cloud dashboard
3. Verify Azure OpenAI quota and rate limits

## 📄 License

Part of the SOX Automation project. See main repository for license details.

---

**Last Updated:** October 30, 2025  
**Version:** 2.0 (Cloud-Optimized)
