import os
import streamlit as st
from pathlib import Path

# Strictly use Streamlit Cloud secrets (no .env)
API_KEY = st.secrets.get("AZURE_OPENAI_API_KEY")
ENDPOINT = st.secrets.get("AZURE_OPENAI_ENDPOINT")

if not API_KEY or not ENDPOINT:
    st.error("Azure OpenAI secrets are missing. Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT in Streamlit Cloud Secrets.")
    st.stop()

# Also export to environment so helper tools can read them when needed
os.environ["AZURE_OPENAI_API_KEY"] = API_KEY
os.environ["AZURE_OPENAI_ENDPOINT"] = ENDPOINT
# Also mirror deployment name and API version for downstream helpers
if st.secrets.get("AZURE_OPENAI_DEPLOYMENT"):
    os.environ["AZURE_OPENAI_DEPLOYMENT"] = st.secrets.get("AZURE_OPENAI_DEPLOYMENT")
if st.secrets.get("AZURE_OPENAI_API_VERSION"):
    os.environ["AZURE_OPENAI_API_VERSION"] = st.secrets.get("AZURE_OPENAI_API_VERSION")

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# Agent helper import (prefer new location, fallback to older ones)
try:
    from langchain.agents import create_agent  # New recommended import
except Exception:
    try:
        from langgraph.prebuilt import create_agent
    except Exception:
        from langgraph.prebuilt import create_react_agent as create_agent

from langgraph.checkpoint.memory import InMemorySaver

# Import local cloud-app tools and models
from tools import (
    update_rcm_file_path,
    update_trail_balance_file_path,
    get_rcm_file_path,
    get_tb_file_path,
    analyze_data,
    get_columns,
    run_sox_automation,
    suggest_control_mappings,
    detect_anomalies,
    generate_executive_summary,
    discover_relevant_controls,
    get_flag_counts_by_account_type,
    list_in_scope_entities,
    qa_results,
)
from models import AnalystState
import pandas as pd

# Model
# Allow overriding deployment config via secrets without changing code
AZURE_DEPLOYMENT_NAME = st.secrets.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
AZURE_API_VERSION = st.secrets.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

model = AzureChatOpenAI(
    azure_deployment=AZURE_DEPLOYMENT_NAME,
    api_version=AZURE_API_VERSION,
    temperature=0.1,
    azure_endpoint=ENDPOINT,
    api_key=API_KEY,
)

checkpointer = InMemorySaver()

# Load master prompt from local cloud-app/prompt.txt
PROMPT_PATH = Path(__file__).resolve().parent / "prompt.txt"
if not PROMPT_PATH.exists():
    st.error("prompt.txt not found in cloud-app/")
    st.stop()

system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

# Tools
TOOLS = [
    get_rcm_file_path, update_rcm_file_path, get_tb_file_path,
    update_trail_balance_file_path, analyze_data, get_columns,
    run_sox_automation, suggest_control_mappings, detect_anomalies,
    generate_executive_summary, discover_relevant_controls,
    get_flag_counts_by_account_type, list_in_scope_entities, qa_results,
]

config = {"configurable": {"thread_id": "1"}}

agent_executor = create_agent(
    model=model,
    tools=TOOLS,
    checkpointer=checkpointer,
    prompt=system_prompt,
    state_schema=AnalystState,
)

st.set_page_config(page_title="AI Sox Automation Assistant (Cloud)", page_icon="🤖", layout="wide")

# Custom CSS
st.markdown(
    """
<style>
    .stProgress .st-bo {background-color: #00c853;}
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px; border-radius: 10px; color: white; text-align: center; }
    .status-badge { display: inline-block; padding: 5px 12px; border-radius: 15px;
        font-size: 12px; font-weight: bold; }
    .status-success {background-color: #00c853; color: white;}
    .status-warning {background-color: #ffa726; color: white;}
    .status-error {background-color: #ef5350; color: white;}
    .ai-insight { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 15px; border-radius: 8px; color: white; margin: 10px 0; }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "analyst_state" not in st.session_state:
    st.session_state["analyst_state"] = AnalystState(
        messages=[],
        automation_excel_path=None,
        automation_pdf_path=None,
        rcm_file_path=None,
        trail_balance_file_path=None,
        rcm_column_mapping=None,
    )

if "workflow_stage" not in st.session_state:
    st.session_state["workflow_stage"] = "init"
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = {"rcm": None, "tb": None}
if "validation_errors" not in st.session_state:
    st.session_state["validation_errors"] = {"rcm": None, "tb": None}
if "analysis_results" not in st.session_state:
    st.session_state["analysis_results"] = None
if "show_ai_insights" not in st.session_state:
    st.session_state["show_ai_insights"] = False


def _sanitize_df_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == object:
            sample_types = set(type(x) for x in df_copy[col].dropna().head(100).tolist())
            non_str_types = {t for t in sample_types if t is not str}
            if non_str_types:
                def _safe_to_str(x):
                    try:
                        if isinstance(x, (bytes, bytearray)):
                            return x.decode(errors="replace")
                        if pd.isna(x):
                            return pd.NA
                        return str(x)
                    except Exception:
                        return str(x)

                df_copy[col] = df_copy[col].apply(_safe_to_str)
    return df_copy


def handle_rcm_upload():
    rcm_file = st.session_state.get("rcm_upload")
    if rcm_file is None:
        return
    file_path = f"uploaded_rcm_{rcm_file.name}"
    with open(file_path, "wb") as f:
        f.write(rcm_file.getbuffer())
    try:
        df = pd.read_excel(file_path) if file_path.endswith(".xlsx") else pd.read_csv(file_path)
        required_columns = ["Control Description", "Control ID", "Entity", "Key? (Y/N)"]
        used_ai = False
        mapped_cols = {}
        try:
            from intelligent_column_mapper import IntelligentColumnMapper  # optional
            mapper = IntelligentColumnMapper()
            mapped_cols = mapper.map_columns(df.columns.tolist(), required_columns, "RCM")
            used_ai = True
        except Exception:
            def _normalize(s: str) -> str:
                return "".join(ch for ch in str(s).lower() if ch.isalnum())
            uploaded_norm = {_normalize(c): c for c in df.columns}
            aliases = {
                "entity": ["brandname", "brand", "bu", "businessunit", "division", "entity", "entityname"],
                "controldescription": ["controldesc", "controldescription", "description", "desc", "controldetails"],
                "controlid": ["ctrlid", "controlid", "controlnumber", "id", "control#"],
                "keyyn": ["keycontrol", "key", "iskey", "keyyn", "keyy/n"],
            }
            for req in required_columns:
                key = _normalize(req)
                found = uploaded_norm.get(key)
                if not found and key in aliases:
                    for a in aliases[key]:
                        if a in uploaded_norm:
                            found = uploaded_norm[a]
                            break
                if not found:
                    for k, v in uploaded_norm.items():
                        if key in k or k in key:
                            found = v
                            break
                mapped_cols[req] = found

        rename_dict = {v: k for k, v in mapped_cols.items() if v}
        if rename_dict:
            df = df.rename(columns=rename_dict)
            if file_path.lower().endswith(".xlsx"):
                df.to_excel(file_path, index=False)
            else:
                df.to_csv(file_path, index=False)

        st.session_state["analyst_state"]["rcm_column_mapping"] = {
            "standard_name": list(mapped_cols.keys()),
            "original_name": [mapped_cols[k] if mapped_cols[k] else "(not found)" for k in mapped_cols.keys()],
            "mapping_method": [
                ("AI-Detected" if (mapped_cols[k] and used_ai) else ("Heuristic" if mapped_cols[k] else "N/A"))
                for k in mapped_cols.keys()
            ],
        }

        required_core = ["Control Description", "Control ID", "Entity"]
        missing_core = [c for c in required_core if c not in df.columns]
        if missing_core:
            st.session_state["validation_errors"]["rcm"] = missing_core
        else:
            if "Key? (Y/N)" not in df.columns:
                st.session_state["validation_errors"]["rcm"] = ["Key? (Y/N)"]
            else:
                st.session_state["validation_errors"]["rcm"] = None
            st.session_state["uploaded_files"]["rcm"] = file_path
            st.session_state["analyst_state"]["rcm_file_path"] = file_path
            st.session_state["rcm_rows"] = len(df)
            st.session_state["rcm_df"] = df
    except Exception as e:
        st.session_state["validation_errors"]["rcm"] = ["__file_error__"]
        st.session_state["rcm_error"] = str(e)


def handle_tb_upload():
    tb_file = st.session_state.get("tb_upload")
    if tb_file is None:
        return
    file_path = f"uploaded_tb_{tb_file.name}"
    with open(file_path, "wb") as f:
        f.write(tb_file.getbuffer())
    try:
        df = pd.read_excel(file_path) if file_path.endswith(".xlsx") else pd.read_csv(file_path)
        df = df.map(lambda x: str(x).replace("\xa0", "").strip() if isinstance(x, str) else x)
        df = df.replace(r"^\s*$", pd.NA, regex=True)
        df = df.convert_dtypes()
        required_tb_cols = ["Account Type"]
        missing_tb = [c for c in required_tb_cols if c not in df.columns]
        if missing_tb:
            st.session_state["validation_errors"]["tb"] = missing_tb
        else:
            st.session_state["validation_errors"]["tb"] = None
            st.session_state["uploaded_files"]["tb"] = file_path
            st.session_state["analyst_state"]["trail_balance_file_path"] = file_path
            st.session_state["tb_rows"] = len(df)
            st.session_state["tb_df"] = df
    except Exception as e:
        st.session_state["validation_errors"]["tb"] = ["__file_error__"]
        st.session_state["tb_error"] = str(e)


def render_file_upload_section():
    st.markdown("### 📁 Upload Your Files")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📊 RCM File**")
        st.markdown(
            "<small>Required columns: Control Description, Control ID, Entity, Key?(Y/N)</small>",
            unsafe_allow_html=True,
        )
        st.file_uploader(
            "Upload RCM",
            type=["xlsx", "csv"],
            key="rcm_upload",
            label_visibility="collapsed",
            on_change=handle_rcm_upload,
        )
        if st.session_state.get("rcm_error"):
            st.error(f"Error loading file: {st.session_state['rcm_error']}")
            st.session_state["rcm_error"] = None
        elif st.session_state["uploaded_files"].get("rcm"):
            rcm_df = st.session_state.get("rcm_df")
            rcm_rows = st.session_state.get("rcm_rows", 0)
            st.info("🤖 Using AI to intelligently map column names...")
            if "rcm_column_mapping" in st.session_state["analyst_state"]:
                mapping = st.session_state["analyst_state"]["rcm_column_mapping"]
                with st.expander("📋 Column Mapping (AI-detected)", expanded=True):
                    mapping_text = "Column Mapping:\n"
                    for std, orig, method in zip(
                        mapping["standard_name"], mapping["original_name"], mapping["mapping_method"]
                    ):
                        if orig != "(not found)":
                            mapping_text += f"  ✓ '{std}' ← '{orig}'\n"
                        else:
                            mapping_text += f"  ✗ '{std}' ← {orig}\n"
                    st.code(mapping_text)
            rcm_errors = st.session_state["validation_errors"].get("rcm")
            if rcm_errors and rcm_errors != ["Key? (Y/N)"]:
                st.error(f"RCM file missing required columns: {', '.join(rcm_errors)}")
                if rcm_df is not None:
                    with st.expander("Preview Data (File invalid)"):
                        preview_df = rcm_df.head(10).copy()
                        preview_df.index = range(1, len(preview_df) + 1)
                        st.dataframe(_sanitize_df_for_streamlit(preview_df), width='stretch')
            else:
                if rcm_errors == ["Key? (Y/N)"]:
                    st.warning(
                        "RCM file does not include a Key column; proceeding but you may need to review Key flags."
                    )
                st.success(f"✓ Loaded {rcm_rows} rows")
                if rcm_df is not None:
                    with st.expander("Preview Data"):
                        preview_df = rcm_df.head(10).copy()
                        preview_df.index = range(1, len(preview_df) + 1)
                        st.dataframe(_sanitize_df_for_streamlit(preview_df), width='stretch')

    with col2:
        st.markdown("**💰 Trial Balance File**")
        st.markdown("<small>First column must be 'Account Type'</small>", unsafe_allow_html=True)
        st.file_uploader(
            "Upload Trial Balance",
            type=["xlsx", "csv"],
            key="tb_upload",
            label_visibility="collapsed",
            on_change=handle_tb_upload,
        )
        if st.session_state.get("tb_error"):
            st.error(f"Error loading file: {st.session_state['tb_error']}")
            st.session_state["tb_error"] = None
        elif st.session_state["uploaded_files"].get("tb"):
            tb_df = st.session_state.get("tb_df")
            tb_rows = st.session_state.get("tb_rows", 0)
            tb_errors = st.session_state["validation_errors"].get("tb")
            if tb_errors:
                st.error(f"Trial Balance missing required columns: {', '.join(tb_errors)}")
                if tb_df is not None:
                    with st.expander("Preview Data (File invalid)"):
                        preview_df = tb_df.head(10).copy()
                        preview_df.index = range(1, len(preview_df) + 1)
                        st.dataframe(_sanitize_df_for_streamlit(preview_df), width='stretch')
            else:
                st.success(f"✓ Loaded {tb_rows} rows")
                if tb_df is not None:
                    with st.expander("Preview Data"):
                        preview_df = tb_df.head(10).copy()
                        preview_df.index = range(1, len(preview_df) + 1)
                        st.dataframe(_sanitize_df_for_streamlit(preview_df), width='stretch')

    if st.session_state["uploaded_files"].get("rcm") and st.session_state["uploaded_files"].get("tb"):
        st.success("✓ Both files loaded! Ready for AI-powered analysis.")
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            if st.button("➡️ Proceed", key="proceed_main", type="primary"):
                st.session_state["workflow_stage"] = "files_uploaded"
                st.rerun()


def render_configuration_section():
    st.markdown("### ⚙️ Configure Analysis Parameters")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**Threshold Percentage**")
        st.markdown(
            "<small>Determines cumulative % for 'In Scope' classification</small>",
            unsafe_allow_html=True,
        )
        threshold_percent = st.slider(
            "Threshold",
            min_value=50,
            max_value=100,
            value=80,
            step=5,
            format="%d%%",
            label_visibility="collapsed",
        )
        threshold = threshold_percent / 100.0
        st.info(
            f"📊 Entities contributing to {threshold:.0%} cumulative value will be marked as 'In Scope'"
        )
    with col2:
        st.metric(label="Selected Threshold", value=f"{threshold:.0%}", delta="Recommended: 75-85%")
    try:
        tb_path = st.session_state["analyst_state"]["trail_balance_file_path"]
        if tb_path:
            df = pd.read_excel(tb_path) if tb_path.endswith(".xlsx") else pd.read_csv(tb_path)
            account_types = sorted(df["Account Type"].dropna().unique())
            st.markdown("**Account Types to Process**")
            select_all = st.checkbox("Select All Account Types")
            selected_accounts = st.multiselect(
                "Select account types (leave empty for all)",
                options=account_types,
                default=account_types if select_all else None,
                label_visibility="collapsed",
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🚀 Start Analysis", type="primary"):
                    st.session_state["workflow_stage"] = "analyzing"
                    st.session_state["threshold"] = threshold
                    st.session_state["selected_accounts"] = selected_accounts if selected_accounts else None
                    st.rerun()
            with c2:
                if st.button("🔍 Detect Anomalies First"):
                    st.session_state["show_ai_insights"] = True
                    st.rerun()
    except Exception as e:
        st.error(f"Error reading trial balance: {str(e)}")


def render_analysis_progress():
    st.markdown("### 🔄 Running SOX Automation with AI Analysis")
    progress_bar = st.progress(0)
    status_text = st.empty()
    return progress_bar, status_text


def render_results_dashboard(summary_data):
    st.markdown("### ✅ Analysis Complete")
    if summary_data is None:
        summary_data = {
            "total_account_types": 0,
            "total_entities": 0,
            "in_scope_entities": 0,
            "total_flags": 0,
            "critical_flags": 0,
            "controls_mapped": 0,
        }
        st.warning("⚠️ Analysis metrics not available. Using default values.")
    c1, c2, c3, c4 = st.columns(4)
    total_accounts = summary_data.get("total_account_types", 0)
    controls_mapped = summary_data.get("controls_mapped", 0)
    total_flags = summary_data.get("total_flags", 0)
    critical_flags = summary_data.get("critical_flags", 0)
    in_scope_pct = (
        summary_data.get("in_scope_entities", 0)
        / max(summary_data.get("total_entities", 1), 1)
        * 100
    )
    with c1:
        st.metric("Account Types", total_accounts)
    with c2:
        st.metric("Controls Mapped", controls_mapped)
    with c3:
        st.metric("Flags Raised", total_flags, delta=f"🔴 {critical_flags} critical")
    with c4:
        st.metric("In Scope", f"{in_scope_pct:.0f}%", delta=f"{in_scope_pct:.0f}%")
    if critical_flags > 0 or total_flags > 5:
        st.markdown(
            """
        <div class="ai-insight">
            <strong>🤖 AI Insight:</strong> I've detected control coverage gaps that require attention.
        </div>
        """,
            unsafe_allow_html=True,
        )


# Sidebar workflow and file status
with st.sidebar:
    st.title("Workflow Progress")
    stages = {
        "init": ("📋", "Initial Setup"),
        "files_uploaded": ("📁", "Files Loaded"),
        "configured": ("⚙️", "Configuration Set"),
        "analyzing": ("🔄", "Running Analysis"),
        "complete": ("✅", "Complete"),
    }
    current_stage = st.session_state["workflow_stage"]
    stage_list = list(stages.keys())
    current_index = stage_list.index(current_stage)
    for idx, (stage_key, (icon, label)) in enumerate(stages.items()):
        if idx < current_index:
            st.markdown(f"{icon} ~~{label}~~ ✓")
        elif idx == current_index:
            st.markdown(f"**{icon} {label}** ⬅️")
        else:
            st.markdown(f"{icon} {label}")
    st.divider()
    st.subheader("📁 File Status")
    rcm_ok = st.session_state["uploaded_files"].get("rcm") and st.session_state["validation_errors"].get("rcm") in (
        None,
        [],
        ["Key? (Y/N)"],
    )
    tb_ok = st.session_state["uploaded_files"].get("tb") and st.session_state["validation_errors"].get("tb") in (
        None,
        [],
    )
    if rcm_ok:
        st.success("✅ RCM File Loaded")
    else:
        st.info("⏳ RCM File Pending")

    if tb_ok:
        st.success("✅ Trial Balance Loaded")
    else:
        st.info("⏳ Trial Balance Pending")
    st.divider()


# Display chat history
for msg in st.session_state["analyst_state"]["messages"] or []:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    avatar = "👤" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg.content)


# Main workflow
if st.session_state["workflow_stage"] == "init":
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(
            """
        Welcome! I'm your **AI-powered SOX Automation Assistant**.

        I combine proven compliance logic with artificial intelligence to provide:

        ✅ Automated Analysis — Rule-based SOX scope determination  
        🤖 Smart Suggestions — AI-powered control mapping recommendations  
        🔍 Anomaly Detection — Pattern recognition across your audit data  
        📊 Executive Summaries — Natural language reports for stakeholders  
        💬 Conversational Interface — Ask me anything about your audit

        Let's start by uploading your files:
        """
        )
        render_file_upload_section()

elif st.session_state["workflow_stage"] == "files_uploaded":
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown("Files validated! Configure your analysis parameters:")
        render_configuration_section()
        if st.session_state.get("show_ai_insights"):
            with st.spinner("🤖 AI analyzing your data for anomalies..."):
                state = st.session_state["analyst_state"]
                state["messages"].append(HumanMessage(content="Detect anomalies in my data"))
                for event in agent_executor.stream(state, config=config, stream_mode="updates"):
                    if "agent" in event:
                        agent = event["agent"]
                        if "messages" in agent:
                            msg = agent["messages"][-1]
                            if isinstance(msg, AIMessage) and msg.content:
                                st.markdown(msg.content)
                                state["messages"].append(msg)
                st.session_state["show_ai_insights"] = False

elif st.session_state["workflow_stage"] == "analyzing":
    with st.chat_message("assistant", avatar="🤖"):
        progress_bar, status_text = render_analysis_progress()
        state = st.session_state["analyst_state"]
        threshold = st.session_state.get("threshold", 0.8)
        selected_accounts = st.session_state.get("selected_accounts")
        try:
            progress_bar.progress(10)
            status_text.markdown("⚙️ Initializing AI-powered automation...")
            user_msg = f"Run SOX automation with threshold {threshold:.0%}"
            if selected_accounts:
                user_msg += f" for: {', '.join(selected_accounts)}"
            state["messages"].append(HumanMessage(content=user_msg))
            progress_bar.progress(30)
            status_text.markdown("📊 Analyzing account types and calculating scope...")
            for event in agent_executor.stream(state, config=config, stream_mode="updates"):
                if "agent" in event:
                    agent = event["agent"]
                    if "messages" in agent:
                        msg = agent["messages"][-1]
                        if isinstance(msg, AIMessage) and msg.content:
                            status_text.markdown(f"🤖 {msg.content[:100]}...")
                if "tools" in event:
                    tool_call = event["tools"]
                    progress_bar.progress(50)
                    status_text.markdown("🗺️ AI mapping controls to entities...")
                    for key in ["automation_excel_path", "automation_pdf_path"]:
                        if key in tool_call:
                            state[key] = tool_call[key]
                            if key == "automation_excel_path":
                                progress_bar.progress(80)
                                status_text.markdown("📄 Generating Excel report...")
                            elif key == "automation_pdf_path":
                                progress_bar.progress(90)
                                status_text.markdown("📈 Creating visualization charts...")
            if state.get("automation_excel_path"):
                try:
                    excel_path = state["automation_excel_path"]
                    summary_df = pd.read_excel(excel_path, sheet_name="ALL_AccountType_Summary")
                    total_entities = summary_df["Entity"].nunique()
                    in_scope_entities = (
                        summary_df[summary_df["Scope"] == "In Scope"]["Entity"].nunique()
                    )
                    total_accounts = summary_df["Account Type"].nunique()
                    total_flags = summary_df["Flag - Manual Auditor Check"].astype(str).str.len().gt(0).sum()
                    critical_flags = summary_df["Flag - Manual Auditor Check"].astype(str).str.contains(
                        "In Scope & not Mapped", na=False
                    ).sum()
                    try:
                        rcm_df = pd.read_excel(excel_path, sheet_name="ALL_RCM_Combined")
                        controls_mapped = rcm_df["Control ID"].nunique()
                    except Exception:
                        controls_mapped = 0
                    st.session_state["analysis_results"] = {
                        "total_account_types": total_accounts,
                        "total_entities": total_entities,
                        "in_scope_entities": in_scope_entities,
                        "total_flags": int(total_flags),
                        "critical_flags": int(critical_flags),
                        "controls_mapped": int(controls_mapped),
                    }
                    try:
                        from utils import save_executive_summary
                        save_executive_summary(st.session_state["analysis_results"])
                    except Exception as summary_error:
                        st.warning(f"Could not auto-generate executive summary: {summary_error}")
                except Exception as e:
                    st.warning(f"Could not extract metrics: {e}")
                    st.session_state["analysis_results"] = {
                        "total_account_types": 0,
                        "total_entities": 0,
                        "in_scope_entities": 0,
                        "total_flags": 0,
                        "critical_flags": 0,
                        "controls_mapped": 0,
                    }
            progress_bar.progress(100)
            status_text.markdown("✅ Analysis complete! Generating AI insights...")
            st.session_state["workflow_stage"] = "complete"
            st.rerun()
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            st.session_state["workflow_stage"] = "configured"

elif st.session_state["workflow_stage"] == "complete":
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown("Analysis complete! Here are your results:")
        summary_data = st.session_state.get("analysis_results", {})
        render_results_dashboard(summary_data)
        state = st.session_state["analyst_state"]
        if state.get("automation_excel_path"):
            excel_path_preview = state["automation_excel_path"]
            st.markdown("### 🧾 Generated report preview")
            with st.expander(
                "View ALL_AccountType_Summary (Account Type TB Summary) - paginated (50 rows)",
                expanded=True,
            ):
                try:
                    df_summary_all = pd.read_excel(
                        excel_path_preview, sheet_name="ALL_AccountType_Summary"
                    )
                    if "Account Type" in df_summary_all.columns:
                        acct_types = [
                            "All",
                        ] + sorted(
                            df_summary_all["Account Type"].dropna().astype(str).unique().tolist()
                        )
                        sel_acct = st.selectbox(
                            "Filter by Account Type", acct_types, index=0, key="preview_acct_filter"
                        )
                        if sel_acct != "All":
                            df_filtered = df_summary_all[
                                df_summary_all["Account Type"].astype(str) == sel_acct
                            ]
                        else:
                            df_filtered = df_summary_all.copy()
                    else:
                        df_filtered = df_summary_all.copy()
                    search_term = st.text_input(
                        "Search entity / Flags", value="", key="preview_search"
                    )
                    if search_term:
                        search_lower = search_term.lower()
                        mask = pd.Series(False, index=df_filtered.index)
                        for c in ["Entity", "Flag - Manual Auditor Check"]:
                            if c in df_filtered.columns:
                                mask = mask | df_filtered[c].astype(str).str.lower().str.contains(
                                    search_lower, na=False
                                )
                        df_filtered = df_filtered[mask]
                    if "S.No" in df_filtered.columns:
                        df_filtered = df_filtered.set_index("S.No")
                    else:
                        df_filtered = df_filtered.copy()
                        df_filtered.index = range(1, len(df_filtered) + 1)
                    df_filtered.index.name = "S.No"
                    page_size = 50
                    total_rows = len(df_filtered)
                    total_pages = max(1, (total_rows + page_size - 1) // page_size)
                    last_filter_key = "preview_last_filter_summary"
                    current_filter_state = (
                        sel_acct if "sel_acct" in locals() else None,
                        search_term,
                    )
                    if st.session_state.get(last_filter_key) != current_filter_state:
                        st.session_state[last_filter_key] = current_filter_state
                        st.session_state["preview_page_summary"] = 1
                    if "preview_page_summary" not in st.session_state:
                        st.session_state["preview_page_summary"] = 1
                    page = st.session_state["preview_page_summary"]
                    colp1, colp2, colp3 = st.columns([1, 6, 1])
                    with colp1:
                        if st.button("⏮ First", key="first_page_summary"):
                            st.session_state["preview_page_summary"] = 1
                            page = 1
                        if st.button("◀ Prev", key="prev_page_summary"):
                            st.session_state["preview_page_summary"] = max(1, page - 1)
                            page = st.session_state["preview_page_summary"]
                    with colp2:
                        try:
                            new_page = st.number_input(
                                "Go to page",
                                min_value=1,
                                max_value=total_pages,
                                value=page,
                                key="page_input_summary",
                            )
                            if int(new_page) != page:
                                st.session_state["preview_page_summary"] = int(new_page)
                                page = st.session_state["preview_page_summary"]
                        except Exception:
                            pass
                    with colp3:
                        if st.button("Next ▶", key="next_page_summary"):
                            st.session_state["preview_page_summary"] = min(
                                total_pages, page + 1
                            )
                            page = st.session_state["preview_page_summary"]
                        if st.button("Last ⏭", key="last_page_summary"):
                            st.session_state["preview_page_summary"] = total_pages
                            page = total_pages
                    start = (page - 1) * page_size
                    end = start + page_size
                    st.markdown(
                        f"Showing rows {start+1} to {min(end, total_rows)} of {total_rows} (Page {page}/{total_pages})"
                    )
                    df_page = df_filtered.iloc[start:end]
                    st.dataframe(_sanitize_df_for_streamlit(df_page), width='stretch')
                    try:
                        csv_buffer = (
                            df_filtered.reset_index().to_csv(index=False).encode("utf-8")
                        )
                    except Exception:
                        csv_buffer = df_summary_all.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download shown (or full) sheet as CSV",
                        data=csv_buffer,
                        file_name="ALL_AccountType_Summary.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.warning(
                        f"Could not load 'ALL_AccountType_Summary' from the generated Excel: {e}"
                    )
            with st.expander(
                "View ALL_RCM_Combined (Combined RCM mapping) - paginated (50 rows)",
                expanded=False,
            ):
                try:
                    df_rcm_all = pd.read_excel(
                        excel_path_preview, sheet_name="ALL_RCM_Combined"
                    )
                    if "Account Type" in df_rcm_all.columns:
                        acct_types_rcm = [
                            "All",
                        ] + sorted(
                            df_rcm_all["Account Type"].dropna().astype(str).unique().tolist()
                        )
                        sel_acct_rcm = st.selectbox(
                            "Filter RCM by Account Type",
                            acct_types_rcm,
                            index=0,
                            key="preview_acct_filter_rcm",
                        )
                        if sel_acct_rcm != "All":
                            df_rcm_filtered = df_rcm_all[
                                df_rcm_all["Account Type"].astype(str) == sel_acct_rcm
                            ]
                        else:
                            df_rcm_filtered = df_rcm_all.copy()
                    else:
                        df_rcm_filtered = df_rcm_all.copy()
                    search_term_rcm = st.text_input(
                        "Search Control / Entity", value="", key="preview_search_rcm"
                    )
                    if search_term_rcm:
                        s = search_term_rcm.lower()
                        mask = pd.Series(False, index=df_rcm_filtered.index)
                        for c in ["Control Description", "Entity", "Control ID"]:
                            if c in df_rcm_filtered.columns:
                                mask = mask | df_rcm_filtered[c].astype(str).str.lower().str.contains(
                                    s, na=False
                                )
                        df_rcm_filtered = df_rcm_filtered[mask]
                    if "S.No" in df_rcm_filtered.columns:
                        df_rcm_filtered = df_rcm_filtered.set_index("S.No")
                    else:
                        df_rcm_filtered = df_rcm_filtered.copy()
                        df_rcm_filtered.index = range(1, len(df_rcm_filtered) + 1)
                    df_rcm_filtered.index.name = "S.No"
                    page_size = 50
                    total_rows_rcm = len(df_rcm_filtered)
                    total_pages_rcm = max(1, (total_rows_rcm + page_size - 1) // page_size)
                    last_filter_key_rcm = "preview_last_filter_rcm"
                    current_filter_state_rcm = (
                        sel_acct_rcm if "sel_acct_rcm" in locals() else None,
                        search_term_rcm,
                    )
                    if st.session_state.get(last_filter_key_rcm) != current_filter_state_rcm:
                        st.session_state[last_filter_key_rcm] = current_filter_state_rcm
                        st.session_state["preview_page_rcm"] = 1
                    if "preview_page_rcm" not in st.session_state:
                        st.session_state["preview_page_rcm"] = 1
                    page_rcm = st.session_state["preview_page_rcm"]
                    colp1, colp2, colp3 = st.columns([1, 6, 1])
                    with colp1:
                        if st.button("⏮ First", key="first_page_rcm"):
                            st.session_state["preview_page_rcm"] = 1
                            page_rcm = 1
                        if st.button("◀ Prev", key="prev_page_rcm"):
                            st.session_state["preview_page_rcm"] = max(1, page_rcm - 1)
                            page_rcm = st.session_state["preview_page_rcm"]
                    with colp2:
                        try:
                            new_page_rcm = st.number_input(
                                "Go to page",
                                min_value=1,
                                max_value=total_pages_rcm,
                                value=page_rcm,
                                key="page_input_rcm",
                            )
                            if int(new_page_rcm) != page_rcm:
                                st.session_state["preview_page_rcm"] = int(new_page_rcm)
                                page_rcm = st.session_state["preview_page_rcm"]
                        except Exception:
                            pass
                    with colp3:
                        if st.button("Next ▶", key="next_page_rcm"):
                            st.session_state["preview_page_rcm"] = min(
                                total_pages_rcm, page_rcm + 1
                            )
                            page_rcm = st.session_state["preview_page_rcm"]
                        if st.button("Last ⏭", key="last_page_rcm"):
                            st.session_state["preview_page_rcm"] = total_pages_rcm
                            page_rcm = total_pages_rcm
                    start_rcm = (page_rcm - 1) * page_size
                    end_rcm = start_rcm + page_size
                    st.markdown(
                        f"Showing rows {start_rcm+1} to {min(end_rcm, total_rows_rcm)} of {total_rows_rcm} (Page {page_rcm}/{total_pages_rcm})"
                    )
                    df_rcm_page = df_rcm_filtered.iloc[start_rcm:end_rcm]
                    st.dataframe(_sanitize_df_for_streamlit(df_rcm_page), width='stretch')
                    try:
                        csv_buffer_rcm = (
                            df_rcm_filtered.reset_index().to_csv(index=False).encode("utf-8")
                        )
                    except Exception:
                        csv_buffer_rcm = df_rcm_all.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download shown (or full) RCM as CSV",
                        data=csv_buffer_rcm,
                        file_name="ALL_RCM_Combined.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.warning(
                        f"Could not load 'ALL_RCM_Combined' from the generated Excel: {e}"
                    )
        st.markdown("### 📥 Download Reports")
        d1, d2, d3 = st.columns(3)
        with d1:
            if state.get("automation_excel_path"):
                with open(state["automation_excel_path"], "rb") as f:
                    st.download_button(
                        label="📊 Excel Report",
                        data=f.read(),
                        file_name="SOX_Report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        width='stretch',
                    )
        with d2:
            if state.get("automation_pdf_path"):
                with open(state["automation_pdf_path"], "rb") as f:
                    st.download_button(
                        label="📈 Charts PDF",
                        data=f.read(),
                        file_name="SOX_Charts.pdf",
                        mime="application/pdf",
                        width='stretch',
                    )
        with d3:
            if st.button("📝 Generate Executive Summary"):
                state["messages"].append(HumanMessage(content="Generate executive summary"))
                with st.spinner("🤖 AI creating executive summary..."):
                    for event in agent_executor.stream(state, config=config, stream_mode="updates"):
                        if "agent" in event:
                            agent = event["agent"]
                            if "messages" in agent:
                                msg = agent["messages"][-1]
                                if isinstance(msg, AIMessage) and msg.content:
                                    st.markdown(msg.content)
                                    state["messages"].append(msg)
    if st.button("🔄 Run New Analysis", width='stretch'):
            st.session_state["workflow_stage"] = "files_uploaded"
            st.rerun()

# Chat input (always available)
if user_input := st.chat_input("Ask me anything about SOX automation..."):
    state = st.session_state["analyst_state"]
    state["messages"].append(HumanMessage(content=user_input))
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    with st.chat_message("assistant", avatar="🤖"):
        response_placeholder = st.empty()
        streamed_text = ""
        try:
            for event in agent_executor.stream(state, config=config, stream_mode="updates"):
                if "agent" in event:
                    agent = event["agent"]
                    if "messages" in agent:
                        msg = agent["messages"][-1]
                        if isinstance(msg, AIMessage) and msg.content:
                            streamed_text += msg.content
                            response_placeholder.markdown(streamed_text + "▌")
            response_placeholder.markdown(streamed_text)
            state["messages"].append(AIMessage(content=streamed_text))
        except Exception as e:
            st.error(f"Error: {str(e)}")
