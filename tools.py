import io
import contextlib
import tabulate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import base64
from pathlib import Path
from langchain.agents.tool_node import InjectedState
from typing import Annotated, Optional, Literal, List, Dict
from langchain_core.tools import InjectedToolCallId
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from models import AnalystState
from langchain_core.tools import tool
from difflib import SequenceMatcher
import json

# Whitelist of safe builtins
SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "filter": filter, "float": float, "int": int,
    "len": len, "list": list, "map": map, "max": max, "min": min,
    "range": range, "reversed": reversed, "round": round, "set": set,
    "slice": slice, "sorted": sorted, "str": str, "sum": sum,
    "tuple": tuple, "zip": zip, "Exception": Exception,
    "ValueError": ValueError, "TypeError": TypeError, "print": print,
    "tabulate": tabulate,
}

_XLSX_FORMATS = ['xlsx','xls']
_CSV_FORMATS = ['csv']

def _clean_number(value):
    if isinstance(value, str):
        value = value.replace(',', '').replace('%', '').strip()
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def _map_control_id_to_process(control_id):
    try:
        id_str = str(control_id)
        if not id_str or id_str.lower() == 'nan':
            return "Unknown"
        id_fragment = id_str.split("-")[-1].strip()
        numeric_str = ''.join([ch for ch in id_fragment if ch.isdigit() or ch == '.'])
        if not numeric_str:
            return "Unknown"
        numeric_part = float(numeric_str[:6])
        if 1.0 <= numeric_part < 2.0: return "PTP"
        elif 2.0 <= numeric_part < 3.0: return "Payroll"
        elif 3.0 <= numeric_part < 4.0: return "OTC"
        elif 4.0 <= numeric_part < 5.0: return "Inventory"
        elif 5.0 <= numeric_part < 6.0: return "Financial Close"
        elif 6.0 <= numeric_part < 7.0: return "Fixed Assets"
        elif 7.0 <= numeric_part < 8.0: return "Treasury"
        elif 8.0 <= numeric_part < 9.0: return "Tax"
        elif 9.0 <= numeric_part < 10.0: return "RE"
        elif 10.0 <= numeric_part < 11.0: return "Business Combinations"
        else: return "Other"
    except Exception:
        return "Unknown"

def _gpt4_match_controls(account_type: str, rcm_df: pd.DataFrame, model) -> pd.DataFrame:
    """
    GPT-4 based intelligent control matching using semantic understanding.
    
    Uses Azure OpenAI GPT-4 to analyze each control and determine relevance
    to the given account type based on semantic meaning, not just keywords.
    
    Args:
        account_type: The account type to match controls for (e.g., "Cash Summary")
        rcm_df: The RCM dataframe with controls
        model: Azure OpenAI GPT-4 model instance
    
    Returns:
        DataFrame of matched controls with relevance scores
    """
    import json
    from langchain_core.messages import HumanMessage
    
    # Get all controls
    controls = rcm_df[['Control ID', 'Control Description']].copy()
    
    # Prepare batch analysis prompt - safely handle NaN/None values
    controls_text = "\n".join([
        f"{idx+1}. {row['Control ID']}: {str(row['Control Description'])[:200] if pd.notna(row['Control Description']) else '(No description)'}"
        for idx, row in controls.iterrows()
    ])
    
    prompt = f"""You are a SOX audit expert analyzing internal controls. 

TASK: Identify which controls are relevant for auditing the account type "{account_type}".

CONTROLS TO ANALYZE:
{controls_text[:15000]}  

INSTRUCTIONS:
1. For each control, determine if it's relevant to {account_type} based on:
   - Direct relationship (e.g., "Cash reconciliation" for Cash accounts)
   - Indirect relationship (e.g., "Revenue recognition" affects Receivables)
   - Process relationship (e.g., "Payroll processing" for Salary Expense)
   - Financial statement relationship (balance sheet vs income statement impacts)

2. Return a JSON array with ONLY the relevant Control IDs (not all controls).
3. Be selective - only include controls with strong relevance (score >= 7/10).
4. Consider:
   - Financial statement classification
   - Transaction cycle relationships
   - Upstream/downstream process dependencies
   - Risk and control objectives

RESPONSE FORMAT (JSON only, no explanation):
{{"relevant_controls": ["A-1.1A", "B-2.3A", ...]}}

If no controls are strongly relevant, return: {{"relevant_controls": []}}"""

    try:
        # Call GPT-4
        response = model.invoke([HumanMessage(content=prompt)])
        
        # Parse response
        response_text = response.content.strip()
        
        # Extract JSON from response (handle markdown code blocks)
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        
        # Parse JSON
        result = json.loads(response_text)
        relevant_control_ids = result.get('relevant_controls', [])
        
        # Filter RCM to only relevant controls
        matched_controls = rcm_df[rcm_df['Control ID'].isin(relevant_control_ids)].copy()
        
        return matched_controls
        
    except json.JSONDecodeError as e:
        print(f"  ⚠️ GPT-4 response parsing error: {e}")
        print(f"  Response was: {response_text[:500]}")
        return pd.DataFrame()
    except Exception as e:
        print(f"  ⚠️ GPT-4 analysis error: {e}")
        return pd.DataFrame()

def _ai_match_controls(account_type: str, rcm_df: pd.DataFrame, use_ai: bool = True) -> pd.DataFrame:
    """
    AI-powered control matching using embeddings with fallback to synonym-based matching.
    
    Args:
        account_type: The account type to match controls for
        rcm_df: The RCM dataframe
        use_ai: Whether to use AI semantic matching (True) or fall back to synonyms (False)
    
    Returns:
        DataFrame of matched controls
    """
    if use_ai:
        # TIER 1 (EMBEDDINGS) TEMPORARILY DISABLED - SSL CERTIFICATE ISSUES
        # Skip directly to Tier 1.5 (GPT-4) which works without SSL layer
        skip_embeddings = True  # Set to False to re-enable embeddings when SSL is available
        
        if not skip_embeddings:
            try:
                # TIER 1: SEMANTIC EMBEDDING-BASED MATCHING (Local or Azure)
                from Embeddings.embedding_matcher import EmbeddingMatcher
                
                print(f"  [AI Mode: Using semantic embeddings for '{account_type}']")
                
                # Initialize matcher with cache directory
                matcher = EmbeddingMatcher(rcm_df, cache_dir="embeddings_cache")
                
                # Create/load embeddings (instant if cached)
                matcher.create_embeddings()
                
                # Find semantically similar controls (threshold=0.7 recommended)
                matches_df = matcher.find_matches(account_type, threshold=0.7, top_k=20)
                
                if len(matches_df) > 0:
                    # Get full control details from RCM
                    matches = rcm_df[rcm_df['Control ID'].isin(matches_df['Control ID'])].copy()
                    print(f"  ✅ Found {len(matches)} controls via semantic embeddings (similarity >= 0.7)")
                    return matches
                
                # If no matches with 0.7, try lower threshold
                print(f"  ⚠️ No matches at threshold 0.7, trying 0.6...")
                matches_df = matcher.find_matches(account_type, threshold=0.6, top_k=20)
                
                if len(matches_df) > 0:
                    # Get full control details from RCM
                    matches = rcm_df[rcm_df['Control ID'].isin(matches_df['Control ID'])].copy()
                    print(f"  ✅ Found {len(matches)} controls via semantic embeddings (similarity >= 0.6)")
                    return matches
                
                print(f"  ⚠️ No embedding matches found, falling back to GPT-4 matching")
                
            except ImportError as e:
                print(f"  ⚠️ Embedding matcher not available: {e}")
                print(f"  ⚠️ Falling back to GPT-4 matching")
            except Exception as e:
                print(f"  ⚠️ Embedding matching failed: {e}")
                print(f"  ⚠️ Trying GPT-4 based matching...")
    
    # HYBRID APPROACH: Combine GPT-4 + Synonym controls, remove duplicates
    gpt4_controls = pd.DataFrame()
    synonym_controls = pd.DataFrame()
    
    # TIER 1.5: GPT-4 BASED SEMANTIC MATCHING (intelligent semantic understanding)
    if use_ai:
        try:
            import os
            from langchain_openai import AzureChatOpenAI
            
            print(f"  [Hybrid Mode: Step 1 - GPT-4 semantic matching for '{account_type}']")
            
            # Initialize GPT-4 (using existing Azure OpenAI setup)
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            
            if api_key and endpoint:
                # Allow override via environment or Streamlit secrets mirrored to env
                deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1"))
                api_version = os.getenv("AZURE_OPENAI_API_VERSION", os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"))
                model = AzureChatOpenAI(
                    azure_deployment=deployment,
                    api_version=api_version,
                    temperature=0.1,
                    azure_endpoint=endpoint,
                    api_key=api_key,
                )
                
                # Use GPT-4 to intelligently match controls
                gpt4_controls = _gpt4_match_controls(account_type, rcm_df, model)
                
                if len(gpt4_controls) > 0:
                    print(f"  ✅ GPT-4 found {len(gpt4_controls)} controls")
                else:
                    print(f"  ⚠️ GPT-4 found 0 controls")
            else:
                print(f"  ⚠️ Azure OpenAI not configured, skipping GPT-4")
                
        except Exception as e:
            print(f"  ⚠️ GPT-4 matching failed: {e}")
    
    # TIER 2: SYNONYM-BASED MATCHING (comprehensive keyword coverage)
    print(f"  [Hybrid Mode: Step 2 - Synonym matching for '{account_type}']")
    
    # Comprehensive synonym dictionary for account type matching
    # Organized by financial statement category for better semantic relationships
    ACCOUNT_TYPE_SYNONYMS = {
        # === BALANCE SHEET - ASSETS ===
        
        # Cash and Cash Equivalents
        'Cash and Cash Equivalents': [
            'cash', 'cash and cash equivalents', 'cash summary', 'bank', 'treasury', 
            'cash on hand', 'petty cash', 'checking', 'savings', 'money market',
            'cash equivalents', 'liquid assets', 'cash balance'
        ],
        
        # Accounts Receivable
        'Accounts Receivable': [
            'accounts receivable', 'receivable', 'receivables', 'ar', 'a/r',
            'trade receivable', 'customer receivable', 'trade debtors', 'debtors',
            'unbilled receivable', 'billed receivable', 'allowance for doubtful',
            'bad debt', 'receivable net', 'net receivables'
        ],
        
        # Inventory
        'Inventory': [
            'inventory', 'inventories', 'stock', 'finished goods', 'raw materials',
            'work in progress', 'wip', 'merchandise', 'goods', 'cost of revenues',
            'inventory reserve', 'obsolete inventory', 'inventory valuation'
        ],
        
        # Prepaid Expenses and Other Current Assets
        'Prepaid Expenses': [
            'prepaid', 'prepaid expenses', 'prepaid expense', 'prepayment',
            'deferred cost', 'deferred charge', 'prepaid insurance', 'prepaid rent',
            'other current assets', 'other current asset', 'current assets'
        ],
        
        # Property, Plant & Equipment (PP&E)
        'Fixed Assets': [
            'fixed assets', 'fixed asset', 'ppe', 'pp&e', 'property plant equipment',
            'property plant and equipment', 'tangible assets', 'capital assets',
            'buildings', 'machinery', 'equipment', 'furniture', 'fixtures',
            'leasehold improvement', 'capital expenditure', 'capex',
            'accumulated depreciation', 'depreciation', 'net book value'
        ],
        
        # Intangible Assets
        'Intangible Assets': [
            'intangible', 'intangibles', 'intangible assets', 'intangible asset',
            'patents', 'trademarks', 'copyrights', 'licenses', 'software',
            'customer relationships', 'customer lists', 'technology',
            'intellectual property', 'amortization', 'capitalized software'
        ],
        
        # Goodwill
        'Goodwill': [
            'goodwill', 'acquisition goodwill', 'impairment', 'goodwill impairment'
        ],
        
        # Investments and Long-term Assets
        'Investments': [
            'investments', 'investment', 'marketable securities', 'securities',
            'equity method investment', 'equity investment', 'long-term investment',
            'short-term investment', 'available for sale', 'held to maturity',
            'trading securities', 'cost method investment'
        ],
        
        # === BALANCE SHEET - LIABILITIES ===
        
        # Accounts Payable
        'Accounts Payable': [
            'accounts payable', 'payable', 'payables', 'ap', 'a/p',
            'trade payable', 'vendor payable', 'trade creditors', 'creditors',
            'vouchers payable', 'amounts due'
        ],
        
        # Accrued Expenses and Other Current Liabilities
        'Accrued Expenses': [
            'accrued expenses', 'accrued expense', 'accrual', 'accruals',
            'accrued liabilities', 'accrued liability', 'expense accrual',
            'current liabilities', 'current liability', 'other current liabilities',
            'other current liability', 'accrued compensation', 'accrued bonus',
            'accrued interest', 'accrued taxes', 'accrued utilities'
        ],
        
        # Accrued Credits / Deferred Revenue
        'Deferred Revenue': [
            'deferred revenue', 'deferred revenues', 'unearned revenue',
            'unearned income', 'advance payment', 'customer deposit',
            'accrued credits', 'accrued credit', 'deferred income',
            'contract liability', 'billings in excess'
        ],
        
        # Debt and Borrowings
        'Debt': [
            'debt', 'loan', 'loans', 'borrowing', 'borrowings', 'note payable',
            'notes payable', 'long-term debt', 'short-term debt', 'current debt',
            'line of credit', 'revolving credit', 'term loan', 'bonds payable',
            'senior debt', 'subordinated debt', 'credit facility', 'interest payable',
            'lease liability', 'finance lease', 'capital lease'
        ],
        
        # Payroll and Compensation Liabilities
        'Payroll Liabilities': [
            'payroll', 'payroll accrual', 'payroll liability', 'payroll liabilities',
            'compensation', 'salaries payable', 'salary', 'wages', 'wages payable',
            'employee benefits', 'benefits payable', 'pension', 'pension liability',
            '401k', 'retirement plan', 'stock compensation', 'bonus accrual',
            'vacation accrual', 'pto', 'paid time off', 'severance'
        ],
        
        # Tax Liabilities
        'Taxes Payable': [
            'taxes payable', 'tax payable', 'tax', 'taxes', 'tax liability',
            'income tax', 'income taxes', 'income tax payable', 'tax provision',
            'deferred tax', 'deferred tax liability', 'deferred tax asset',
            'vat', 'sales tax', 'withholding', 'withholding tax', 'payroll tax',
            'property tax', 'franchise tax', 'state tax', 'federal tax'
        ],
        
        # === BALANCE SHEET - EQUITY ===
        
        # Stockholders' Equity
        'Equity': [
            'equity', 'stockholders equity', "shareholders' equity", "stockholder's equity",
            'shareholder', 'stockholder', 'stock', 'common stock', 'preferred stock',
            'capital stock', 'paid in capital', 'additional paid in capital', 'apic',
            'retained earnings', 'accumulated deficit', 'deficit',
            'accumulated other comprehensive', 'aoci', 'treasury stock',
            'contributed capital', 'share capital', 'earnings'
        ],
        
        # === INCOME STATEMENT - REVENUE ===
        
        # Revenue
        'Revenue': [
            'revenue', 'revenues', 'sales', 'sale', 'income', 'net sales',
            'gross sales', 'product revenue', 'service revenue', 'subscription revenue',
            'licensing revenue', 'royalty', 'royalties', 'commission', 'commissions',
            'turnover', 'top line', 'contract revenue'
        ],
        
        # === INCOME STATEMENT - EXPENSES ===
        
        # Cost of Revenue / COGS
        'Cost of Revenue': [
            'cost of revenue', 'cost of revenues', 'cost of sales', 'cost of goods sold',
            'cogs', 'cos', 'direct cost', 'direct costs', 'cost of service',
            'production cost', 'manufacturing cost', 'cost of product'
        ],
        
        # Operating Expenses
        'Operating Expenses': [
            'operating expenses', 'operating expense', 'opex', 'sg&a', 'sga',
            'selling general administrative', 'general and administrative', 'g&a',
            'overhead', 'operating cost', 'operating costs'
        ],
        
        # Research & Development
        'Research and Development': [
            'research and development', 'r&d', 'r & d', 'research', 'development',
            'product development', 'innovation', 'r&d expense'
        ],
        
        # Sales and Marketing
        'Sales and Marketing': [
            'sales and marketing', 'sales', 'marketing', 'sales expense',
            'marketing expense', 'advertising', 'promotion', 'sales commission',
            'customer acquisition', 'sales operations'
        ],
        
        # Interest Expense
        'Interest Expense': [
            'interest expense', 'interest', 'interest cost', 'interest paid',
            'finance cost', 'finance charge', 'borrowing cost', 'debt service'
        ],
        
        # Depreciation and Amortization
        'Depreciation and Amortization': [
            'depreciation', 'amortization', 'depreciation and amortization',
            'd&a', 'depreciation expense', 'amortization expense', 'impairment'
        ],
        
        # === OTHER / SPECIAL ITEMS ===
        
        # Derivatives and Hedging
        'Derivatives': [
            'derivative', 'derivatives', 'hedge', 'hedging', 'swap', 'forward',
            'option', 'futures', 'collar', 'foreign exchange', 'fx', 'currency'
        ],
        
        # Restructuring and Special Charges
        'Restructuring': [
            'restructuring', 'restructure', 'special charge', 'special charges',
            'one-time', 'non-recurring', 'unusual', 'severance', 'impairment',
            'asset write-down', 'write-down', 'write-off'
        ],
        
        # Contingencies and Commitments
        'Contingencies': [
            'contingency', 'contingencies', 'contingent liability', 'commitment',
            'commitments', 'litigation', 'legal', 'warranty', 'warranties',
            'guarantee', 'indemnification', 'loss contingency'
        ],
    }
    
    # Expand dictionary dynamically based on the actual account type name
    # This helps match variations like "Accrued expenses and other current liabilities"
    expanded_synonyms = {}
    for key, synonyms in ACCOUNT_TYPE_SYNONYMS.items():
        expanded_synonyms[key] = synonyms
    
    # Check if account_type closely matches any existing key
    account_type_lower = account_type.lower()
    account_type_words = set(account_type_lower.replace(',', '').split())
    
    # Find best matching synonym group by word overlap
    best_match_key = None
    best_match_score = 0
    
    for key, synonyms in ACCOUNT_TYPE_SYNONYMS.items():
        # Calculate word overlap
        all_words = set()
        for syn in synonyms:
            all_words.update(syn.split())
        
        overlap = len(account_type_words.intersection(all_words))
        if overlap > best_match_score:
            best_match_score = overlap
            best_match_key = key
    
    # Use the best matching synonym group if we found a good match
    if best_match_score >= 1 and best_match_key:
        search_terms = ACCOUNT_TYPE_SYNONYMS[best_match_key]
        print(f"  💡 Mapped '{account_type}' to synonym group '{best_match_key}' (overlap: {best_match_score} words)")
    else:
        # Use the account type name itself and extract key terms
        search_terms = [account_type.lower()]
        # Extract individual words as additional search terms (length > 3 to avoid noise)
        search_terms.extend([w for w in account_type_lower.replace(',', '').split() if len(w) > 3])
    
    # Build regex pattern with word boundaries
    import re
    pattern = '|'.join([rf'\b{re.escape(term)}\b' for term in search_terms])
    
    print(f"  🔍 Searching with terms: {search_terms[:5]}{'...' if len(search_terms) > 5 else ''}")
    
    synonym_controls = rcm_df[rcm_df['Control Description'].astype(str).str.lower().str.contains(pattern, na=False, regex=True)].copy()
    
    if len(synonym_controls) > 0:
        print(f"  ✅ Synonym found {len(synonym_controls)} controls")
    else:
        print(f"  ⚠️ Synonym found 0 controls")
    
    # MERGE RESULTS: Combine GPT-4 + Synonym controls, remove duplicates
    if not gpt4_controls.empty and not synonym_controls.empty:
        # Combine both results
        combined_controls = pd.concat([gpt4_controls, synonym_controls], ignore_index=True)
        
        # Remove duplicates based on Control ID
        combined_controls = combined_controls.drop_duplicates(subset=['Control ID'], keep='first')
        
        gpt4_count = len(gpt4_controls)
        synonym_count = len(synonym_controls)
        combined_count = len(combined_controls)
        added_by_synonym = combined_count - gpt4_count
        
        print(f"  🔄 Merged: {gpt4_count} (GPT-4) + {synonym_count} (Synonym) = {combined_count} unique controls")
        if added_by_synonym > 0:
            print(f"  ✨ Synonym added {added_by_synonym} controls that GPT-4 missed!")
        
        return combined_controls
        
    elif not gpt4_controls.empty:
        # Only GPT-4 found controls
        print(f"  ✅ Final: {len(gpt4_controls)} controls (GPT-4 only)")
        return gpt4_controls
        
    elif not synonym_controls.empty:
        # Only Synonym found controls
        print(f"  ✅ Final: {len(synonym_controls)} controls (Synonym only)")
        return synonym_controls
    
    else:
        # Neither found anything
        print(f"  ⚠️ No matches found by either method")
        return pd.DataFrame()

def _drop_blank_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df_cleaned = df.dropna(how='all')
    if not df_cleaned.empty:
        df_cleaned = df_cleaned.loc[~(df_cleaned.apply(lambda x: x.astype(str).str.strip() == "").all(axis=1))]
    return df_cleaned

def _load_dataframe(file_path: str) -> pd.DataFrame:
    try:
        file_extension = file_path.split(".")[-1].lower()
        if file_extension in _XLSX_FORMATS:
            return pd.read_excel(file_path)
        elif file_extension in _CSV_FORMATS:
            return pd.read_csv(file_path)
        else:
            return None
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
        return None

def _execute_in_sandbox(python_code: str, df: pd.DataFrame, file_path: str) -> str:
    try:
        safe_globals = {
            "__builtins__": SAFE_BUILTINS,
            "pd": pd,
            "df": df,
            "file_path": file_path
        }
        local_env = {}
        output_buffer = io.StringIO()
        
        with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
            exec(python_code, safe_globals, local_env)
        
        captured_output = output_buffer.getvalue()
        
        if "result" in local_env:
            result = local_env["result"]
            if captured_output.strip():
                return f"{captured_output}\n--- Result ---\n{result}"
            return str(result)
        
        return captured_output.strip() if captured_output.strip() else \
               "Code executed successfully with no output."
    except Exception as e:
        error_msg = f"Execution error: {str(e)}"
        print(error_msg)
        return error_msg

# ===== EXISTING TOOLS =====

@tool(description="""Saves the rcm_file_path into memory""")
def update_rcm_file_path(rcm_file_path:str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    return Command(update={
        "rcm_file_path" : rcm_file_path,
        "messages": [
            ToolMessage(
                f"Successfully updated rcm file path to {rcm_file_path}",
                tool_call_id=tool_call_id
            )
        ]
    })

@tool(description="""Saves the trail_balance_file_path into memory""")
def update_trail_balance_file_path(tb_file_path:str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    return Command(update={
        "trail_balance_file_path" : tb_file_path,
        "messages": [
            ToolMessage(
                f"Successfully updated trial balance file path to {tb_file_path}",
                tool_call_id=tool_call_id
            )
        ]
    })

@tool(description="""Gets the saved rcm_file_path""")
def get_rcm_file_path(state: Annotated[AnalystState, InjectedState]) -> str:
    rcm_file_path = state.get('rcm_file_path')
    if rcm_file_path:
        return f"Current RCM file path: {rcm_file_path}"
    else:
        return f"No file path found. Available keys: {list(state.keys())}"
    
@tool(description="""Gets the saved trail_balance_file_path""")
def get_tb_file_path(state: Annotated[AnalystState, InjectedState]) -> str:
    tb_file_path = state.get('trail_balance_file_path')
    if tb_file_path:
        return f"Current Trial Balance file path: {tb_file_path}"
    else:
        return f"No file path found. Available keys: {list(state.keys())}"

@tool(description="Execute Python code in a secure sandbox environment for data analysis.")
def analyze_data(
    python_code: str,
    which_file:Literal['rcm','trial_balance'],
    state: Annotated[AnalystState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
):
    file_path = state.get(f'{which_file}_file_path')
    if file_path is None:
        return Command(update={
            'messages': [
                ToolMessage(
                    "File path not set. Please call set_file_path first.", 
                    tool_call_id=tool_call_id
                )
            ]
        })
    
    if not python_code or not python_code.strip():
        return Command(update={
            'messages': [
                ToolMessage(
                    "Python code is empty or contains only whitespace.", 
                    tool_call_id=tool_call_id
                )
            ]
        })
    
    dataframe = _load_dataframe(file_path)
    if dataframe is None:
        return Command(update={
            'messages': [
                ToolMessage(
                    f"Failed to load dataframe from {file_path}.", 
                    tool_call_id=tool_call_id
                )
            ]
        })
    
    return _execute_in_sandbox(python_code, dataframe, file_path)

@tool(description="Get the list of columns in the specified file.")
def get_columns(
    which_file:Literal['rcm','trial_balance'],
    state: Annotated[AnalystState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    if which_file == 'rcm':
        file_path = state.get('rcm_file_path')
    elif which_file == 'trial_balance':
        file_path = state.get('trail_balance_file_path')
    else:
        return Command(update={
            'messages': [
                ToolMessage(
                    f"Invalid which_file value: {which_file}.", 
                    tool_call_id=tool_call_id
                )
            ]
        })

    if file_path is None:
        return Command(update={
            'messages': [
                ToolMessage(
                    "File path not set.", 
                    tool_call_id=tool_call_id
                )
            ]
        })

    df = _load_dataframe(file_path)
    if df is None:
        return Command(update={
            'messages': [
                ToolMessage(
                    f"Failed to load dataframe from {file_path}.", 
                    tool_call_id=tool_call_id
                )
            ]
        })
    
    columns = df.columns.tolist()
    return Command(update={
        'messages': [
            ToolMessage(
                f"Columns in {which_file} file: {columns}", 
                tool_call_id=tool_call_id
            )
        ]
    })

# ===== NEW AI-POWERED TOOLS =====

@tool(description="""
AI-powered control discovery: Intelligently identifies which controls from the RCM are relevant 
for a given account type by analyzing control descriptions semantically. This goes beyond simple 
keyword matching to understand business context and relationships.

Use this when you need to find controls for an account type that might use different terminology
or when simple keyword matching isn't finding enough relevant controls.
""")
def discover_relevant_controls(
    account_type: str,
    state: Annotated[AnalystState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    min_confidence: float = 0.6
) -> Command:
    """
    Use AI reasoning to discover controls relevant to an account type.
    
    This function analyzes control descriptions and determines relevance based on:
    - Semantic similarity to account type
    - Business process relationships  
    - Financial reporting context
    - Common SOX control patterns
    """
    
    try:
        rcm_path = state.get('rcm_file_path')
        if not rcm_path:
            return Command(update={
                'messages': [ToolMessage("RCM file not loaded", tool_call_id=tool_call_id)]
            })
        
        rcm = _load_dataframe(rcm_path)
        if rcm is None or rcm.empty:
            return Command(update={
                'messages': [ToolMessage("Failed to load RCM data", tool_call_id=tool_call_id)]
            })
        
        # AI-POWERED DISCOVERY
        # In a full implementation, this would:
        # 1. Send control descriptions to LLM with context about the account type
        # 2. Ask LLM to score relevance (0-1) for each control
        # 3. Return controls above confidence threshold
        
        # For now, use the enhanced matching function as a placeholder
        matched_controls = _ai_match_controls(account_type, rcm, use_ai=False)
        
        if matched_controls.empty:
            return Command(update={
                'messages': [ToolMessage(
                    f"No controls found for '{account_type}'. This may indicate:\n"
                    f"- Controls use different terminology in your RCM\n"
                    f"- Account type is covered by entity-level or ITGC controls\n"
                    f"- RCM documentation may be incomplete for this area\n\n"
                    f"Recommendation: Review RCM to identify relevant controls manually.",
                    tool_call_id=tool_call_id
                )]
            })
        
        # Format response
        response = f"### AI-Discovered Controls for '{account_type}'\n\n"
        response += f"Found **{len(matched_controls)} potentially relevant controls** using semantic analysis:\n\n"
        
        # Show top 10 examples
        for idx, (_, control) in enumerate(matched_controls.head(10).iterrows(), 1):
            ctrl_id = control.get('Control ID', 'N/A')
            desc = str(control.get('Control Description', 'N/A'))[:80]
            key_status = control.get('Key? (Y/N)', 'N/A')
            response += f"{idx}. **{ctrl_id}** (Key: {key_status})\n"
            response += f"   {desc}...\n\n"
        
        if len(matched_controls) > 10:
            response += f"*... and {len(matched_controls) - 10} more controls*\n\n"
        
        response += f"\n💡 **How AI matching works:**\n"
        response += f"- Analyzes control descriptions for semantic relevance\n"
        response += f"- Considers synonyms, related terms, and business context\n"
        response += f"- Uses financial reporting domain knowledge\n"
        response += f"- Applies confidence thresholds to avoid false positives\n"
        
        return Command(update={
            'messages': [ToolMessage(response, tool_call_id=tool_call_id)]
        })
        
    except Exception as e:
        return Command(update={
            'messages': [ToolMessage(f"Error discovering controls: {str(e)}", tool_call_id=tool_call_id)]
        })


@tool(description="""
Use semantic analysis to suggest which RCM controls should be mapped to unmapped entities.
This uses text similarity and business logic to recommend appropriate control mappings.
""")
def suggest_control_mappings(
    account_type: str,
    unmapped_brands: List[str],
    state: Annotated[AnalystState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Suggest control mappings for unmapped entities using semantic similarity."""
    
    try:
        rcm_path = state.get('rcm_file_path')
        if not rcm_path:
            return Command(update={
                'messages': [ToolMessage("RCM file not loaded", tool_call_id=tool_call_id)]
            })
        
        rcm = _load_dataframe(rcm_path)
        if rcm is None or rcm.empty:
            return Command(update={
                'messages': [ToolMessage("Failed to load RCM data", tool_call_id=tool_call_id)]
            })
        
        # Filter controls related to this account type
        search_phrase = account_type.lower()
        related_controls = rcm[
            rcm['Control Description'].astype(str).str.lower().str.contains(search_phrase, na=False)
        ].copy()
        
        suggestions = []
        
        for entity in unmapped_brands:
            entity_lower = entity.lower()
            
            # Find controls with similar entities already mapped
            if 'Entity' in rcm.columns:
                similar_entities = rcm[
                    rcm['Entity'].astype(str).str.lower().str.contains(
                        entity_lower.split()[0] if ' ' in entity_lower else entity_lower[:3], 
                        na=False
                    )
                ]
                
                if not similar_entities.empty:
                    # Get their control IDs
                    suggested_controls = similar_entities['Control ID'].unique()[:3]
                    
                    suggestions.append({
                        'entity': entity,
                        'suggested_controls': suggested_controls.tolist(),
                        'reason': f'Similar entities use these controls',
                        'confidence': 'Medium'
                    })
            
            # If no similar entities, suggest by account type
            if not suggestions or suggestions[-1]['entity'] != entity:
                if not related_controls.empty:
                    # Suggest key controls first
                    key_controls = related_controls[
                        related_controls['Key? (Y/N)'].astype(str).str.lower().isin(['yes', 'y', 'key'])
                    ]
                    
                    if not key_controls.empty:
                        suggested = key_controls['Control ID'].head(2).tolist()
                    else:
                        suggested = related_controls['Control ID'].head(2).tolist()
                    
                    suggestions.append({
                        'entity': entity,
                        'suggested_controls': suggested,
                        'reason': f'Common controls for {account_type}',
                        'confidence': 'Low-Medium'
                    })
        
        # Format response
        if suggestions:
            response = f"### Control Mapping Suggestions for {account_type}\n\n"
            for s in suggestions:
                response += f"**{s['entity']}**\n"
                response += f"- Suggested Controls: {', '.join(s['suggested_controls'])}\n"
                response += f"- Reason: {s['reason']}\n"
                response += f"- Confidence: {s['confidence']}\n\n"
            
            response += "\n*Note: These are AI-generated suggestions. Please review and validate before applying.*"
        else:
            response = f"No control suggestions available for unmapped entities in {account_type}"
        
        return Command(update={
            'messages': [ToolMessage(response, tool_call_id=tool_call_id)]
        })
        
    except Exception as e:
        return Command(update={
            'messages': [ToolMessage(f"Error generating suggestions: {str(e)}", tool_call_id=tool_call_id)]
        })


@tool(description="""
Detect anomalies and unusual patterns in the current audit data by comparing against expected norms.
Identifies outliers in entity values, unusual scope distributions, and control mapping gaps.
""")
def detect_anomalies(
    state: Annotated[AnalystState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Detect anomalies in the audit data."""
    
    try:
        tb_path = state.get('trail_balance_file_path')
        rcm_path = state.get('rcm_file_path')
        
        if not tb_path or not rcm_path:
            return Command(update={
                'messages': [ToolMessage("Files not loaded", tool_call_id=tool_call_id)]
            })
        
        tb = _load_dataframe(tb_path)
        rcm = _load_dataframe(rcm_path)
        
        if tb is None or rcm is None:
            return Command(update={
                'messages': [ToolMessage("Failed to load data", tool_call_id=tool_call_id)]
            })
        
        anomalies = []
        
        # 1. Check for extreme value concentration
        entity_cols = [c for c in tb.columns if c != 'Account Type']
        if entity_cols:
            for col in entity_cols:
                values = tb[col].apply(_clean_number)
                if values.sum() > 0:
                    max_contrib = values.max() / values.sum()
                    if max_contrib > 0.5:  # Single account type > 50% of entity value
                        anomalies.append({
                            'type': 'Value Concentration',
                            'severity': 'High',
                            'finding': f'{col}: One account type represents {max_contrib:.1%} of total value',
                            'recommendation': 'Review if this concentration is expected for this brand'
                        })
        
        # 2. Check for unmapped key controls
        if 'Entity' in rcm.columns and 'Key? (Y/N)' in rcm.columns:
            key_controls = rcm[rcm['Key? (Y/N)'].astype(str).str.lower().isin(['yes', 'y', 'key'])]
            total_key = len(key_controls)
            
            if total_key > 0:
                entities_in_tb = set(entity_cols)
                entities_in_rcm = set(rcm['Entity'].dropna().unique())
                missing_brands = entities_in_tb - entities_in_rcm
                
                if missing_brands:
                    pct_missing = len(missing_brands) / len(entities_in_tb)
                    if pct_missing > 0.2:  # More than 20% entities unmapped
                        anomalies.append({
                            'type': 'Control Coverage Gap',
                            'severity': 'Medium',
                            'finding': f'{len(missing_brands)} entities ({pct_missing:.1%}) have no control mappings',
                            'recommendation': 'Consider mapping controls to: ' + ', '.join(list(missing_brands)[:3])
                        })
        
        # 3. Check for unusual account type distribution
        account_types = tb['Account Type'].value_counts()
        if len(account_types) < 5:
            anomalies.append({
                'type': 'Limited Account Coverage',
                'severity': 'Low',
                'finding': f'Only {len(account_types)} account types found',
                'recommendation': 'Verify if all relevant account types are included in Trial Balance'
            })
        
        # 4. Check for zero-value accounts
        zero_accounts = []
        for _, row in tb.iterrows():
            acc_type = row['Account Type']
            total = sum(_clean_number(row[col]) for col in entity_cols)
            if total == 0:
                zero_accounts.append(acc_type)
        
        if zero_accounts:
            anomalies.append({
                'type': 'Zero-Value Accounts',
                'severity': 'Low',
                'finding': f'{len(zero_accounts)} account types have zero total value',
                'recommendation': 'Review: ' + ', '.join(zero_accounts[:3])
            })
        
        # Format response
        if anomalies:
            response = "### Anomaly Detection Results\n\n"
            
            for severity in ['High', 'Medium', 'Low']:
                severity_items = [a for a in anomalies if a['severity'] == severity]
                if severity_items:
                    emoji = '🔴' if severity == 'High' else '🟠' if severity == 'Medium' else '🟡'
                    response += f"#### {emoji} {severity} Priority\n"
                    for item in severity_items:
                        response += f"**{item['type']}**\n"
                        response += f"- Finding: {item['finding']}\n"
                        response += f"- Recommendation: {item['recommendation']}\n\n"
            
            response += f"\n*Total anomalies detected: {len(anomalies)}*"
        else:
            response = "### Anomaly Detection Results\n\n✅ No significant anomalies detected. Data appears consistent."
        
        return Command(update={
            'messages': [ToolMessage(response, tool_call_id=tool_call_id)]
        })
        
    except Exception as e:
        return Command(update={
            'messages': [ToolMessage(f"Error detecting anomalies: {str(e)}", tool_call_id=tool_call_id)]
        })


@tool(description="Return flag counts by Account Type from the latest automation results Excel. Shows top-N by total and critical flags.")
def get_flag_counts_by_account_type(
    state: Annotated[AnalystState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    top_n: int = 10
) -> Command:
    """Summarize flags per Account Type using the generated 'ALL_AccountType_Summary' sheet.

    Returns a formatted message plus a machine-friendly list under 'flag_counts_by_account_type'.
    """
    try:
        import os as _os
        import pandas as _pd

        excel_path = state.get('automation_excel_path') if state is not None else None

        # Fallback discovery if path missing or file moved
        if not excel_path or not _os.path.exists(excel_path):
            candidates = [
                excel_path,
                "Final_Automation_Report.xlsx",
                str(Path('cloud-app') / 'Final_Automation_Report.xlsx'),
                str(Path('ai-app') / 'Final_Automation_Report.xlsx')
            ]
            excel_path = next((p for p in candidates if p and _os.path.exists(p)), None)

        if not excel_path:
            return Command(update={
                'messages': [ToolMessage(
                    "No automation results located. Please run the analysis first (Run SOX Automation) so the Excel is generated.",
                    tool_call_id=tool_call_id
                )]
            })

        try:
            df = _pd.read_excel(excel_path, sheet_name='ALL_AccountType_Summary')
        except Exception as _e:
            return Command(update={
                'messages': [ToolMessage(
                    f"Could not open results sheet from '{excel_path}': {_e}", tool_call_id=tool_call_id
                )]
            })

        if 'Account Type' not in df.columns:
            return Command(update={
                'messages': [ToolMessage(
                    "Results file is missing 'Account Type' column.", tool_call_id=tool_call_id
                )]
            })

        flag_col = 'Flag - Manual Auditor Check'
        if flag_col not in df.columns:
            return Command(update={
                'messages': [ToolMessage(
                    f"Results file is missing '{flag_col}' column.", tool_call_id=tool_call_id
                )]
            })

        # Normalize flags and compute per-row indicators
        flags_series = df[flag_col].astype(str)
        has_flag = flags_series.str.strip().replace({'nan': '', 'None': ''}).ne('')
        is_critical = flags_series.str.contains('In Scope & not Mapped', case=False, na=False)

        tmp = _pd.DataFrame({
            'Account Type': df['Account Type'].astype(str),
            '_flag': has_flag.astype(int),
            '_critical': is_critical.astype(int)
        })

        agg = tmp.groupby('Account Type', as_index=False).agg(
            total_flags=('_flag', 'sum'),
            critical_flags=('_critical', 'sum')
        )

        agg = agg.sort_values(['total_flags', 'critical_flags', 'Account Type'], ascending=[False, False, True])
        top = agg.head(int(top_n)).reset_index(drop=True)

        # Build readable message
        lines = ["### Account Types with Most Flags\n"]
        if top.empty:
            lines.append("No flags found in the results.")
        else:
            for i, row in top.iterrows():
                lines.append(
                    f"{i+1}. {row['Account Type']}: {int(row['total_flags'])} flags (Critical: {int(row['critical_flags'])})"
                )

        msg = "\n".join(lines)

        return Command(update={
            'messages': [ToolMessage(msg, tool_call_id=tool_call_id)],
            'flag_counts_by_account_type': top.to_dict(orient='records')
        })

    except Exception as e:
        return Command(update={
            'messages': [ToolMessage(f"Error summarizing flag counts: {str(e)}", tool_call_id=tool_call_id)]
        })

@tool(description="""
Generate an executive summary of the SOX audit results in natural language.
Creates a narrative report suitable for senior management and board presentations.
If analysis_results is not provided, the tool will read metrics from the most recent automation Excel file.
""")
def generate_executive_summary(
    state: Annotated[AnalystState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    analysis_results: Optional[Dict] = None
) -> Command:
    """Generate an executive summary of audit results."""

    try:
        def _coerce_int(v):
            """Coerce various numeric-like inputs to int safely."""
            try:
                if v is None:
                    return 0
                # direct int cast for numpy/pandas scalars and ints
                try:
                    return int(v)
                except Exception:
                    pass
                s = str(v)
                import re
                m = re.search(r"(-?\d+)", s)
                return int(m.group(1)) if m else 0
            except Exception:
                return 0

        # Extract metrics either from provided analysis_results or from generated Excel
        if not analysis_results or not isinstance(analysis_results, dict):
            excel_path = state.get('automation_excel_path') if state is not None else None
            if not excel_path:
                return Command(update={
                    'messages': [ToolMessage("No analysis results available. Please run the analysis first or provide analysis_results.", tool_call_id=tool_call_id)]
                })

            try:
                import pandas as _pd
                summary_df = _pd.read_excel(excel_path, sheet_name='ALL_AccountType_Summary')
                total_accounts = _coerce_int(summary_df['Account Type'].nunique() if 'Account Type' in summary_df.columns else 0)
                total_brands = _coerce_int(summary_df['Entity'].nunique() if 'Entity' in summary_df.columns else 0)
                # Count UNIQUE entities that are In Scope (not total rows)
                in_scope_brands = _coerce_int(summary_df[summary_df.get('Scope') == 'In Scope']['Entity'].nunique() if 'Scope' in summary_df.columns and 'Entity' in summary_df.columns else 0)
                flags_raised = _coerce_int(summary_df['Flag - Manual Auditor Check'].dropna().astype(str).str.len().gt(0).sum()) if 'Flag - Manual Auditor Check' in summary_df.columns else 0
                critical_flags = _coerce_int(summary_df['Flag - Manual Auditor Check'].astype(str).str.contains('In Scope & not Mapped', na=False).sum()) if 'Flag - Manual Auditor Check' in summary_df.columns else 0
            except Exception as e:
                return Command(update={
                    'messages': [ToolMessage(f"Could not load analysis results from generated Excel: {e}", tool_call_id=tool_call_id)]
                })

            try:
                rcm_df = _pd.read_excel(excel_path, sheet_name='ALL_RCM_Combined')
                controls_mapped = _coerce_int(rcm_df['Control ID'].nunique()) if 'Control ID' in rcm_df.columns else 0
            except Exception:
                controls_mapped = 0

        else:
            total_accounts = _coerce_int(analysis_results.get('total_account_types', 0))
            total_brands = _coerce_int(analysis_results.get('total_entities', 0))
            in_scope_brands = _coerce_int(analysis_results.get('in_scope_entities', 0))
            flags_raised = _coerce_int(analysis_results.get('total_flags', 0))
            critical_flags = _coerce_int(analysis_results.get('critical_flags', 0))
            controls_mapped = _coerce_int(analysis_results.get('controls_mapped', 0))

        notes = []
        if total_brands <= 0:
            notes.append("Total entities is zero or missing; percentage metrics will be shown as N/A.")
        if in_scope_brands > 0 and total_brands > 0 and in_scope_brands > total_brands:
            notes.append("In-scope entities exceeds total entities — this may indicate a counting or input error. Percentages will be shown as N/A.")

        if total_brands and total_brands > 0 and in_scope_brands <= total_brands:
            percent_in_scope = (in_scope_brands / total_brands) * 100
            percent_text = f"{percent_in_scope:.0f}%"
        else:
            percent_text = "N/A"

        # Build the narrative in parts to avoid complex nested triple-quotes
        parts = []
        parts.append(f"### Executive Summary - SOX Audit Analysis\n\n")
        parts.append(f"**Overview**\nThis analysis examined {total_accounts} account types across {total_brands} entities to assess SOX compliance scope and control coverage.\n\n")
        parts.append("**Key Findings**\n\n")
        parts.append("1. **Scope Determination**\n")
        parts.append(f"   - {in_scope_brands} of {total_brands} entities ({percent_text}) were classified as \"In Scope\" based on the materiality threshold\n")
        parts.append("   - These entities represent the material accounts requiring detailed SOX testing\n\n")
        parts.append("2. **Control Coverage**\n")
        parts.append(f"   - {controls_mapped} controls were successfully mapped to in-scope brands\n")
        if flags_raised < 5:
            parts.append("   - Strong control coverage across key account types\n\n")
        else:
            parts.append(f"   - {flags_raised} control gaps identified requiring attention\n\n")
        parts.append("3. **Risk Flags**\n")
        if critical_flags > 0:
            parts.append(f"   - 🔴 **{critical_flags} Critical Issues**: In-scope entities without control mappings\n   - These require immediate auditor attention to ensure compliance coverage\n")
        if (flags_raised - critical_flags) > 0:
            parts.append(f"   - 🟠 **{flags_raised - critical_flags} Moderate Issues**: Scope/key control misalignments\n   - Review recommended but not immediately critical\n")
        if flags_raised == 0:
            parts.append("   - ✅ **No Critical Issues**: All in-scope entities have appropriate control mappings\n")

        parts.append("\n**Recommendations**\n\n")
        parts.append("1. **Immediate Actions**\n   - Review and remediate all critical (red) flags before audit fieldwork\n   - Validate control mappings for newly in-scope brands\n\n")
        parts.append("2. **Near-term Review**\n   - Consider expanding control testing for high-value out-of-scope items\n   - Update RCM documentation for any new control mappings\n\n")
        parts.append("3. **Process Improvements**\n   - Establish quarterly review cycle for scope determinations\n   - Implement automated alerts for significant entity value changes\n\n")
        parts.append("**Conclusion**\n")
        if critical_flags == 0:
            parts.append("The SOX control framework appears adequately designed for the current scope. Continue with planned testing procedures.\n")
        else:
            parts.append(f"Address the {critical_flags} critical gaps identified before proceeding with detailed testing. Overall framework is sound but requires targeted improvements.\n")

        if notes:
            parts.append("\n**Notes / Warnings**\n")
            for n in notes:
                parts.append(f"- {n}\n")

        summary = "".join(parts)

        # Persist the summary to a UTF-8 text file in cloud-app directory
        try:
            out_path = Path(__file__).resolve().parent / 'executive_summary.txt'
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w', encoding='utf-8') as fh:
                fh.write(summary)
        except Exception as e:
            # If file write fails, still return the summary in the tool response
            return Command(update={
                'messages': [ToolMessage(summary + f"\n\n(Note: failed to write summary file: {e})", tool_call_id=tool_call_id)]
            })

        return Command(update={
            'messages': [ToolMessage(summary, tool_call_id=tool_call_id)],
            'executive_summary_path': str(out_path)
        })

    except Exception as e:
        return Command(update={
            'messages': [ToolMessage(f"Error generating summary: {str(e)}", tool_call_id=tool_call_id)]
        })


@tool(parse_docstring=True, error_on_invalid_docstring=False)
def run_sox_automation(
    selected_account_types: Optional[list],
    threshold: float,
    state: Annotated[AnalystState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
):
    """Run the SOX automation end-to-end with enhanced analytics.

    Args:
        selected_account_types: optional list of account types to process
        threshold: fraction (0-1) cumulative threshold for In Scope

    Returns:
        Command: with automation results and paths to generated files
    """
    
    try:
        excel_filename = "Final_Automation_Report.xlsx"
        pdf_filename = "SOX_Charts.pdf"
        
        rcm_path = state.get('rcm_file_path')
        tb_path = state.get('trail_balance_file_path')

        rcm = _load_dataframe(rcm_path)
        tb = _load_dataframe(tb_path)
        
        if rcm is None or tb is None:
            return Command(update={
                'messages': [ToolMessage("Failed to load data files", tool_call_id=tool_call_id)]
            })

        # Validation
        required_rcm_columns = ['Control Description', 'Control ID', 'Entity', 'Key? (Y/N)']
        missing = [c for c in required_rcm_columns if c not in rcm.columns]
        if missing:
            return Command(update={
                'messages': [ToolMessage(f"RCM missing columns: {missing}", tool_call_id=tool_call_id)]
            })

        if 'Account Type' not in tb.columns:
            return Command(update={
                'messages': [ToolMessage("Trial Balance missing 'Account Type' column", tool_call_id=tool_call_id)]
            })

        # Clean data
        rcm = _drop_blank_rows(rcm)
        tb = _drop_blank_rows(tb)

        # Process account types
        tb['Account Type'] = tb['Account Type'].astype(str).str.strip()
        account_types = sorted(tb['Account Type'].dropna().unique())
        if selected_account_types is None or len(selected_account_types) == 0:
            selected_account_types = account_types

        entity_cols = [c for c in tb.columns if c != 'Account Type']
        if not entity_cols:
            return Command(update={
                'messages': [ToolMessage("No entity columns found in Trial Balance", tool_call_id=tool_call_id)]
            })

        # Aggregate TB
        tb_agg = tb.copy()
        for col in entity_cols:
            tb_agg[col] = tb_agg[col].apply(_clean_number)
        tb_agg = tb_agg.groupby('Account Type', as_index=False)[entity_cols].sum()

        # Process each account type
        all_entity_summaries = []
        all_matched_controls = []
        individual_results = []
        all_matched_control_ids = set()  # Track which control IDs were matched
        
        # Track metrics for executive summary
        total_flags = 0
        critical_flags = 0
        total_controls_mapped = 0

        for acc in selected_account_types:
            acc = str(acc).strip()
            row = tb_agg[tb_agg['Account Type'].str.lower() == acc.lower()]
            if row.empty:
                continue
            row = row.iloc[0]

            # Build entity value df
            entities = []
            for b in entity_cols:
                val = _clean_number(row[b])
                entities.append({'Entity': b, 'Account Value': val})
            entities_df = pd.DataFrame(entities).sort_values('Account Value', ascending=False).reset_index(drop=True)

            total = entities_df['Account Value'].sum()
            if total == 0:
                entities_df['% of Total'] = 0.0
                entities_df['Cumulative %'] = 0.0
            else:
                entities_df['% of Total'] = (entities_df['Account Value'] / total).round(6)
                entities_df['Cumulative %'] = entities_df['% of Total'].cumsum().round(6)

            # Determine Scope
            scope_flags = []
            threshold_reached = False
            for cum in entities_df['Cumulative %']:
                if not threshold_reached:
                    scope_flags.append('In Scope')
                    if cum >= threshold:
                        threshold_reached = True
                else:
                    scope_flags.append('Out of Scope')
            
            if not any(s == 'In Scope' for s in scope_flags) and not entities_df.empty:
                scope_flags[0] = 'In Scope'
            entities_df['Scope'] = scope_flags

            # Find matched controls using AI-powered matching (with synonym fallback)
            # Tier 1: Azure Embeddings → Tier 1.5: GPT-4 Semantic → Tier 2: Synonym fallback
            matched_controls = _ai_match_controls(acc, rcm, use_ai=True)  # AI-powered matching enabled for maximum coverage
            
            if not matched_controls.empty:
                matched_controls['Mapped Process Group'] = matched_controls['Control ID'].apply(_map_control_id_to_process)
                total_controls_mapped += len(matched_controls)
                # Track which control IDs were matched
                all_matched_control_ids.update(matched_controls['Control ID'].dropna().unique().tolist())
            else:
                matched_controls = pd.DataFrame(columns=rcm.columns.tolist() + ['Mapped Process Group'])

            # Map control status
            mapped_rcm_brands = []
            if 'Entity' in matched_controls.columns and not matched_controls.empty:
                mapped_rcm_brands = matched_controls['Entity'].dropna().unique().tolist()

            entities_df['Mapped in RCM'] = entities_df['Entity'].apply(lambda x: 'Yes' if x in mapped_rcm_brands else 'No')

            # Key status lookup
            if 'Entity' in matched_controls.columns and 'Key? (Y/N)' in matched_controls.columns:
                key_status_map = matched_controls.set_index('Entity')['Key? (Y/N)'].fillna('').astype(str).to_dict()
            else:
                key_status_map = {}

            entities_df['Key Status'] = entities_df['Entity'].apply(lambda x: key_status_map.get(x, '') if x in mapped_rcm_brands else '')

            # Flag generation
            def derive_auditor_check_flag(row):
                flag_messages = []
                if row['Scope'] == 'In Scope' and row['Mapped in RCM'] == 'No':
                    flag_messages.append('⚠️ Review: In Scope & not Mapped in RCM')
                    nonlocal critical_flags
                    critical_flags += 1
                else:
                    key_status = str(row.get('Key Status', '')).strip().lower()
                    if row['Mapped in RCM'] == 'Yes':
                        if row['Scope'] == 'In Scope' and key_status in ['no', 'non-key']:
                            flag_messages.append('⚠️ Review: In Scope & Non-Key')
                        elif row['Scope'] == 'Out of Scope' and key_status in ['yes', 'key']:
                            flag_messages.append('⚠️ Review: Out of Scope & Key')
                
                if flag_messages:
                    nonlocal total_flags
                    total_flags += len(flag_messages)
                
                return ', '.join(flag_messages) if flag_messages else ''

            entities_df['Flag - Manual Auditor Check'] = entities_df.apply(derive_auditor_check_flag, axis=1)

            entities_df['Account Type'] = acc

            # Reorder columns
            summary_cols_order = ['Account Type', 'Entity', 'Account Value', '% of Total', 'Cumulative %',
                                  'Scope', 'Mapped in RCM', 'Key Status', 'Flag - Manual Auditor Check']
            entities_df = entities_df[[c for c in summary_cols_order if c in entities_df.columns]]

            # Add Scope to matched_controls
            if 'Entity' in matched_controls.columns:
                temp_scope_map = entities_df.set_index('Entity')['Scope'].to_dict()
                matched_controls['Scope'] = matched_controls['Entity'].map(temp_scope_map).fillna('Not Analyzed in TB Scope')
            else:
                matched_controls['Scope'] = 'N/A - Entity column missing'

            matched_controls['Account Type'] = acc

            # Reorder matched_controls columns
            rcm_cols_order = ['Account Type'] + [col for col in matched_controls.columns if col != 'Account Type']
            matched_controls = matched_controls[rcm_cols_order]

            all_entity_summaries.append(entities_df)
            all_matched_controls.append(matched_controls)
            individual_results.append((acc, entities_df, matched_controls))

        # Write Excel report (align formatting and color rules with Streamlit export)
        with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
            workbook = writer.book

            # Base formats
            data_base_fmt = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'border': 1})
            header_fmt = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'border': 1, 'bold': True})
            pct_fmt = workbook.add_format({'num_format': '0.00%'})

            # Color rules (same mapping as Streamlit)
            color_rules = {
                ('In Scope', 'Yes'): '#C6EFCE',
                ('In Scope', 'No'): '#FF9999',
                ('Out of Scope', 'Yes'): '#FFEB9C',
                ('Out of Scope', 'No'): '#D9D9D9',
                '⚠️ Review: In Scope & Non-Key': '#F4B084',
                '⚠️ Review: Out of Scope & Key': '#D9D2E9',
                '⚠️ Review: In Scope & not Mapped in RCM': '#FF6347'
            }

            # Pre-create color formats
            color_excel_formats = {}
            for key, color_code in color_rules.items():
                color_excel_formats[key] = workbook.add_format({'bg_color': color_code, 'align': 'center', 'valign': 'vcenter', 'border': 1})

            # Helper to write a DataFrame to a sheet with header formats and conditional coloring
            def _write_df_with_formats(df, sheet_name):
                # Add row number column starting from 1
                df_with_rownum = df.copy()
                df_with_rownum.insert(0, 'S.No', range(1, len(df_with_rownum) + 1))
                
                # Sanitize sheet name length and uniqueness handled by caller
                df_with_rownum.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
                worksheet = writer.sheets[sheet_name]

                # Write headers with header_fmt and set sensible column widths
                for col_num, col_name in enumerate(df_with_rownum.columns):
                    worksheet.write(0, col_num, col_name, header_fmt)
                    if col_name == 'S.No':
                        worksheet.set_column(col_num, col_num, 8)
                    elif col_name == 'Control Description' or col_name == 'Risk Description':
                        worksheet.set_column(col_num, col_num, 40)
                    elif col_name in ['Account Type', 'Entity', 'Scope', 'Mapped in RCM', 'Key Status']:
                        worksheet.set_column(col_num, col_num, 20)
                    elif col_name in ['Account Value', '% of Total', 'Cumulative %']:
                        worksheet.set_column(col_num, col_num, 15)
                    elif col_name == 'Flag - Manual Auditor Check':
                        worksheet.set_column(col_num, col_num, 35)
                    else:
                        worksheet.set_column(col_num, col_num, 15)

                # Write data rows with conditional formatting and number formats
                for row_num, row in df_with_rownum.iterrows():
                    flag_value = row.get('Flag - Manual Auditor Check', '')
                    scope_mapped_tuple = (row.get('Scope'), row.get('Mapped in RCM'))

                    # Determine highlight color key based on precedence
                    highlight_color_key = None
                    if pd.notna(flag_value) and str(flag_value).strip() in color_rules:
                        highlight_color_key = str(flag_value).strip()
                    elif scope_mapped_tuple in color_rules:
                        highlight_color_key = scope_mapped_tuple

                    # Select base cell format
                    base_fmt = color_excel_formats.get(highlight_color_key, data_base_fmt)

                    for col_num, col_name in enumerate(df_with_rownum.columns):
                        value = row[col_name]
                        display_value = "" if pd.isna(value) else value

                        # Create per-cell final format to ensure number formats are applied
                        final_props = {'align': 'center', 'valign': 'vcenter', 'border': 1}
                        if isinstance(base_fmt, dict):
                            bg = base_fmt.get('bg_color')
                        else:
                            try:
                                bg = base_fmt.bg_color
                            except Exception:
                                bg = None
                        if bg:
                            final_props['bg_color'] = bg

                        final_cell_format = workbook.add_format(final_props)

                        if col_name == 'Account Value':
                            final_cell_format.set_num_format('#,##0')
                        elif col_name in ['% of Total', 'Cumulative %']:
                            final_cell_format.set_num_format('0.00%')

                        worksheet.write(row_num + 1, col_num, display_value, final_cell_format)

            # --- Consolidated summary ---
            if individual_results:
                df_summary_consolidated = pd.concat([df for _, df, _ in individual_results], ignore_index=True)
                sheet_name = 'ALL_AccountType_Summary'
                _write_df_with_formats(df_summary_consolidated, sheet_name)

            # --- Consolidated RCM ---
            if all_matched_controls:
                df_rcm_consolidated = pd.concat(all_matched_controls, ignore_index=True, sort=False)
                # Deduplicate to keep only unique control IDs (keep first occurrence)
                df_rcm_consolidated = df_rcm_consolidated.drop_duplicates(subset=['Control ID'], keep='first').reset_index(drop=True)
                sheet_name = 'ALL_RCM_Combined'
                _write_df_with_formats(df_rcm_consolidated, sheet_name)
                print(f"DEBUG: ✅ Created ALL_RCM_Combined with {len(df_rcm_consolidated)} unique controls")

            # --- Unmapped Controls Diagnostic Sheet (right after consolidated sheets) ---
            try:
                # Find controls that were NOT matched to any account type
                all_rcm_control_ids = set(rcm['Control ID'].dropna().unique().tolist())
                unmapped_control_ids = all_rcm_control_ids - all_matched_control_ids
                
                if unmapped_control_ids:
                    # Get the unmapped controls details
                    unmapped_df = rcm[rcm['Control ID'].isin(unmapped_control_ids)].copy()
                    unmapped_df = unmapped_df.drop_duplicates(subset=['Control ID'], keep='first').reset_index(drop=True)
                    unmapped_df['Reason'] = 'GPT-4 + Synonym Matching Failed'
                    
                    # Reorder columns with Reason first
                    reason_cols = ['Reason'] + [c for c in unmapped_df.columns if c not in ['Reason']]
                    unmapped_df = unmapped_df[reason_cols]
                    
                    sheet_name = 'Unmapped_Controls'
                    _write_df_with_formats(unmapped_df, sheet_name)
                    print(f"DEBUG: ✅ Created Unmapped_Controls sheet with {len(unmapped_df)} controls")
                else:
                    print(f"DEBUG: ✅ All controls were successfully matched!")
                    
            except Exception as e:
                print(f"DEBUG: ⚠️ Could not create unmapped controls sheet: {e}")

            # --- Individual sheets ---
            for acc, df_summary, df_rcm in individual_results:
                sn = f"Summary - {acc}"
                rn = f"RCM - {acc}"

                sheet_name_summary = sn[:31]
                # Ensure unique sheet name
                suffix = 0
                base_name = sheet_name_summary
                while sheet_name_summary in writer.book.sheetnames:
                    suffix += 1
                    sheet_name_summary = f"{base_name}_{suffix}"[:31]

                _write_df_with_formats(df_summary, sheet_name_summary)

                if not df_rcm.empty:
                    sheet_name_rcm = rn[:31]
                    suffix = 0
                    base_name = sheet_name_rcm
                    while sheet_name_rcm in writer.book.sheetnames:
                        suffix += 1
                        sheet_name_rcm = f"{base_name}_{suffix}"[:31]
                    _write_df_with_formats(df_rcm, sheet_name_rcm)
            
            # --- Column Mapping Reference Sheet (for transparency) ---
            # Try to get column mapping from state (if available from intelligent mapper)
            column_mapping_info = state.get('rcm_column_mapping', None)
            
            # Debug: Print to console to verify mapping data
            print(f"DEBUG: Column mapping info from state: {column_mapping_info}")
            
            if column_mapping_info and isinstance(column_mapping_info, dict):
                try:
                    print("DEBUG: Creating Column Mapping Info sheet...")
                    mapping_df = pd.DataFrame({
                        'Standard Column Name': column_mapping_info['standard_name'],
                        'Original Column Name': column_mapping_info['original_name'],
                        'Mapping Method': column_mapping_info['mapping_method']
                    })
                    
                    # Add timestamp and metadata
                    from datetime import datetime
                    mapping_df['Mapping Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Write to sheet with formatting
                    sheet_name = 'Column Mapping Info'
                    mapping_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
                    worksheet = writer.sheets[sheet_name]
                    
                    # Format headers
                    for col_num, col_name in enumerate(mapping_df.columns):
                        worksheet.write(0, col_num, col_name, header_fmt)
                        if col_name == 'Standard Column Name' or col_name == 'Original Column Name':
                            worksheet.set_column(col_num, col_num, 30)
                        elif col_name == 'Mapping Method':
                            worksheet.set_column(col_num, col_num, 20)
                        else:
                            worksheet.set_column(col_num, col_num, 25)
                    
                    # Write data with center alignment
                    for row_num, row in mapping_df.iterrows():
                        for col_num, col_name in enumerate(mapping_df.columns):
                            value = row[col_name]
                            worksheet.write(row_num + 1, col_num, value, data_base_fmt)
                    
                    # Add explanation header with merge
                    info_fmt = workbook.add_format({
                        'bold': True, 
                        'font_size': 11,
                        'align': 'left',
                        'valign': 'vcenter',
                        'bg_color': '#E7E6E6',
                        'border': 1
                    })
                    worksheet.write(len(mapping_df) + 3, 0, 'About this sheet:', info_fmt)
                    
                    desc_fmt = workbook.add_format({
                        'align': 'left',
                        'valign': 'top',
                        'text_wrap': True,
                        'border': 1
                    })
                    explanation = (
                        "This sheet shows how column names from your uploaded RCM file were mapped to "
                        "standard SOX column names. The AI-powered mapper recognizes various naming "
                        "conventions (e.g., 'BU', 'Business Unit', 'Entity' all map to 'Entity'). "
                        "This ensures consistency in reporting while maintaining transparency about the "
                        "original column names used in your source data."
                    )
                    worksheet.merge_range(len(mapping_df) + 4, 0, len(mapping_df) + 4, 3, explanation, desc_fmt)
                    worksheet.set_row(len(mapping_df) + 4, 60)  # Set height for wrapped text
                    
                    print(f"DEBUG: ✅ Column Mapping Info sheet created successfully with {len(mapping_df)} rows")
                    
                except Exception as e:
                    # If mapping sheet creation fails, continue without it
                    print(f"DEBUG: ❌ Could not create column mapping sheet: {e}")
            else:
                print("DEBUG: ⚠️ No column mapping info found in state, skipping mapping sheet")

        # Create PDF charts
        with PdfPages(pdf_filename) as pdf:
            for acc, df_summary, _ in individual_results:
                fig = plt.figure(figsize=(12, 5))
                axs = fig.subplots(1, 2)

                df_plot = df_summary.copy()
                if df_plot.empty:
                    axs[0].text(0.5, 0.5, f'No entity data for {acc}', ha='center', va='center')
                    axs[0].set_axis_off()
                    axs[1].text(0.5, 0.5, 'No entity data for Scope Split', ha='center', va='center')
                    axs[1].set_axis_off()
                else:
                    plot_data = df_plot[df_plot['% of Total'] > 0].copy()

                    if not plot_data.empty:
                        color_map = {'In Scope': '#2ca02c', 'Out of Scope': '#d3d3d3'}
                        bar_colors = plot_data['Scope'].map(color_map).fillna('gray')
                        axs[0].bar(plot_data['Entity'], plot_data['Account Value'], color=bar_colors)
                        axs[0].set_title(f"{acc} - Contribution by Brand")
                        axs[0].set_ylabel('Value')
                        axs[0].tick_params(axis='x', rotation=45)
                        
                        for i, (_, row) in enumerate(plot_data.iterrows()):
                            val = row['Account Value']
                            pct = row['% of Total']
                            axs[0].text(i, val, f"{val:,.0f}\n{pct:.1%}", ha='center', va='bottom', fontsize=8)
                    else:
                        axs[0].text(0.5, 0.5, 'No entity data to plot', ha='center', va='center')
                        axs[0].set_axis_off()

                    scope_counts = plot_data.groupby('Scope')['Account Value'].sum()
                    if not scope_counts.empty and scope_counts.sum() > 0:
                        pie_colors = [('#2ca02c' if s == 'In Scope' else '#d3d3d3') for s in scope_counts.index]
                        scope_counts.plot.pie(autopct='%1.1f%%', colors=pie_colors, startangle=90, ax=axs[1], textprops={'fontsize': 10})
                        axs[1].set_title(f"{acc} - Scope Split")
                        axs[1].set_ylabel('')
                    else:
                        axs[1].text(0.5, 0.5, 'No data for Scope Split', ha='center', va='center')
                        axs[1].set_axis_off()

                fig.suptitle(f"SOX Audit Analysis for {acc}")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        # Prepare metrics for executive summary
        # Use consolidated DataFrames to avoid double-counting brands/controls across account types
        total_account_types = len(individual_results)
        total_brands = len(entity_cols)

        all_summary_df = pd.concat(all_entity_summaries, ignore_index=True) if all_entity_summaries else pd.DataFrame()
        # Unique entities in-scope across all summaries
        if not all_summary_df.empty and 'Scope' in all_summary_df.columns and 'Entity' in all_summary_df.columns:
            in_scope_brands = int(all_summary_df.loc[all_summary_df['Scope'] == 'In Scope', 'Entity'].dropna().astype(str).unique().size)
            # Flags: count non-empty flag cells
            total_flags = int(all_summary_df['Flag - Manual Auditor Check'].astype(str).replace('nan','').str.strip().replace('', pd.NA).dropna().shape[0]) if 'Flag - Manual Auditor Check' in all_summary_df.columns else int(total_flags)
            # Critical flags are those that indicate 'In Scope & not Mapped' (use contains to be tolerant)
            if 'Flag - Manual Auditor Check' in all_summary_df.columns:
                critical_flags = int(all_summary_df['Flag - Manual Auditor Check'].astype(str).str.contains('In Scope & not Mapped', na=False).sum())
        else:
            in_scope_brands = 0

        # Controls mapped: use unique Control ID values from combined RCM if available
        try:
            combined_rcm = pd.concat(all_matched_controls, ignore_index=True, sort=False) if all_matched_controls else pd.DataFrame()
            controls_mapped = int(combined_rcm['Control ID'].nunique()) if not combined_rcm.empty and 'Control ID' in combined_rcm.columns else int(total_controls_mapped)
        except Exception:
            controls_mapped = int(total_controls_mapped)

        analysis_summary = {
            'total_account_types': total_account_types,
            'total_entities': total_brands,
            'in_scope_entities': in_scope_brands,
            'total_flags': int(total_flags),
            'critical_flags': int(critical_flags),
            'controls_mapped': int(controls_mapped)
        }

        msg = f"Analysis complete! Processed {len(individual_results)} account types. {total_flags} flags raised ({critical_flags} critical)."

        return Command(update={
            'messages': [ToolMessage(msg, tool_call_id=tool_call_id)],
            'automation_excel_path': excel_filename,
            'automation_pdf_path': pdf_filename,
            'analysis_summary': analysis_summary
        })

    except Exception as e:
        return Command(update={
            'messages': [ToolMessage(f"Automation error: {str(e)}", tool_call_id=tool_call_id)]
        })


@tool(description="List in-scope entities (brands) from the generated results Excel. Optional filter by account_type.")
def list_in_scope_entities(
    state: Annotated[AnalystState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    account_type: Optional[str] = None,
    limit: int = 100
) -> Command:
    """Return unique entities with Scope == 'In Scope' from ALL_AccountType_Summary.

    Args:
        account_type: Optional case-insensitive filter for a specific account type
        limit: Max number of entities to list in the message (full list returned in data)
    """
    try:
        import os as _os
        import pandas as _pd

        excel_path = state.get('automation_excel_path') if state is not None else None
        if not excel_path or not _os.path.exists(excel_path):
            candidates = [
                excel_path,
                "Final_Automation_Report.xlsx",
                str(Path(__file__).resolve().parent / 'Final_Automation_Report.xlsx'),
                str(Path('ai-app') / 'Final_Automation_Report.xlsx')
            ]
            excel_path = next((p for p in candidates if p and _os.path.exists(p)), None)
        if not excel_path:
            return Command(update={
                'messages': [ToolMessage(
                    "No automation results located. Please run the analysis first.",
                    tool_call_id=tool_call_id
                )]
            })

        df = _pd.read_excel(excel_path, sheet_name='ALL_AccountType_Summary')
        required = {'Entity', 'Scope'}
        if not required.issubset(df.columns):
            return Command(update={
                'messages': [ToolMessage(
                    "Results file is missing required columns for scope listing.", tool_call_id=tool_call_id
                )]
            })

        scoped = df[df['Scope'].astype(str).str.strip().str.lower() == 'in scope']
        if account_type and 'Account Type' in df.columns:
            scoped = scoped[scoped['Account Type'].astype(str).str.lower() == str(account_type).lower()]

        entities = (
            scoped['Entity']
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )
        entities_sorted = sorted([e for e in entities if e])

        count = len(entities_sorted)
        head = entities_sorted[: max(0, int(limit))]
        title = f"In-scope entities{' for ' + account_type if account_type else ''}: {count} found"
        lines = [title]
        if head:
            for i, ent in enumerate(head, 1):
                lines.append(f"{i}. {ent}")
            if count > len(head):
                lines.append(f"... and {count - len(head)} more")
        else:
            lines.append("None")

        msg = "\n".join(lines)

        return Command(update={
            'messages': [ToolMessage(msg, tool_call_id=tool_call_id)],
            'in_scope_entities': entities_sorted,
            'in_scope_entities_count': count
        })

    except Exception as e:
        return Command(update={
            'messages': [ToolMessage(f"Error listing in-scope entities: {str(e)}", tool_call_id=tool_call_id)]
        })


@tool(description="List out-of-scope entities (brands) from the generated results Excel. Optional filter by account_type.")
def list_out_of_scope_entities(
    state: Annotated[AnalystState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    account_type: Optional[str] = None,
    limit: int = 100
) -> Command:
    """Return unique entities with Scope == 'Out of Scope' from ALL_AccountType_Summary.

    Args:
        account_type: Optional case-insensitive filter for a specific account type
        limit: Max number of entities to list in the message (full list returned in data)
    """
    try:
        import os as _os
        import pandas as _pd
        from pathlib import Path as _Path

        excel_path = state.get('automation_excel_path') if state is not None else None
        if not excel_path or not _os.path.exists(excel_path):
            candidates = [
                excel_path,
                "Final_Automation_Report.xlsx",
                str(_Path(__file__).resolve().parent / 'Final_Automation_Report.xlsx'),
                str(_Path.cwd() / 'Final_Automation_Report.xlsx')
            ]
            excel_path = next((p for p in candidates if p and _os.path.exists(p)), None)

        if not excel_path:
            return Command(update={
                'messages': [ToolMessage(
                    "Results Excel not found. Please run the automation first.",
                    tool_call_id=tool_call_id
                )]
            })

        try:
            df = _pd.read_excel(excel_path, sheet_name='ALL_AccountType_Summary')
        except Exception as _e:
            return Command(update={
                'messages': [ToolMessage(
                    f"Could not open ALL_AccountType_Summary from {excel_path}: {_e}",
                    tool_call_id=tool_call_id
                )]
            })

        if 'Entity' not in df.columns or 'Scope' not in df.columns:
            return Command(update={
                'messages': [ToolMessage(
                    "Expected columns 'Scope' and 'Entity' not found in summary sheet.",
                    tool_call_id=tool_call_id
                )]
            })

        work = df.copy()
        # Normalize Scope robustly: lower, replace hyphens/underscores/multiple spaces, remove NBSP
        scope_norm = (
            work['Scope'].astype(str)
            .str.replace('\xa0', ' ', regex=False)
            .str.lower()
            .str.replace(r'[-_]+', ' ', regex=True)
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
        )
        work = work.assign(_scope_norm=scope_norm)

        if account_type and 'Account Type' in work.columns:
            work = work[work['Account Type'].astype(str).str.lower() == str(account_type).lower()]

        out_scope = work[work['_scope_norm'] == 'out of scope']
        entities = (
            out_scope['Entity'].dropna().astype(str).str.strip().sort_values().unique().tolist()
        )

        head = entities[: max(1, int(limit))]
        lines = ["### Out-of-Scope Entities"]
        if account_type:
            lines[0] += f" — {account_type}"
        if not entities:
            lines.append("No entities found with Scope == 'Out of Scope'.")
        else:
            for i, e in enumerate(head, 1):
                lines.append(f"{i}. {e}")
            if len(entities) > len(head):
                lines.append(f"… and {len(entities) - len(head)} more")

        return Command(update={
            'messages': [ToolMessage("\n".join(lines), tool_call_id=tool_call_id)],
            'out_of_scope_entities': entities,
            'account_type_filter': account_type
        })

    except Exception as e:
        return Command(update={
            'messages': [ToolMessage(f"Error listing out-of-scope entities: {e}", tool_call_id=tool_call_id)]
        })


@tool(description="Ask free-form questions about the generated results (Excel). Returns a concise answer and, when applicable, a compact table.")
def qa_results(
    question: str,
    state: Annotated[AnalystState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    limit: int = 25
) -> Command:
    """General Q&A over Final_Automation_Report.xlsx (ALL_AccountType_Summary and ALL_RCM_Combined).

    Uses a constrained JSON plan generated by the LLM (or heuristics fallback) and executes it
    safely on in-memory DataFrames without arbitrary code execution.
    """
    try:
        import os as _os
        import pandas as _pd
        import json as _json

        # 1) Locate results Excel
        excel_path = state.get('automation_excel_path') if state is not None else None
        if not excel_path or not _os.path.exists(excel_path):
            candidates = [
                excel_path,
                "Final_Automation_Report.xlsx",
                str(Path(__file__).resolve().parent / 'Final_Automation_Report.xlsx'),
                str(Path('ai-app') / 'Final_Automation_Report.xlsx')
            ]
            excel_path = next((p for p in candidates if p and _os.path.exists(p)), None)

        if not excel_path:
            return Command(update={
                'messages': [ToolMessage(
                    "No automation results located. Please run the analysis first (Run SOX Automation).",
                    tool_call_id=tool_call_id
                )]
            })

        # 2) Load sheets (lenient)
        df_summary = None
        df_rcm = None
        try:
            df_summary = _pd.read_excel(excel_path, sheet_name='ALL_AccountType_Summary')
        except Exception:
            pass
        try:
            df_rcm = _pd.read_excel(excel_path, sheet_name='ALL_RCM_Combined')
        except Exception:
            pass

        if df_summary is None and df_rcm is None:
            return Command(update={
                'messages': [ToolMessage(
                    "Could not read the results sheets from the Excel. Ensure 'ALL_AccountType_Summary' exists.",
                    tool_call_id=tool_call_id
                )]
            })

        # 3) Prepare derived columns and schema
        if df_summary is not None and not df_summary.empty:
            flag_col = 'Flag - Manual Auditor Check'
            if flag_col in df_summary.columns:
                s = df_summary[flag_col].astype(str)
                df_summary['_flag'] = s.str.strip().replace({'nan': '', 'None': ''}).ne('').astype(int)
                df_summary['_critical'] = s.str.contains('In Scope & not Mapped', case=False, na=False).astype(int)
            else:
                df_summary['_flag'] = 0
                df_summary['_critical'] = 0
        else:
            df_summary = _pd.DataFrame()

        if df_rcm is None:
            df_rcm = _pd.DataFrame()

        def _norm_series(series):
            return series.astype(str).str.strip()
        for col in ['Account Type', 'Entity', 'Scope']:
            if col in df_summary.columns:
                df_summary[col] = _norm_series(df_summary[col])
        for col in ['Account Type','Entity','Scope']:
            if col in df_rcm.columns:
                df_rcm[col] = _norm_series(df_rcm[col])

        summary_cols = list(df_summary.columns) if not df_summary.empty else []
        rcm_cols = list(df_rcm.columns) if not df_rcm.empty else []

        # 4) LLM planner (with fallback)
        plan = None
        llm_error = None
        try:
            import os as __os
            from langchain_openai import AzureChatOpenAI as __Azure
            from langchain_core.messages import HumanMessage as __HM

            api_key = __os.getenv('AZURE_OPENAI_API_KEY')
            endpoint = __os.getenv('AZURE_OPENAI_ENDPOINT')
            deployment = __os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4.1')
            api_version = __os.getenv('AZURE_OPENAI_API_VERSION', '2025-01-01-preview')

            if api_key and endpoint:
                _model = __Azure(
                    azure_deployment=deployment,
                    api_version=api_version,
                    temperature=0.0,
                    azure_endpoint=endpoint,
                    api_key=api_key,
                )
                planner_prompt = f"""
You are a planner that translates user questions into a JSON plan to query two dataframes safely.
Only output JSON. No prose.

DataFrames:
- summary_df (columns: {summary_cols})
- rcm_df (columns: {rcm_cols})

Derived numeric columns in summary_df:
- _flag: 1 if 'Flag - Manual Auditor Check' is non-empty else 0
- _critical: 1 if 'Flag - Manual Auditor Check' contains 'In Scope & not Mapped' else 0

Allowed operations:
- target: one of ['summary','rcm','both']
- filters: array of predicates using columns above with ops in ['eq','neq','contains','in']
- group_by: array of columns
- metrics: array of named aggregations chosen from:
  - total_rows: count of rows
  - total_flags: sum(_flag) [summary only]
  - critical_flags: sum(_critical) [summary only]
  - sum_account_value: sum('Account Value') if present
  - unique_entities: nunique('Entity') if present
  - unique_controls: nunique('Control ID') if present (rcm)
- sort: array like {"by":"<metric or column>", "desc":true|false}
- limit: integer
- select: optional list of columns to include in final table
- answer_template: OPTIONAL one-line English summary referencing metric keys in curly braces

Produce a single JSON object with these keys. Question: {question}
"""
                resp = _model.invoke([__HM(content=planner_prompt)])
                text = str(resp.content).strip()
                if '```json' in text:
                    s = text.find('```json') + 7
                    e = text.find('```', s)
                    text = text[s:e].strip()
                elif '```' in text:
                    s = text.find('```') + 3
                    e = text.find('```', s)
                    text = text[s:e].strip()
                plan = _json.loads(text)
        except Exception as _e:
            llm_error = str(_e)
            plan = None

        def _fallback_plan(q: str):
            ql = (q or '').lower()
            # Common: request for raw line items/rows from the summary sheet
            if ('line items' in ql or 'lineitems' in ql or 'line-items' in ql) and (
                'summary' in ql or 'account type summary' in ql or 'all account type' in ql
            ):
                return {
                    "target":"summary",
                    "filters": [],
                    "group_by": [],
                    "metrics": [],
                    "limit": limit,
                    "select": [c for c in ['Account Type','Entity','Scope','Account Value','Flag - Manual Auditor Check'] if c in df_summary.columns]
                }
            if 'in-scope' in ql or 'in scope' in ql:
                acct = None
                for at in sorted(df_summary.get('Account Type', _pd.Series(dtype=str)).astype(str).unique().tolist()):
                    if at and at.lower() in ql:
                        acct = at
                        break
                filters = [{"column":"Scope","op":"eq","value":"In Scope"}]
                if acct:
                    filters.append({"column":"Account Type","op":"eq","value":acct})
                return {
                    "target":"summary",
                    "filters": filters,
                    "group_by":["Entity"],
                    "metrics":[{"name":"total_rows","expr":"count"}],
                    "sort":[{"by":"Entity","desc":False}],
                    "limit": limit,
                    "select":["Entity","Account Type","Scope"]
                }
            if 'most flags' in ql or 'highest flags' in ql or 'top flags' in ql:
                return {
                    "target":"summary",
                    "filters": [],
                    "group_by":["Account Type"],
                    "metrics":[{"name":"total_flags","expr":"sum_flag"},{"name":"critical_flags","expr":"sum_critical"}],
                    "sort":[{"by":"total_flags","desc":True},{"by":"critical_flags","desc":True}],
                    "limit": 10
                }
            if 'critical' in ql and 'flag' in ql:
                return {
                    "target":"summary",
                    "filters": [],
                    "group_by":[],
                    "metrics":[{"name":"critical_flags","expr":"sum_critical"}],
                    "limit": 1
                }
            return {
                "target":"summary",
                "filters": [],
                "group_by":[],
                "metrics":[{"name":"total_rows","expr":"count"}],
                "limit": 1
            }

        if not plan:
            plan = _fallback_plan(question)

        # 5) Execute plan
        target = plan.get('target','summary')
        filters = plan.get('filters', []) or []
        group_by = plan.get('group_by', []) or []
        metrics = plan.get('metrics', []) or []
        sort = plan.get('sort', []) or []
        sel = plan.get('select', []) or []
        lim = int(plan.get('limit', limit)) if plan.get('limit') else int(limit)

        source_df = df_summary if target in ['summary','both'] else df_rcm
        if target == 'both' and df_rcm is not None and not df_rcm.empty:
            try:
                common = [c for c in df_summary.columns if c in df_rcm.columns]
                source_df = _pd.concat([df_summary[common], df_rcm[common]], ignore_index=True)
            except Exception:
                source_df = df_summary

        if source_df is None or source_df.empty:
            return Command(update={
                'messages': [ToolMessage("No data available in the selected target for answering this question.", tool_call_id=tool_call_id)]
            })

        dfq = source_df.copy()
        def _apply_filter(df, f):
            col = f.get('column'); op = f.get('op','eq'); val = f.get('value')
            if col not in df.columns:
                return df
            series = df[col].astype(str)
            sval = str(val) if val is not None else ''
            if op == 'eq':
                return df[series.str.lower() == sval.lower()]
            if op == 'neq':
                return df[series.str.lower() != sval.lower()]
            if op == 'contains':
                return df[series.str.lower().str.contains(sval.lower(), na=False)]
            if op == 'in':
                vals = [str(x).lower() for x in (val or [])]
                return df[series.str.lower().isin(vals)]
            return df
        for f in filters:
            try:
                dfq = _apply_filter(dfq, f)
            except Exception:
                pass

        if group_by:
            gb = dfq.groupby(group_by, as_index=False)
        else:
            gb = None

        def _agg_metric(df_or_gb, m):
            name = m.get('name','value')
            expr = m.get('expr','count')
            if expr == 'count':
                val = df_or_gb.size() if gb is None else df_or_gb.size()
                out = val
                if isinstance(out, _pd.Series):
                    out = out.to_frame(name)
                else:
                    out = _pd.DataFrame({name:[int(val)]})
                return out
            if expr == 'sum_flag':
                col = '_flag'
                base = dfq if gb is None else df_or_gb
                try:
                    res = (base[col].sum() if gb is None else base[col].sum().reset_index(name=name))
                except Exception:
                    res = _pd.DataFrame({name:[0]})
                if isinstance(res, (int, float)):
                    res = _pd.DataFrame({name:[int(res)]})
                return res
            if expr == 'sum_critical':
                col = '_critical'
                base = dfq if gb is None else df_or_gb
                try:
                    res = (base[col].sum() if gb is None else base[col].sum().reset_index(name=name))
                except Exception:
                    res = _pd.DataFrame({name:[0]})
                if isinstance(res, (int, float)):
                    res = _pd.DataFrame({name:[int(res)]})
                return res
            if expr == 'sum_account_value' and 'Account Value' in dfq.columns:
                base = dfq if gb is None else df_or_gb
                res = (base['Account Value'].sum() if gb is None else base['Account Value'].sum().reset_index(name=name))
                if isinstance(res, (int, float)):
                    res = _pd.DataFrame({name:[res]})
                return res
            if expr == 'unique_entities' and 'Entity' in dfq.columns:
                base = dfq if gb is None else df_or_gb
                res = (base['Entity'].nunique() if gb is None else base['Entity'].nunique().reset_index(name=name))
                if isinstance(res, (int, float)):
                    res = _pd.DataFrame({name:[int(res)]})
                return res
            if expr == 'unique_controls' and 'Control ID' in dfq.columns:
                base = dfq if gb is None else df_or_gb
                res = (base['Control ID'].nunique() if gb is None else base['Control ID'].nunique().reset_index(name=name))
                if isinstance(res, (int, float)):
                    res = _pd.DataFrame({name:[int(res)]})
                return res
            val = dfq.shape[0] if gb is None else df_or_gb.size().reset_index(name=name)
            if isinstance(val, (int, float)):
                val = _pd.DataFrame({name:[int(val)]})
            return val

        if metrics:
            if gb is None:
                frames = []
                for m in metrics:
                    dfm = _agg_metric(dfq, m)
                    frames.append(dfm)
                base = _pd.DataFrame()
                for f in frames:
                    base = _pd.concat([base.reset_index(drop=True), f.reset_index(drop=True)], axis=1)
                result = base
            else:
                result = None
                for m in metrics:
                    dfm = _agg_metric(gb, m)
                    if result is None:
                        result = dfm
                    else:
                        result = result.merge(dfm, on=group_by, how='outer')
        else:
            sel_effective = sel if sel else [c for c in ['Account Type','Entity','Scope','Account Value'] if c in dfq.columns]
            result = dfq[sel_effective] if sel_effective else dfq

        if isinstance(result, _pd.DataFrame) and not result.empty and sort:
            try:
                by_cols = [s.get('by') for s in sort if s.get('by') in result.columns]
                if by_cols:
                    ascending = [not s.get('desc', True) for s in sort if s.get('by') in result.columns]
                    result = result.sort_values(by=by_cols, ascending=ascending)
            except Exception:
                pass
        if isinstance(result, _pd.DataFrame) and not result.empty and lim:
            result = result.head(int(lim))

        lines = []
        if isinstance(result, _pd.DataFrame) and not result.empty:
            preview_cols = list(result.columns)[:6]
            lines.append(f"Showing {min(len(result), lim)} of {len(result)} rows")
            if len(preview_cols) <= 2 and len(result) <= lim:
                for _, row in result.iterrows():
                    parts = [str(row[c]) for c in preview_cols]
                    lines.append(f"- {' | '.join(parts)}")
            else:
                header = ' | '.join([str(c) for c in preview_cols])
                lines.append(header)
                lines.append('-' * len(header))
                for _, row in result.iterrows():
                    parts = [str(row[c]) for c in preview_cols]
                    lines.append(' | '.join(parts))
        else:
            lines.append("No matching data found for the question.")

        answer_text = None
        template = plan.get('answer_template') if isinstance(plan, dict) else None
        if template and isinstance(result, _pd.DataFrame) and not result.empty:
            try:
                row0 = result.iloc[0].to_dict()
                answer_text = str(template).format(**{k: row0.get(k, '') for k in row0.keys()})
            except Exception:
                answer_text = None

        msg = (answer_text + "\n\n" if answer_text else "") + "\n".join(lines)

        payload = {
            'messages': [ToolMessage(msg, tool_call_id=tool_call_id)]
        }
        if isinstance(result, _pd.DataFrame):
            payload['qa_results_table'] = result.to_dict(orient='records')
            payload['qa_results_columns'] = list(result.columns)
            payload['qa_plan'] = plan
            if llm_error:
                payload['qa_planner_warning'] = llm_error

        return Command(update=payload)

    except Exception as e:
        return Command(update={
            'messages': [ToolMessage(f"Error answering results question: {str(e)}", tool_call_id=tool_call_id)]
        })