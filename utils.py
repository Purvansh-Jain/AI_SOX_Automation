import json
from pathlib import Path
import pandas as pd


def format_message_content(message):
    """Convert message content to displayable string."""
    parts = []
    tool_calls_processed = False

    # Handle main content
    if isinstance(message.content, str):
        parts.append(message.content)
    elif isinstance(message.content, list):
        # Handle complex content like tool calls (Anthropic format)
        for item in message.content:
            if item.get("type") == "text":
                parts.append(item["text"])
            elif item.get("type") == "tool_use":
                parts.append(f"\n🔧 Tool Call: {item['name']}")
                parts.append(f"   Args: {json.dumps(item['input'], indent=2, ensure_ascii=False)}")
                parts.append(f"   ID: {item.get('id', 'N/A')}")
                tool_calls_processed = True
    else:
        parts.append(str(message.content))

    # Handle tool calls attached to the message (OpenAI format) - only if not already processed
    if (
        not tool_calls_processed
        and hasattr(message, "tool_calls")
        and message.tool_calls
    ):
        for tool_call in message.tool_calls:
            parts.append(f"\n🔧 Tool Call: {tool_call['name']}")
            parts.append(f"   Args: {json.dumps(tool_call['args'], indent=2, ensure_ascii=False)}")
            parts.append(f"   ID: {tool_call['id']}")

    return "\n".join(parts)


def generate_executive_summary_from_metrics(analysis_results):
    """
    Generate executive summary text from analysis metrics dictionary.
    
    Args:
        analysis_results: Dict with keys: total_account_types, total_entities, 
                         in_scope_entities, total_flags, critical_flags, controls_mapped
    
    Returns:
        str: The formatted executive summary text
    """
    def _coerce_int(v):
        """Coerce various numeric-like inputs to int safely."""
        try:
            if v is None:
                return 0
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
    
    # Extract and coerce metrics
    total_accounts = _coerce_int(analysis_results.get('total_account_types', 0))
    total_entities = _coerce_int(analysis_results.get('total_entities', 0))
    in_scope_entities = _coerce_int(analysis_results.get('in_scope_entities', 0))
    flags_raised = _coerce_int(analysis_results.get('total_flags', 0))
    critical_flags = _coerce_int(analysis_results.get('critical_flags', 0))
    controls_mapped = _coerce_int(analysis_results.get('controls_mapped', 0))
    
    # Validate and calculate percentages
    notes = []
    if total_entities <= 0:
        notes.append("Total entities is zero or missing; percentage metrics will be shown as N/A.")
    if in_scope_entities > 0 and total_entities > 0 and in_scope_entities > total_entities:
        notes.append("In-scope entities exceeds total entities — this may indicate a counting or input error. Percentages will be shown as N/A.")
    
    if total_entities and total_entities > 0 and in_scope_entities <= total_entities:
        percent_in_scope = (in_scope_entities / total_entities) * 100
        percent_text = f"{percent_in_scope:.0f}%"
    else:
        percent_text = "N/A"
    
    # Build the narrative
    parts = []
    parts.append(f"### Executive Summary - SOX Audit Analysis\n\n")
    parts.append(f"**Overview**\nThis analysis examined {total_accounts} account types across {total_entities} entities to assess SOX compliance scope and control coverage.\n\n")
    parts.append("**Key Findings**\n\n")
    parts.append("1. **Scope Determination**\n")
    parts.append(f"   - {in_scope_entities} of {total_entities} entities ({percent_text}) were classified as \"In Scope\" based on the materiality threshold\n")
    parts.append("   - These entities represent the material accounts requiring detailed SOX testing\n\n")
    parts.append("2. **Control Coverage**\n")
    parts.append(f"   - {controls_mapped} controls were successfully mapped to in-scope entities\n")
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
    parts.append("1. **Immediate Actions**\n   - Review and remediate all critical (red) flags before audit fieldwork\n   - Validate control mappings for newly in-scope entities\n\n")
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
    
    return "".join(parts)


def save_executive_summary(analysis_results, output_path=None):
    """
    Generate and save executive summary to file.
    
    Args:
        analysis_results: Dict with analysis metrics
        output_path: Optional path to save to (defaults to executive_summary.txt)
    
    Returns:
        Path: The path where the summary was saved
    """
    summary_text = generate_executive_summary_from_metrics(analysis_results)
    
    if output_path is None:
        output_path = Path('executive_summary.txt')
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as fh:
        fh.write(summary_text)
    
    return output_path
