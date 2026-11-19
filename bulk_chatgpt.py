import streamlit as st
import pandas as pd
import asyncio
import logging
import csv
import os
import tiktoken
from openai import AsyncOpenAI
import string


# ---- Intrepid look (gointrepid.com) -------------------------------------------------------------------------
# Colours and fonts come from .streamlit/config.toml; this adds the logo, the link back to the tools page, the
# title and the card around the tool. static/logo-white.png is served because enableStaticServing = true.
st.set_page_config(page_title="Bulk ChatGPT | Intrepid Digital", layout="centered")
st.html("""
<style>
/* Streamlit chrome: hide the top bar, its menu and the status widget. */
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
.block-container { padding-top: 0 !important; max-width: 900px !important; }

/* Top row: the site logo at the size gointrepid.com uses (63px tall) and a link back to the tools page. */
.id-top { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 28px 0 0; }
.id-top img { height: 63px; width: auto; display: block; }
.id-back { font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
  color: #a2ff2c !important; text-decoration: none !important; white-space: nowrap; }
.id-back:hover { color: #95ff0e !important; }
@media (max-width: 640px) { .id-top img { height: 48px; } }

/* Hero */
.id-hero { padding: 56px 0 36px; }
.id-hero h1 { font-size: clamp(36px, 5.5vw, 64px) !important; line-height: 1.08 !important; letter-spacing: -.02em; padding: 0 0 18px !important; }
.id-hero h1 { margin: 0 !important; }
.id-hero p { margin: 4px 0 0 !important; font-size: 18px; line-height: 1.6; color: #bfbaba; max-width: 700px; }
.id-hero p strong { color: #e8eaf0; }

/* Tool card: the outer block gets the site's surface + volt edge. */
.st-key-tool { background: #1e1c1c; border: 1px solid rgba(255,255,255,.08); border-left: 3px solid #a2ff2c; padding: 36px 36px 32px; }

/* Uploader: dashed drop zone, centred. */
[data-testid="stFileUploaderDropzone"] { border: 1.5px dashed rgba(255,255,255,.18) !important; background: rgba(255,255,255,.02) !important; padding: 32px 20px !important; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: #a2ff2c !important; background: rgba(162,255,44,.05) !important; }
[data-testid="stWidgetLabel"] p { font-family: 'IBM Plex Mono', monospace; font-size: 11px !important; letter-spacing: .14em; text-transform: uppercase; color: #6b6666 !important; }

/* Buttons: volt, dark ink, uppercase, as the site CTAs. */
.stButton button, .stDownloadButton button, [data-testid="stFileUploaderDropzone"] button { border-radius: 2px !important; }
.stButton button, .stDownloadButton button { background: #a2ff2c !important; border: 0 !important; padding: 14px 28px !important; }
.stButton button:hover, .stDownloadButton button:hover { background: #95ff0e !important; }
.stButton button p, .stDownloadButton button p { color: #0a0b0f !important; font-weight: 700 !important; font-size: 13px !important; letter-spacing: .04em; text-transform: uppercase; }

/* Metrics as the stat tiles. */
[data-testid="stMetric"] { background: #2a2828; padding: 14px 18px; }
[data-testid="stMetricValue"] { font-family: Outfit, sans-serif; font-weight: 800; }
[data-testid="stMetricLabel"] p { font-family: 'IBM Plex Mono', monospace; font-size: 11px !important; letter-spacing: .12em; text-transform: uppercase; color: #6b6666 !important; }
</style>
""")
st.html("""
<div class="id-top">
  <a href="https://www.gointrepid.com/tools/" target="_blank" rel="noopener"><img src="app/static/logo-white.png" alt="Intrepid Digital"></a>
  <a class="id-back" href="https://www.gointrepid.com/tools/" target="_blank" rel="noopener">&larr; All Tools</a>
</div>
<div class="id-hero">
  <h1>Bulk ChatGPT Tool</h1>
  <p>Run a ChatGPT prompt on every row of a CSV file. Map your columns to variables, write the prompts, and download the answers as a CSV. Uses your own OpenAI API key.</p>
</div>
""")
# --------------------------------------------------------------------------------------------------------------

# Set up logging
logging.basicConfig(level=logging.INFO)

# Constants
TOKEN_LIMIT = 128000  # GPT-4o-mini context window
SAFETY_MARGIN = 1000  # Leave room for responses
MAX_CONCURRENT_REQUESTS = 5  # Adjust based on rate limits
OUTPUT_FILE = "responses.csv"

# --- minimal helpers to prevent KeyError and support case-insensitive placeholders ---

class SafeDict(dict):
    """Missing keys return empty string instead of raising KeyError."""
    def __missing__(self, key):
        return ""

def build_context(row, column_to_variable: dict) -> dict:
    """
    Builds a dict usable by str.format_map with:
      - user-mapped variable names
      - their lowercase variants
      - original column names
      - their lowercase variants
    So {pattern} or {Pattern} both work, and {ColumnName} also works.
    """
    ctx = {}
    for col, var in column_to_variable.items():
        val = row[col]
        # user variable name
        ctx[var] = val
        ctx[var.lower()] = val
        # original column name
        ctx[col] = val
        ctx[col.lower()] = val
    return ctx

def extract_placeholders(template: str) -> set[str]:
    """Collect placeholders from a format string (for preflight checks)."""
    out = set()
    for literal, fname, fmt, conv in string.Formatter().parse(template):
        if fname:
            out.add(fname)
    return out


with st.container(key="tool"):
    api_key = st.text_input("Enter your OpenAI API key", type="password")
    uploaded_file = st.file_uploader("Choose your CSV file", type=['csv'])

    if uploaded_file and api_key:
        # Robust CSV load with friendly error for empty/unreadable files
        try:
            try:
                df = pd.read_csv(uploaded_file)
            except pd.errors.EmptyDataError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        if df.empty:
            st.error("Your CSV appears to be empty. Please upload a file with at least one row.")
            st.stop()

        columns = df.columns.tolist()

        # Column mapping
        st.write("Map each column to a variable name that will be used in the prompts:")
        column_to_variable = {}
        for column in columns:
            variable_name = st.text_input(f"Enter a variable name for {column}", value=column)
            column_to_variable[column] = variable_name

        # Prompt customization (unchanged)
        system_prompt = st.text_area("Edit the system prompt", 
                                     value="Edit the system prompt. You can include {variable_name}.")
        user_prompt_template = st.text_area("Edit the user prompt", 
                                            value="Edit the user prompt. You can include {variable_name}.")

        # Preflight: warn if prompts reference variables that don't obviously map to columns/vars
        # (non-blocking: SafeDict will still prevent crashes)
        used_placeholders = extract_placeholders(system_prompt) | extract_placeholders(user_prompt_template)
        mapped_names = set(column_to_variable.values()) | {c for c in columns} | {c.lower() for c in columns} | {v.lower() for v in column_to_variable.values()}
        unknown = [v for v in used_placeholders if v not in mapped_names]
        if unknown:
            st.info(f"Note: these placeholders are not directly mapped to a column or variable (they will be treated as empty): {unknown}")

        # Token counter
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        def count_tokens(text: str) -> int:
            try:
                return len(encoding.encode(text))
            except Exception:
                return max(1, len(text) // 4)

        # Async OpenAI client
        client = AsyncOpenAI(api_key=api_key)

        # Async function to generate response
        async def generate_response(row):
            ctx = build_context(row, column_to_variable)
            formatted_user_prompt = user_prompt_template.format_map(SafeDict(ctx))
            formatted_system_prompt = system_prompt.format_map(SafeDict(ctx))

            retry_count = 0
            while retry_count < 5:
                try:
                    response = await client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": formatted_system_prompt},
                            {"role": "user", "content": formatted_user_prompt}
                        ],
                        model="gpt-4o-mini"
                    )
                    return (response.choices[0].message.content or "").strip()
                except Exception as e:
                    logging.error(f"Error generating response: {e}")
                    retry_count += 1
                    await asyncio.sleep(2 ** retry_count)
            return "ERROR"

        # Async batch processing
        async def process_batch(batch_data):
            tasks = [generate_response(row) for row in batch_data]
            return await asyncio.gather(*tasks)

        # Button to start processing
        if st.button("Generate Responses"):
            progress_bar = st.progress(0)
            response_list = []
            num_rows = len(df)
        
            async def process_data():
                processed_rows = 0
                while processed_rows < num_rows:
                    batch_data = []
                    batch_tokens = 0
                    batch_size = 10

                    for i in range(batch_size):
                        if processed_rows + i >= num_rows:
                            break
                        row = df.iloc[processed_rows + i]
                        ctx = build_context(row, column_to_variable)

                        user_prompt_tokens = count_tokens(user_prompt_template.format_map(SafeDict(ctx)))
                        system_prompt_tokens = count_tokens(system_prompt.format_map(SafeDict(ctx)))
                        total_prompt_tokens = user_prompt_tokens + system_prompt_tokens
                        batch_tokens += total_prompt_tokens

                        if batch_tokens > TOKEN_LIMIT - SAFETY_MARGIN:
                            break
                        batch_data.append(row)

                    if not batch_data:
                        break

                    responses = await process_batch(batch_data)
                    for row, response in zip(batch_data, responses):
                        response_list.append([row[col] for col in columns] + [response])

                    processed_rows += len(batch_data)
                    progress_bar.progress(processed_rows / num_rows)

            asyncio.run(process_data())
            response_df = pd.DataFrame(response_list, columns=columns + ['Response'])
            csv_data = response_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download as CSV", csv_data, "responses.csv", "text/csv")
