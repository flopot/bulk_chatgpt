import streamlit as st
import pandas as pd
import asyncio
import logging
import csv
import os
import tiktoken
from openai import AsyncOpenAI
import string

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

# Streamlit UI
st.title('Bulk ChatGPT')
st.markdown("""by [Florian Potier](https://twitter.com/FloPots) - [Intrepid Digital](https://www.intrepidonline.com/)
""", unsafe_allow_html=True)

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
