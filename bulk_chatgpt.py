import streamlit as st
import pandas as pd
from openai import OpenAI
import time

# Title and Setup
st.title('Bulk ChatGPT')

# Subtitle and Documentation
st.markdown(
    """
    by [Florian Potier](https://twitter.com/FloPots) - [Intrepid Digital](https://www.intrepidonline.com/)
    """
)

# Input for the OpenAI API key
api_key = st.text_input("Enter your OpenAI API key", type="password")

# File upload
uploaded_file = st.file_uploader("Choose your CSV file", type=['csv'])

# If a file is uploaded, show column selection and prompt customization
if uploaded_file and api_key:
    # Initialize the OpenAI client with the user-provided API key
    client = OpenAI(api_key=api_key)

    # Read the uploaded file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Show the column names to the user and let them select the relevant ones
    st.write("Columns in your CSV:", list(df.columns))
    prompt_columns = st.multiselect("Select the columns to use in prompts:", list(df.columns))
    
    # Allow users to create their own prompts
    system_prompt = st.text_area("Enter your system prompt template:", value="You can add variables between brackets (e.g. {VARIABLE1}, {VARIABLE2}).", height=100)
    user_prompt = st.text_area("Enter your user prompt template:", value="You can add variables between brackets (e.g. {VARIABLE1}, {VARIABLE2}).", height=100)
    
    all_responses = []

    # Debug output
    st.write("Selected columns for prompts:", prompt_columns)    

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
    try:
        # Create a dictionary of column data ensuring all values are strings
        prompt_data = {col: str(row[col]) if pd.notna(row[col]) else "N/A" for col in prompt_columns}

        # Dynamically generate the messages
        system_message = system_prompt.format(**prompt_data)
        user_message = user_prompt.format(**prompt_data)

        # Generate the SEO advice using the defined function
        seo_advice = generate_response(system_message, user_message)
        all_responses.append({col: row[col] for col in prompt_columns} | {'Recommendations': seo_advice})

    except KeyError as e:
        st.error(f"Missing a placeholder for '{e.args[0]}' in your prompt template. Please adjust your template.")
        continue  # Skip to the next row

    time.sleep(1)  # Pause to avoid rate limits

    # Convert the responses into a DataFrame
    results_df = pd.DataFrame(all_responses)

    # Convert DataFrame to CSV and create download button
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Content Briefs as CSV", csv, "content-briefs.csv", "text/csv")
