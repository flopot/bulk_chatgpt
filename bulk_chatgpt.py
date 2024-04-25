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

def generate_response(client, system_message, user_message):
    try:
        response = client.chat_completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            model="gpt-4"
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        st.error(f"Failed to generate response: {str(e)}")
        return None

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

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        prompt_data = {col: str(row[col]) if pd.notna(row[col]) else "" for col in prompt_columns}
        system_message = system_prompt.format(**prompt_data)
        user_message = user_prompt.format(**prompt_data)
        
        seo_advice = generate_response(client, system_message, user_message)
        if seo_advice is not None:
            all_responses.append({col: row[col] for col in prompt_columns} | {'Recommendations': seo_advice})

    # Convert the responses into a DataFrame
    results_df = pd.DataFrame(all_responses)

    # Convert DataFrame to CSV and create download button
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Content Briefs as CSV", csv, "content-briefs.csv", "text/csv")
