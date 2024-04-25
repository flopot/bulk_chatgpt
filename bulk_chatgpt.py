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
    system_prompt = st.text_area("Enter your system prompt template:", value="Enter details about {keyword} and {url}.", height=100)
    user_prompt = st.text_area("Enter your user prompt template:", value="First, visit '{url}'. Then check the top 10 results for '{keyword}'.", height=100)
    
    all_responses = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Replace placeholders in the prompts with actual data
        system_message = system_prompt.format(**{col: row[col] for col in prompt_columns})
        user_message = user_prompt.format(**{col: row[col] for col in prompt_columns})

        # Function to generate responses using the OpenAI client
        def generate_response(system_message, user_message):
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                model="gpt-4"
            )
            return response.choices[0].message.content.strip()

        seo_advice = generate_response(system_message, user_message)
        all_responses.append({col: row[col] for col in prompt_columns} | {'Recommendations': seo_advice})
        time.sleep(1)  # To avoid hitting API rate limits

    # Convert the responses into a DataFrame
    results_df = pd.DataFrame(all_responses)

    # Convert DataFrame to CSV and create download button
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Content Briefs as CSV", csv, "content-briefs.csv", "text/csv")