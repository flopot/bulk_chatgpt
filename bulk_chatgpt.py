import streamlit as st
import pandas as pd
from openai import OpenAI
import time
import io

# Title and Setup
st.title('Bulk ChatGPT')

# Subtitle
st.markdown(
    """
    by [Florian Potier](https://twitter.com/FloPots) - [Intrepid Digital](https://www.intrepidonline.com/)
    """,
    unsafe_allow_html=True
)

# Input for the OpenAI API key
api_key = st.text_input("Enter your OpenAI API key", type="password")

# File upload
uploaded_file = st.file_uploader("Choose your CSV file", type=['csv'])

if uploaded_file:
    # Read the uploaded file into a DataFrame to get column names
    df = pd.read_csv(uploaded_file)
    columns = df.columns.tolist()

    # Allow user to map columns to variable names
    st.write("Map each column to a variable name that will be used in the prompts:")
    column_to_variable = {}
    for column in columns:
        variable_name = st.text_input(f"Enter a variable name for {column}", value=column)
        column_to_variable[column] = variable_name

    # System and User Prompts customization
    system_prompt = st.text_area("Edit the system prompt", value="Edit the system prompt. You can include any of the variable names defined above surrounded by curly braces, like {variable_name}.")
    user_prompt_template = st.text_area("Edit the user prompt", value="Edit the user prompt. You can include any of the variable names defined above surrounded by curly braces, like {variable_name}.")

    # Initialize the OpenAI client with the user-provided API key if entered
    if api_key and st.button("Generate Response"):
        client = OpenAI(api_key=api_key)
        all_responses = []

        # Function to generate SEO recommendations using the OpenAI client
        def generate_response(row):
            # Format the user prompt dynamically with variables from row
            formatted_user_prompt = user_prompt_template.format(**{var: row[col] for col, var in column_to_variable.items()})
            formatted_system_prompt = system_prompt.format(**{var: row[col] for col, var in column_to_variable.items()})
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": formatted_system_prompt},
                    {"role": "user", "content": formatted_user_prompt}
                ],
                model="gpt-4"
            )
            return response.choices[0].message.content.strip()

        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            response = generate_response(row)
            all_responses.append(response)
            time.sleep(1)  # To avoid hitting API rate limits

        # Convert responses into a DataFrame
        response_df = pd.DataFrame(all_responses, columns=columns + ['Response'])

        # Convert DataFrame to CSV and create download button
        csv = response_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Responses as CSV", csv, "responses.csv", "text/csv")

