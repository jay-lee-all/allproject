import streamlit as st
import pandas as pd
import openpyxl
import io
from io import BytesIO
import tempfile

st.title("Chatbot Data Analysis")

st.write(
    "Upload your chatbot conversation file and category reference file to analyze the data"
)

# File upload widgets
conversation_file = st.file_uploader(
    "Upload conversation file (Excel/CSV)", type=["xlsx", "csv"]
)
category_file = st.file_uploader(
    "Upload category reference file (Excel)", type=["xlsx"]
)


def process_and_categorize_data(conversation_df, category_file_path):
    """
    Process chatbot data and add categories based on reference file
    """
    # Process chatbot data
    # Find all user message columns
    user_columns = [col for col in conversation_df.columns if col.startswith("user.")]
    created_at_columns = [
        col for col in conversation_df.columns if col.startswith("created_at.")
    ]

    output_df = pd.DataFrame(
        columns=["First Name", "Last Name", "UserID", "question", "date", "time"]
    )

    for index, row in conversation_df.iterrows():
        first_name = row.get("First Name", "")
        last_name = row.get("Last Name", "")
        user_id = row.get("UserID", "")

        # Process each user message
        for i in range(len(user_columns)):
            user_col = f"user.{i}"
            created_at_col = f"created_at.{i}"

            if (
                user_col in conversation_df.columns
                and created_at_col in conversation_df.columns
            ):
                if pd.notna(row[user_col]) and row[user_col] != "":
                    # Convert timestamp to date and time components
                    if pd.notna(row[created_at_col]):
                        timestamp = pd.to_datetime(row[created_at_col])
                        date = timestamp.date()
                        time = timestamp.time()

                        temp_df = pd.DataFrame(
                            {
                                "First Name": [first_name],
                                "Last Name": [last_name],
                                "UserID": [user_id],
                                "question": [row[user_col]],
                                "date": [date],
                                "time": [time],
                            }
                        )
                        output_df = pd.concat([output_df, temp_df], ignore_index=True)

    # Now add category information
    enriched_data = output_df.copy()
    enriched_data["category"] = "Unknown"

    # Load the Excel file for categories
    workbook = openpyxl.load_workbook(category_file_path)

    # Categories are the last five sheets
    categories = list(workbook.sheetnames)[-5:]

    # Create a dictionary to map questions to categories
    question_to_category = {}

    # Process each category tab
    for category in categories:
        # Read the tab as a DataFrame
        category_df = pd.read_excel(category_file_path, sheet_name=category)

        # Get the 'Sentence' column which contains questions
        if "Sentence" in category_df.columns:
            # For each row in the 'Sentence' column
            for sentence in category_df["Sentence"].dropna():
                # Split the sentence by newline character to get individual questions
                questions = sentence.strip().split("\n")

                # Map each question to this category
                for q in questions:
                    q = q.strip()
                    if q:  # Only add non-empty questions
                        question_to_category[q] = category

    # Match questions in processed data to categories
    for idx, row in enriched_data.iterrows():
        question = row["question"].strip()
        if question in question_to_category:
            enriched_data.at[idx, "category"] = question_to_category[question]

    return enriched_data


def count_questions_by_user(processed_df, exclude_category="기타"):
    """
    Count questions by user, excluding a specific category if needed
    """
    if processed_df.empty:
        st.warning("Warning: The input dataframe is empty.")
        return pd.DataFrame(columns=["First Name", "Last Name", "UserID", "count"])

    # Filter out the excluded category if specified
    if exclude_category:
        filtered_df = processed_df[processed_df["category"] != exclude_category]
    else:
        filtered_df = processed_df

    if filtered_df.empty:
        st.warning(f"No data after excluding '{exclude_category}' category.")
        return pd.DataFrame(columns=["First Name", "Last Name", "UserID", "count"])

    user_counts = filtered_df.groupby("UserID").size().reset_index(name="count")
    user_info = filtered_df[["UserID", "First Name", "Last Name"]].drop_duplicates(
        "UserID"
    )
    result = pd.merge(user_info, user_counts, on="UserID", how="right")
    result = result[["First Name", "Last Name", "UserID", "count"]]
    result = result.sort_values("count", ascending=False).reset_index(drop=True)
    return result


def to_excel_with_colors(df):
    """
    Convert DataFrame to Excel file with color highlighting for 기타 category
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Write the full data to the first sheet
        df.to_excel(writer, sheet_name="Processed Data", index=False)

        # Create a filtered version excluding 기타 category
        filtered_df = df[df["category"] != "기타"]

        # Calculate user statistics on the filtered data
        user_stats = count_questions_by_user(df, exclude_category="기타")
        user_stats.to_excel(writer, sheet_name="User Statistics", index=False)

        # Apply conditional formatting to highlight 기타 category
        workbook = writer.book
        worksheet = writer.sheets["Processed Data"]

        # Find the column index for 'question' and 'category'
        question_col = None
        category_col = None
        for i, col_name in enumerate(df.columns, start=1):
            if col_name == "question":
                question_col = i
            elif col_name == "category":
                category_col = i

        if question_col and category_col:
            # Apply conditional formatting to highlight rows where category is '기타'
            for row_idx, row in enumerate(
                df.iterrows(), start=2
            ):  # Start from 2 to account for header
                if row[1]["category"] == "기타":
                    cell = worksheet.cell(row=row_idx, column=question_col)
                    cell.fill = openpyxl.styles.PatternFill(
                        start_color="FF7F7F", end_color="FF7F7F", fill_type="solid"
                    )

    output.seek(0)
    return output


if conversation_file and category_file:
    st.write("Processing your files...")

    # Read the uploaded files
    try:
        if conversation_file.name.endswith(".csv"):
            conversation_df = pd.read_csv(conversation_file)
        else:
            conversation_df = pd.read_excel(conversation_file)

        # Save category file to a temporary location
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp.write(category_file.getvalue())
            category_file_path = tmp.name

        # Process and categorize the data
        processed_data = process_and_categorize_data(
            conversation_df, category_file_path
        )

        # Show a sample of the processed data
        st.subheader("Sample of Processed Data")
        st.dataframe(processed_data.head(10))

        # Category distribution
        st.subheader("Category Distribution")
        category_counts = processed_data["category"].value_counts().reset_index()
        category_counts.columns = ["Category", "Count"]
        st.bar_chart(category_counts.set_index("Category"))

        # Generate Excel file with color highlighting
        excel_output = to_excel_with_colors(processed_data)

        # Provide download button
        st.download_button(
            label="Download Processed Data",
            data=excel_output,
            file_name="processed_chatbot_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload both files to begin processing")

st.sidebar.header("Instructions")
st.sidebar.markdown(
    """
## How to use this tool:
1. Upload your conversation file (Excel or CSV)
2. Upload your category reference file (Excel)
3. The app will process the data and add categories
4. Questions in the '기타' category will be highlighted in red
5. User statistics will exclude questions in the '기타' category
6. Download the processed data as an Excel file
"""
)
