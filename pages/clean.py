import streamlit as st
import pandas as pd


def process_chatbot_data_basic(
    df, include_types=["bot", "agent", "user"], start_date=None, end_date=None
):
    """
    Basic processing of chatbot data.
    """
    relevant_columns = [
        col
        for col in df.columns
        if "bot." in col or "agent." in col or "user." in col or "created_at." in col
    ]
    df_relevant = df[relevant_columns]
    output_df = pd.DataFrame(columns=["type", "text", "time"])

    for index, row in df_relevant.iterrows():
        for col in range(len(relevant_columns) // 4):
            bot_col = f"bot.{col}"
            agent_col = f"agent.{col}"
            user_col = f"user.{col}"
            created_at_col = f"created_at.{col}"

            if "bot" in include_types and pd.notna(row[bot_col]) and row[bot_col] != "":
                temp_df = pd.DataFrame(
                    {
                        "type": ["bot"],
                        "text": [row[bot_col]],
                        "time": [row[created_at_col]],
                    }
                )
                if not temp_df.empty and not temp_df.isna().all(axis=None):
                    output_df = pd.concat([output_df, temp_df], ignore_index=True)
            elif (
                "agent" in include_types
                and pd.notna(row[agent_col])
                and row[agent_col] != ""
            ):
                temp_df = pd.DataFrame(
                    {
                        "type": ["agent"],
                        "text": [row[agent_col]],
                        "time": [row[created_at_col]],
                    }
                )
                if not temp_df.empty and not temp_df.isna().all(axis=None):
                    output_df = pd.concat([output_df, temp_df], ignore_index=True)
            elif (
                "user" in include_types
                and pd.notna(row[user_col])
                and row[user_col] != ""
            ):
                temp_df = pd.DataFrame(
                    {
                        "type": ["user"],
                        "text": [row[user_col]],
                        "time": [row[created_at_col]],
                    }
                )
                if not temp_df.empty and not temp_df.isna().all(axis=None):
                    output_df = pd.concat([output_df, temp_df], ignore_index=True)

    output_df["time"] = pd.to_datetime(output_df["time"]).dt.date

    if start_date:
        start_date = pd.to_datetime(start_date).date()
        output_df = output_df[output_df["time"] >= start_date]

    if end_date:
        end_date = pd.to_datetime(end_date).date()
        output_df = output_df[output_df["time"] <= end_date]

    output_df = output_df[output_df["time"].notna() & (output_df["time"] != "")]

    return output_df


def process_chatbot_data_advanced(df, start_date=None, end_date=None):
    """
    Advanced processing of chatbot data.
    """
    relevant_columns = [
        col
        for col in df.columns
        if "bot." in col or "agent." in col or "user." in col or "created_at." in col
    ]
    df_relevant = df[relevant_columns]

    # Get first and last date in the file for default values
    dates = pd.to_datetime(df.filter(like="created_at").stack().values).date
    default_start_date, default_end_date = dates.min(), dates.max()

    output_df = pd.DataFrame(
        columns=["First Name", "Last Name", "UserID", "ask", "answer", "date", "time"]
    )

    for index, row in df.iterrows():
        for col in range(1, len(relevant_columns) // 4):  # Start from user.1
            user_col = f"user.{col}"
            bot_col = f"bot.{col+1}"
            created_at_col_user = f"created_at.{col}"
            created_at_col_bot = f"created_at.{col+1}"

            if (
                pd.notna(row[user_col])
                and row[user_col] != ""
                and pd.notna(row[bot_col])
                and row[bot_col] != ""
            ):
                ask_text = row[user_col]
                answer_text = row[bot_col]
                answer_time = pd.to_datetime(row[created_at_col_bot])

                # Ensure answer_time is not NaT
                if pd.notna(answer_time):
                    temp_df = pd.DataFrame(
                        {
                            "First Name": [row["First Name"]],
                            "Last Name": [row["Last Name"]],
                            "UserID": [row["UserID"]],
                            "ask": [ask_text],
                            "answer": [answer_text],
                            "date": [answer_time.date()],
                            "time": [answer_time.time()],
                        }
                    )

                    if not temp_df.empty and not temp_df.isna().all(axis=None):
                        output_df = pd.concat([output_df, temp_df], ignore_index=True)

    # Date range filtering
    if start_date:
        start_date = pd.to_datetime(start_date).date()
        output_df = output_df[output_df["date"] >= start_date]

    if end_date:
        end_date = pd.to_datetime(end_date).date()
        output_df = output_df[output_df["date"] <= end_date]

    return output_df, default_start_date, default_end_date


# Streamlit UI

st.title("Chat Data Processor")

tab1, tab2 = st.tabs(["Basic Processing", "Advanced Processing"])

with tab1:
    st.header("Basic Processing")

    uploaded_file = st.file_uploader(
        "Upload your Excel file", type=["xlsx"], key="file_uploader_basic"
    )

    if uploaded_file:
        df = pd.read_excel(uploaded_file)

        st.subheader("Select types to include:")
        include_bot = st.checkbox("bot", value=True, key="include_bot_basic")
        include_agent = st.checkbox("agent", value=True, key="include_agent_basic")
        include_user = st.checkbox("user", value=True, key="include_user_basic")

        include_types = []
        if include_bot:
            include_types.append("bot")
        if include_agent:
            include_types.append("agent")
        if include_user:
            include_types.append("user")

        dates = pd.to_datetime(df.filter(like="created_at").stack().values).date
        default_start_date, default_end_date = dates.min(), dates.max()

        st.subheader("Select date range:")
        start_date = st.date_input(
            "Start date", value=default_start_date, key="start_date_basic"
        )
        end_date = st.date_input(
            "End date", value=default_end_date, key="end_date_basic"
        )

        if st.button("Process", key="process_basic"):
            processed_data = process_chatbot_data_basic(
                df,
                include_types=include_types,
                start_date=start_date,
                end_date=end_date,
            )
            st.subheader("Preview of processed data:")
            st.write(processed_data.head(20))

            st.subheader("Download processed data:")
            processed_data.to_excel("processed_conversation_data.xlsx", index=False)
            with open("processed_conversation_data.xlsx", "rb") as f:
                st.download_button(
                    "Download Excel file",
                    f,
                    file_name="processed_conversation_data.xlsx",
                )

with tab2:
    st.header("Paired Processing")

    uploaded_file = st.file_uploader(
        "Upload your Excel file", type=["xlsx"], key="file_uploader_advanced"
    )

    if uploaded_file:
        df = pd.read_excel(uploaded_file)

        processed_data, default_start_date, default_end_date = (
            process_chatbot_data_advanced(df)
        )

        st.subheader("Select date range:")
        start_date = st.date_input(
            "Start date", value=default_start_date, key="start_date_advanced"
        )
        end_date = st.date_input(
            "End date", value=default_end_date, key="end_date_advanced"
        )

        if st.button("Process", key="process_advanced"):
            processed_data = process_chatbot_data_advanced(
                df, start_date=start_date, end_date=end_date
            )[0]
            st.subheader("Preview of processed data:")
            st.write(processed_data.head(20))

            st.subheader("Download processed data:")
            processed_data.to_excel(
                "processed_conversation_data_advanced.xlsx", index=False
            )
            with open("processed_conversation_data_advanced.xlsx", "rb") as f:
                st.download_button(
                    "Download Excel file",
                    f,
                    file_name="processed_conversation_data_advanced.xlsx",
                )
