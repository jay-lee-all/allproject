import streamlit as st
import pandas as pd
import requests
import json
import time

st.title("Intent Classifier Tester")

st.download_button(
    label="Download Sample Input File",
    data=open("example.xlsx", "rb").read(),
    file_name="example.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
submit_button = st.button("Submit")

if uploaded_file and submit_button:
    df = pd.read_excel(uploaded_file)

    url = "https://backend.alli.ai/webapi/skill"
    headers = {
        "API-KEY": "SU7GGVZERW8O5PBMDDJTQ2N3UCF88U8D4U",
        "Content-Type": "application/json",
    }

    def get_skill_response(question):
        payload = {
            "id": "Q2FtcGFpZ246NjZmZjU4M2NkMjY1MjIyMDc2NGE4ODg4",
            "text": question,
            "variables": {},
        }
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            result = response.json()
            return result.get("result", "") if not result["errors"] else None
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")
            return None

    correct_count = 0

    for index, row in df.iterrows():
        question = row["Question"]
        correct_answer = row["Answer"]

        outcome = get_skill_response(question)
        df.at[index, "Outcome"] = outcome

        if outcome == correct_answer:
            correct_count += 1

        time.sleep(1)

    total_questions = len(df)
    accuracy = correct_count / total_questions * 100
    grouped = df.groupby("Answer")
    accuracy_details = []
    mismatch_details = {answer: [] for answer in df["Answer"].unique()}

    accuracy_details.append(
        {
            "Answer Type": "Total",
            "Total Count": total_questions,
            "Correct Count": correct_count,
            "Wrong Count": total_questions - correct_count,
            "Accuracy (%)": round(accuracy, 2),
        }
    )

    for answer, group in grouped:
        correct_count = (group["Answer"] == group["Outcome"]).sum()
        total_count = len(group)
        wrong_count = total_count - correct_count
        accuracy = correct_count / total_count * 100
        accuracy_details.append(
            {
                "Answer Type": answer,
                "Total Count": total_count,
                "Correct Count": correct_count,
                "Wrong Count": wrong_count,
                "Accuracy (%)": round(accuracy, 2),
            }
        )

        wrong_questions = group[group["Answer"] != group["Outcome"]][
            "Question"
        ].tolist()
        mismatch_details[answer].extend(wrong_questions)

    accuracy_df = pd.DataFrame(accuracy_details)
    mismatch_df = pd.DataFrame(
        dict([(k, pd.Series(v)) for k, v in mismatch_details.items()])
    )

    output_path = "results.xlsx"
    with pd.ExcelWriter(output_path) as writer:
        df.to_excel(writer, sheet_name="Results", index=False)
        accuracy_df.to_excel(writer, sheet_name="Accuracy", index=False)
        mismatch_df.to_excel(writer, sheet_name="Mismatch", index=False)

    st.write("### Accuracy Information")
    st.dataframe(accuracy_df)

    st.download_button(
        label="Download Results File",
        data=open(output_path, "rb").read(),
        file_name="results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
