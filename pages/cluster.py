import streamlit as st
import pandas as pd
import numpy as np
import hdbscan
from sklearn.metrics import pairwise_distances
from langchain_openai import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import RegexParser
import os
import toml
import plotly.express as px
import warnings

# Suppress specific FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)

# Load secrets
secrets = toml.load(".streamlit/secrets.toml")
os.environ["OPENAI_API_KEY"] = secrets["OPENAI_KEY"]

# Streamlit UI setup
st.title("User Data Clustering and Visualization")

# Initialize session state
if "step" not in st.session_state:
    st.session_state.step = 0  # 0: Start, 1: Processed, 2: Clustered, 3: Analyzed

if "processed_data" not in st.session_state:
    st.session_state.processed_data = None

if "organized_df" not in st.session_state:
    st.session_state.organized_df = None

if "default_start_date" not in st.session_state:
    st.session_state.default_start_date = None

if "default_end_date" not in st.session_state:
    st.session_state.default_end_date = None


# Function for Basic Processing of Chatbot Data
def process_chatbot_data_basic(
    df, include_types=["bot", "agent", "user"], start_date=None, end_date=None
):
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


# File uploader
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file and st.session_state.step == 0:
    df = pd.read_excel(uploaded_file)

    st.subheader("Select types to include:")
    include_bot = st.checkbox("bot", value=True)
    include_agent = st.checkbox("agent", value=True)
    include_user = st.checkbox("user", value=True)

    include_types = []
    if include_bot:
        include_types.append("bot")
    if include_agent:
        include_types.append("agent")
    if include_user:
        include_types.append("user")

    dates = pd.to_datetime(df.filter(like="created_at").stack().values).date
    if len(dates) == 0:
        st.warning(
            "No valid 'created_at' data found. Please upload a file with appropriate date information."
        )
        st.session_state.default_start_date = None
        st.session_state.default_end_date = None
    else:
        st.session_state.default_start_date, st.session_state.default_end_date = (
            dates.min(),
            dates.max(),
        )

    if st.session_state.default_start_date and st.session_state.default_end_date:
        start_date = st.date_input(
            "Start date", value=st.session_state.default_start_date
        )
        end_date = st.date_input("End date", value=st.session_state.default_end_date)

    if st.button("Process Data"):
        st.session_state.processed_data = process_chatbot_data_basic(
            df, include_types=include_types, start_date=start_date, end_date=end_date
        )
        st.session_state.step = 1  # Move to the next step

if st.session_state.step >= 1:
    st.subheader("Processed Data Preview")
    st.write(st.session_state.processed_data.head(20))
    st.subheader("Proceed to Clustering and Visualization")

    # Clustering Parameters
    min_cluster_size = st.slider(
        "Minimum Cluster Size", min_value=2, max_value=20, value=5
    )
    min_samples = st.slider("Minimum Samples", min_value=1, max_value=10, value=2)
    cluster_selection_epsilon = st.slider(
        "Cluster Selection Epsilon", min_value=0.0, max_value=0.1, step=0.01, value=0.01
    )

    topic = st.text_input("Enter topic for relevance evaluation", "금융")

    if st.button("Run Clustering") and st.session_state.step == 1:
        try:
            openai_embeddings = OpenAIEmbeddings(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                model="text-embedding-3-small",
            )

            text = st.session_state.processed_data["text"].tolist()
            embeddings = openai_embeddings.embed_documents(text)
            vector = np.array(embeddings)
            distance_matrix = pairwise_distances(vector, metric="cosine")

            hdbscan_model = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="precomputed",
                cluster_selection_epsilon=cluster_selection_epsilon,
            )
            result = hdbscan_model.fit_predict(distance_matrix)

            st.session_state.processed_data["result"] = result
            minus_one_count = (st.session_state.processed_data["result"] == -1).sum()
            num_clusters = len(set(result)) - (1 if -1 in result else 0)

            st.write(f"Noise count: {minus_one_count}")
            st.write(f"Number of clusters: {num_clusters}")

            st.session_state.step = 2  # Move to the next step

        except Exception as e:
            st.error(f"An error occurred during clustering: {e}")

if st.session_state.step >= 2:
    if "result" in st.session_state.processed_data:
        processed_data = st.session_state.processed_data

        counts_df = processed_data.groupby("result").size().reset_index(name="count")
        organized_df = (
            processed_data.groupby("result")
            .agg({"text": lambda x: "\n".join(x)})
            .reset_index()
        )

        organized_df = organized_df.merge(counts_df, on="result", how="left")
        organized_df["unique"] = organized_df["text"].apply(
            lambda text: len(set(text.split("\n")))
        )
        organized_df = organized_df[organized_df["result"] != -1]

        st.session_state.organized_df = organized_df

        combined_prompt_template = PromptTemplate(
            input_variables=["text", "topic"],
            template=(
                """다음 텍스트는 유저들이 챗봇에 작성한 질문들입니다.
                연관성을 기반으로 군집화된 질문들을 바탕으로 한국어로 짧게 질문들의 주제/목적을 생성하고,
                생성된 주제가 지정된 주제 '{topic}'와 얼마나 관련이 있는지 평가하십시오.
                관련성은 1(전혀 관련 없음)에서 4(매우 관련 있음) 사이의 숫자로 평가되어야 합니다.
                
                텍스트: {text}
                
                출력 형식 예시:
                주제: [생성된 주제]
                관련성: [1-4 숫자]
                """
            ),
        )

        output_parser = RegexParser(
            regex=r"주제:\s*(?P<label>.+?)\n관련성:\s*(?P<related>[1-4])",
            output_keys=["label", "related"],
        )

        llm = ChatOpenAI(model="gpt-4o")
        chain = LLMChain(
            llm=llm, prompt=combined_prompt_template, output_parser=output_parser
        )

        def generate_labels_and_relevance(texts, topic):
            results = []
            for text in texts:
                inputs = {"text": text, "topic": topic}
                try:
                    parsed_output = chain.run(inputs)
                    results.append(
                        (parsed_output["label"], int(parsed_output["related"]))
                    )
                except Exception as e:
                    st.error(f"Error processing text: {e}")
                    results.append((None, None))
            return results

        results = generate_labels_and_relevance(organized_df["text"].tolist(), topic)
        organized_df["label"], organized_df["related"] = zip(*results)

        organized_df["related_sort"] = organized_df["related"].apply(
            lambda x: 4 if x >= 3 else x
        )
        organized_df = organized_df.sort_values(
            by=["related_sort", "unique", "count"], ascending=[False, False, False]
        )

        st.session_state.organized_df = organized_df
        organized_df.to_excel("clustered.xlsx", index=False)

        st.subheader("Download processed and clustered data:")
        with open("clustered.xlsx", "rb") as f:
            st.download_button(
                label="Download Clustered Excel file",
                data=f,
                file_name="clustered.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        st.subheader("Visualize Clusters and Questions")

        df_vis = organized_df[organized_df["related"].isin([3, 4])]
        df_vis["split_questions"] = df_vis["text"].apply(lambda x: x.split("\n"))
        df_exploded = df_vis.explode("split_questions").reset_index(drop=True)

        df_exploded["size"] = 1
        cluster_sizes = (
            df_exploded.groupby("label").size().reset_index(name="total_size")
        )

        total_questions = df_exploded["size"].sum()
        cluster_sizes["percentage"] = (
            cluster_sizes["total_size"] / total_questions * 100
        ).round(2)

        top_clusters = cluster_sizes.nlargest(5, "percentage").sort_values(
            "percentage", ascending=False
        )
        df_top_clusters = df_exploded[df_exploded["label"].isin(top_clusters["label"])]
        df_top_questions = (
            df_top_clusters.groupby("label")
            .apply(lambda x: x.nlargest(6, "size"))
            .reset_index(drop=True)
        )

        def wrap_text(text, length):
            if len(text) <= length:
                return text

            first_break = length
            for i in range(length, len(text)):
                if text[i] == " ":
                    first_break = i
                    break

            wrapped_text = text[:first_break] + "<br>"

            remaining_text = text[first_break:].strip()
            if len(remaining_text) > length:
                second_break = length
                for j in range(length, len(remaining_text)):
                    if remaining_text[j] == " ":
                        second_break = j
                        break
                wrapped_text += (
                    remaining_text[:second_break]
                    + "<br>"
                    + remaining_text[second_break:].strip()
                )
            else:
                wrapped_text += remaining_text

            return wrapped_text

        df_top_questions["wrapped_question"] = df_top_questions[
            "split_questions"
        ].apply(lambda x: wrap_text(x, 16))
        df_treemap = pd.merge(
            df_top_questions, top_clusters[["label", "percentage"]], on="label"
        )

        df_treemap["label_with_percentage"] = (
            df_treemap["label"] + " (" + df_treemap["percentage"].astype(str) + "%)"
        )
        df_treemap = df_treemap.sort_values(by="percentage", ascending=False)
        df_treemap["label_with_percentage"] = pd.Categorical(
            df_treemap["label_with_percentage"],
            categories=df_treemap["label_with_percentage"].unique(),
            ordered=True,
        )

        df_final = pd.DataFrame(
            {
                "label": df_treemap["label_with_percentage"],
                "question": df_treemap["wrapped_question"],
                "size": df_treemap["size"],
            }
        )

        color_mapping = {
            df_final["label"].unique()[0]: "#D3D3D3",
            df_final["label"].unique()[1]: "#e1e1e1",
            df_final["label"].unique()[2]: "#ececec",
            df_final["label"].unique()[3]: "#F5F5F5",
            df_final["label"].unique()[4]: "#FFFFFF",
        }

        fig = px.treemap(
            df_final,
            path=["label", "question"],
            values="size",
            hover_data={"label": False, "question": False},
            color="label",
            color_discrete_map=color_mapping,
        )

        fig.update_traces(
            hovertemplate="<b>Label:</b> %{label}<br><b>Count:</b> %{value}<extra></extra>",
            tiling=dict(packing="slice-dice", pad=5),
            textinfo="label+text",
            textfont_size=15,
            marker=dict(line=dict(color="#000000", width=1)),
            pathbar_textfont=dict(size=18),
        )

        fig.update_layout(
            title="Interactive Treemap of Questions by Label (Top 5 Clusters and Top 6 Questions)",
            margin=dict(t=50, l=25, r=25, b=25),
            font=dict(size=15),
        )

        st.plotly_chart(fig, use_container_width=True)
