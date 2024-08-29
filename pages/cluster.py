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

import hmac

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop() 

# Load secrets
secrets = toml.load(".streamlit/secrets.toml")
os.environ["OPENAI_API_KEY"] = secrets["OPENAI_KEY"]

# Streamlit UI setup
st.title("User Data Clustering")

# File uploader
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("File uploaded successfully!")

    st.markdown("#### Minimum Cluster Size")
    st.markdown(
        "<p style='font-size: 12px;'>클러스터의 최소 크기를 설정합니다. "
        "이 값이 작으면 작은 그룹이 많이 생기고, 값이 크면 큰 그룹이 적게 생깁니다.</p>",
        unsafe_allow_html=True
    )
    min_cluster_size = st.slider(
        "Minimum Cluster Size", min_value=2, max_value=20, value=5
    )

    # Minimum Samples 설명
    st.markdown("#### Minimum Samples")
    st.markdown(
        "<p style='font-size: 12px;'>그룹을 만들 때 필요한 데이터의 최소 개수입니다. "
        "값이 높을수록 클러스터링이 더 엄격해지고, 값이 낮으면 더 많은 포인트가 클러스터에 포함됩니다.</p>",
        unsafe_allow_html=True
    )
    min_samples = st.slider("Minimum Samples", min_value=1, max_value=10, value=2)

    # Cluster Selection Epsilon 설명
    st.markdown("#### Cluster Selection Epsilon")
    st.markdown(
        "<p style='font-size: 12px;'>그룹을 결정할 때 얼마나 다양한 그룹을 만들지 결정합니다. "
        "값이 낮으면 안정적인 그룹이 만들어지고, 값이 높으면 더 다양한 그룹이 만들어집니다.</p>",
        unsafe_allow_html=True
    )
    cluster_selection_epsilon = st.slider(
        "Cluster Selection Epsilon", min_value=0.0, max_value=0.1, step=0.01, value=0.01
    )

    # Topic input
    topic = st.text_input("Enter topic for relevance evaluation", "금융")

    # Button to run clustering
    if st.button("Run Clustering"):
        try:
            # Initialize OpenAIEmbeddings with your API key
            openai_embeddings = OpenAIEmbeddings(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                model="text-embedding-3-small",
            )

            # Prepare text for embeddings
            text = df["text"].tolist()

            # Get OpenAI embeddings for the text
            embeddings = openai_embeddings.embed_documents(text)

            # Convert embeddings to a format suitable for further processing
            vector = np.array(embeddings)

            # Compute Distance Matrix
            distance_matrix = pairwise_distances(vector, metric="cosine")

            # HDBSCAN Clustering
            hdbscan_model = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="precomputed",
                cluster_selection_epsilon=cluster_selection_epsilon,
            )
            result = hdbscan_model.fit_predict(distance_matrix)

            df["result"] = result
            minus_one_count = (df["result"] == -1).sum()
            num_clusters = len(set(result)) - (1 if -1 in result else 0)

            # Display noise and cluster count
            st.write(f"Noise count: {minus_one_count}")
            st.write(f"Number of clusters: {num_clusters}")

            # Store data in session state to persist it across interactions
            st.session_state["df"] = df
            st.session_state["organized_df"] = None

        except Exception as e:
            st.error(f"An error occurred during clustering: {e}")

    # Proceed button to continue analysis
    if "df" in st.session_state and st.button("Proceed with Analysis"):
        try:
            df = st.session_state["df"]

            # Count the number of rows in each cluster
            counts_df = df.groupby("result").size().reset_index(name="count")

            # Create a DataFrame with cluster text
            organized_df = (
                df.groupby("result")
                .agg(
                    {
                        "text": lambda x: "\n".join(
                            x
                        )  # Combine all text entries in each cluster with line breaks
                    }
                )
                .reset_index()
            )

            # Merge the counts back to the organized DataFrame
            organized_df = organized_df.merge(counts_df, on="result", how="left")

            # Check for repetition in each cluster
            organized_df["repeat"] = organized_df.apply(
                lambda row: (
                    1
                    if (
                        row["text"].split("\n").count(row["text"].split("\n")[0])
                        / row["count"]
                    )
                    > 0.9
                    else 0
                ),
                axis=1,
            )
            organized_df = organized_df[organized_df["result"] != -1]

            # Store organized_df in session state
            st.session_state["organized_df"] = organized_df

            # Prepare the combined prompt using LangChain's PromptTemplate
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

            # Define an output parser to parse both the label and relevance from the response
            output_parser = RegexParser(
                regex=r"주제:\s*(?P<label>.+?)\n관련성:\s*(?P<related>[1-4])",  # Regex to capture the label and relevance score
                output_keys=["label", "related"],
            )

            # Initialize ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o")

            # Create an LLMChain with the prompt template and output parser
            chain = LLMChain(
                llm=llm, prompt=combined_prompt_template, output_parser=output_parser
            )

            # Function to get cluster labels and relevance ratings using a single API call
            def generate_labels_and_relevance(texts, topic):
                results = []
                for text in texts:
                    # Prepare input for the chain
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

            # Generate labels and relevance ratings using the LLMChain
            results = generate_labels_and_relevance(
                organized_df["text"].tolist(), topic
            )
            organized_df["label"], organized_df["related"] = zip(*results)

            # Sort DataFrame to push 'repeat == 1' to the bottom, then by 'count' descending, and 'related' 2 and 1 to the bottom
            organized_df = organized_df.sort_values(
                by=["repeat", "related", "count"], ascending=[True, False, False]
            )

            # Ensure 'related' 1 and 2 are at the bottom if 'repeat' is 0
            organized_df = pd.concat(
                [
                    organized_df[
                        (organized_df["repeat"] == 0) & (organized_df["related"] >= 3)
                    ].sort_values(by="count", ascending=False),
                    organized_df[
                        (organized_df["repeat"] == 0) & (organized_df["related"] < 3)
                    ].sort_values(by="related", ascending=False),
                    organized_df[organized_df["repeat"] == 1],
                ]
            )

            # Generate labels and relevance ratings using the LLMChain
            results = generate_labels_and_relevance(
                organized_df["text"].tolist(), topic
            )
            organized_df["label"], organized_df["related"] = zip(*results)

            # Filter out clusters with 'repeat == 1' or 'related' scores of 1 or 2
            organized_df = organized_df[
                (organized_df["repeat"] != 1) & (organized_df["related"] >= 3)
            ]

            # Sort DataFrame by 'count' descending
            organized_df = organized_df.sort_values(by=["count"], ascending=False)

            # Set the number of top clusters to display to 10
            top_clusters_df = organized_df.head(10)

            # Store top_clusters_df in session state to persist it across interactions
            st.session_state["top_clusters_df"] = top_clusters_df

            # Create a Treemap using Plotly with pastel colors
            fig = px.treemap(
                top_clusters_df,
                path=[px.Constant("All"), "label"],
                values="count",
                title="Cluster Treemap",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )

            # Display the Treemap
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Download processed data:")
            organized_df.to_excel("cluster.xlsx", index=False)

            with open("cluster.xlsx", "rb") as f:
                st.download_button(
                    label="Download Excel file",
                    data=f,
                    file_name="cluster.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
