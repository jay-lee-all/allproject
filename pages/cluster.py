import streamlit as st
import pandas as pd
from konlpy.tag import Okt
from tqdm import tqdm
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
import numpy as np
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import RegexParser
import os
import toml

# Load secrets
secrets = toml.load(".streamlit/secrets.toml")
os.environ["OPENAI_API_KEY"] = secrets["OPENAI_KEY"]

# Streamlit UI setup
st.title("HDBSCAN Clustering and LangChain Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("File uploaded successfully!")

    # HDBSCAN parameter sliders
    min_cluster_size = st.slider(
        "Minimum Cluster Size", min_value=2, max_value=20, value=5
    )
    min_samples = st.slider("Minimum Samples", min_value=1, max_value=10, value=2)
    cluster_selection_epsilon = st.slider(
        "Cluster Selection Epsilon", min_value=0.0, max_value=0.1, step=0.01, value=0.01
    )

    # Topic input
    topic = st.text_input("Enter topic for relevance evaluation", "금융")

    # Button to run clustering
    if st.button("Run Clustering"):
        # Extract nouns from text using Okt tokenizer
        okt = Okt()
        noun_list = []
        for content in tqdm(df["text"]):
            nouns = okt.nouns(content)
            noun_list.append(nouns)

        df["nouns"] = noun_list

        # Drop rows with no nouns
        df = df[df["nouns"].map(len) > 0]
        df.reset_index(drop=True, inplace=True)

        # Prepare text for clustering
        text = [" ".join(noun) for noun in df["nouns"]]

        # TF-IDF Vectorization
        tfidf_vectorizer = TfidfVectorizer(min_df=3, ngram_range=(1, 3))
        tfidf_vectorizer.fit(text)
        vector = tfidf_vectorizer.transform(text).toarray()

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

        # Proceed button to continue
        if st.button("Proceed with Analysis"):
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

            # Only process the first 10 clusters for the example
            organized_df = organized_df.head(10)

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
                    # Use the invoke method to run the chain and parse the output
                    parsed_output = chain.invoke(inputs)
                    results.append(
                        (parsed_output["label"], int(parsed_output["related"]))
                    )
                return results

            # Generate labels and relevance ratings using the LLMChain
            results = generate_labels_and_relevance(
                organized_df["text"].tolist(), topic
            )
            organized_df["label"], organized_df["related"] = zip(*results)

            # Display the resulting DataFrame
            st.write("Clustered data with labels and relevance scores:")
            st.dataframe(organized_df)
