import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
import openai
import warnings
import umap
import hdbscan
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import RegexParser
import os
import matplotlib.pyplot as plt
import streamlit as st
import toml

warnings.simplefilter(action="ignore", category=FutureWarning)
secrets = toml.load(".streamlit/secrets.toml")
os.environ["OPENAI_API_KEY"] = secrets["OPENAI_KEY"]


# Main function that processes the file and generates the Excel output
def process_file(file):
    df = pd.read_excel(file)
    user_columns = [col for col in df.columns if col.startswith("user.")]
    created_at_columns = [col for col in df.columns if col.startswith("created_at.")]

    user_interactions = pd.DataFrame()

    for user_col, created_at_col in zip(user_columns, created_at_columns):
        temp_df = df[[user_col, created_at_col]].dropna()
        temp_df.columns = ["text", "time"]
        user_interactions = pd.concat([user_interactions, temp_df], ignore_index=True)

    user_interactions["time"] = pd.to_datetime(
        user_interactions["time"], errors="coerce"
    )
    user_interactions["date"] = user_interactions["time"].dt.date

    user_interactions = user_interactions[user_interactions["date"].notna()]
    user_interactions["user_id"] = df["UserID"]

    daily_summary = (
        user_interactions.groupby("date")
        .agg(
            unique_users=("user_id", pd.Series.nunique),
            total_questions=("text", "count"),
            avg_questions_per_user=("text", lambda x: x.count() / x.nunique()),
        )
        .reset_index()
    )

    daily_summary["avg_questions_per_user"] = (
        daily_summary["total_questions"] / daily_summary["unique_users"]
    )
    daily_summary["avg_questions_per_user"] = daily_summary[
        "avg_questions_per_user"
    ].round(2)

    sentences = user_interactions["text"].tolist()
    keywords = ["임신과 출산", "육아", "제품", "멘탈", "기타"]

    openai_embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-large"
    )

    sentences_embeddings = openai_embeddings.embed_documents(sentences)
    keywords_embeddings = openai_embeddings.embed_documents(keywords)

    sentences_embeddings = np.array(sentences_embeddings)
    keywords_embeddings = np.array(keywords_embeddings)

    additional_terms = {
        "임신과 출산": [
            "임신 초기",
            "임신 증상",
            "임산부",
            "임신 기간",
            "출산 준비",
            "출산 후 회복",
            "산후조리",
            "출산 방법",
        ],
        "육아": [
            "아기",
            "발달",
            "아이 돌보기",
            "육아법",
            "자녀 교육",
            "아이 양육",
            "이유식",
            "배변",
            "분유",
            "어떻게",
            "몇시",
            "개월",
            "낮잠",
        ],
        "제품": [
            "육아 제품",
            "아기 용품",
            "기저귀",
            "추천",
            "화장지",
            "휴지",
            "물티슈",
            "하기스",
        ],
        "멘탈": ["감정", "심리", "피곤", "힘들어", "스트레스", "기분", "우울", "정신"],
        "기타": ["날씨", "챗봇", "인사", "스크립트", "이벤트"],
    }

    keyword_clusters = {keyword: [] for keyword in keywords}
    for keyword, terms in additional_terms.items():
        keyword_embeddings = openai_embeddings.embed_documents(terms)
        keyword_clusters[keyword].extend(keyword_embeddings)
        centroid = np.mean(keyword_clusters[keyword], axis=0)
        keyword_clusters[keyword] = centroid

    centroids = np.array(list(keyword_clusters.values()))
    similarity_to_centroids = cosine_similarity(sentences_embeddings, centroids)

    keywords_list = list(keyword_clusters.keys())
    category_assignment = [
        keywords_list[np.argmax(similarities)]
        for similarities in similarity_to_centroids
    ]

    user_interactions["Category"] = category_assignment

    total_questions = user_interactions["Category"].value_counts().sum()

    category_summary = (
        user_interactions.groupby("Category")
        .agg(
            unique_users=("user_id", pd.Series.nunique),
            total_questions=("text", "count"),
            avg_questions_per_user=("user_id", lambda x: x.count() / x.nunique()),
        )
        .reset_index()
    )

    category_summary["avg_questions_per_user"] = (
        category_summary["total_questions"] / category_summary["unique_users"]
    ).round(2)
    category_summary["percentage_of_total"] = (
        category_summary["total_questions"] / total_questions * 100
    ).round(2)

    categories = user_interactions["Category"].unique()
    clustered_results = {}

    for category in categories:
        print(f"\nProcessing category: {category}")
        sentences = user_interactions[user_interactions["Category"] == category][
            "text"
        ].tolist()
        openai_embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-large"
        )
        embeddings = openai_embeddings.embed_documents(sentences)
        embeddings_array = np.array(embeddings)

        umap_reducer = umap.UMAP(n_neighbors=15, n_components=5, metric="cosine")
        reduced_embeddings = umap_reducer.fit_transform(embeddings_array)
        distance_matrix = pairwise_distances(
            reduced_embeddings, metric="cosine"
        ).astype(np.float64)

        if category == "육아":
            hdbscan_model = hdbscan.HDBSCAN(
                min_cluster_size=6, min_samples=2, metric="precomputed"
            )
        else:
            hdbscan_model = hdbscan.HDBSCAN(
                min_cluster_size=5, min_samples=2, metric="precomputed"
            )

        cluster_labels = hdbscan_model.fit_predict(distance_matrix)
        cluster_df = pd.DataFrame({"Sentence": sentences, "Cluster": cluster_labels})
        cluster_df = cluster_df[cluster_df["Cluster"] != -1]
        combined_sentences = (
            cluster_df.groupby("Cluster")["Sentence"]
            .apply(lambda x: "\n".join(x))
            .reset_index()
        )

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

        llm = ChatOpenAI(model="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))
        chain = LLMChain(
            llm=llm, prompt=combined_prompt_template, output_parser=output_parser
        )

        def generate_labels_and_relevance(texts, topic):
            results = []
            for text in texts:
                inputs = {"text": text, "topic": category}
                try:
                    parsed_output = chain.run(inputs)
                    results.append(
                        (parsed_output["label"], int(parsed_output["related"]))
                    )
                except Exception as e:
                    print(f"Error processing text: {e}")
                    results.append((None, None))
            return results

        results = generate_labels_and_relevance(
            combined_sentences["Sentence"].tolist(), category
        )
        combined_sentences["label"], combined_sentences["related"] = zip(*results)

        combined_sentences["question_count"] = combined_sentences["Sentence"].apply(
            lambda x: len(x.split("\n"))
        )
        combined_sentences = combined_sentences.sort_values(
            by=["related", "question_count"], ascending=[False, False]
        )

        clustered_results[category] = combined_sentences

    # Create Excel output
    colors = ["#1F4E79", "#4A90E2", "#D6E4F0", "#ECECEC", "#34495E"]

    # Function to auto-adjust column widths
    def adjust_column_widths(df, worksheet):
        for i, col in enumerate(df.columns):
            # Convert all data in the column to strings before calculating length
            max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(i, i, max_len)

    # Function to create side-by-side layout in Excel with table outlines and headers
    def write_side_by_side(writer, sheet_name, data, col_space=1):
        worksheet = writer.book.add_worksheet(sheet_name)
        col_offset = 0

        # Define column widths to adjust
        wide_columns = [0, 4, 8, 12, 16, 20, 24]  # Corresponding to A, E, I, M, Q, U, Y
        for col in wide_columns:
            worksheet.set_column(col, col, 30)

        # Add borders for table style
        border_format = writer.book.add_format({"border": 1})

        for category, result_df in data.items():
            # Get the 'label', 'count', and 'related' columns
            category_data = result_df[["label", "question_count", "related"]].copy()
            category_data.columns = ["Label", "Count", "Related"]

            # Write the category name at the top
            worksheet.write(0, col_offset, category)

            # Add headers under the category name
            worksheet.write(1, col_offset, "Label")
            worksheet.write(1, col_offset + 1, "Count")
            worksheet.write(1, col_offset + 2, "Related")

            # Write the data under each category
            for row_idx, row in category_data.iterrows():
                worksheet.write(row_idx + 2, col_offset, row["Label"], border_format)
                worksheet.write(
                    row_idx + 2, col_offset + 1, row["Count"], border_format
                )
                worksheet.write(
                    row_idx + 2, col_offset + 2, row["Related"], border_format
                )

            # Move the column offset by the number of columns used + space between categories
            col_offset += 3 + col_space

            worksheet.autofilter(0, 0, row_idx + 2, col_offset - 1)

    def format_category_sheet(worksheet):
        # Set column A width to 30
        worksheet.set_column("A:A", 30)

        # Set column C width to 13
        worksheet.set_column("C:C", 13)

        # Set column D width to 50 and apply wrap text
        wrap_format = workbook.add_format({"text_wrap": True})
        worksheet.set_column("D:D", 50, wrap_format)

    # Create a Pandas Excel writer using XlsxWriter as the engine
    output_file = "combined_output_with_styled_summary.xlsx"

    with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
        # Write daily_summary to the first sheet
        daily_summary.to_excel(writer, sheet_name="Daily Summary", index=False)

        # Write category_summary to the second sheet
        category_summary.to_excel(writer, sheet_name="Category Summary", index=False)

        # Access the workbook and worksheet objects
        workbook = writer.book
        daily_sheet = writer.sheets["Daily Summary"]
        category_sheet = writer.sheets["Category Summary"]

        # Auto-adjust column widths for daily_summary and category_summary
        adjust_column_widths(daily_summary, daily_sheet)
        adjust_column_widths(category_summary, category_sheet)

        # Plot line graph for daily_summary
        plt.figure(figsize=(10, 6))
        dates_formatted = pd.to_datetime(daily_summary["date"]).dt.strftime("%m-%d")

        # Plot the lines with formatted dates
        plt.plot(
            dates_formatted,
            daily_summary["unique_users"],
            label="Unique Users",
            color=colors[0],
        )
        plt.plot(
            dates_formatted,
            daily_summary["total_questions"],
            label="Total Questions",
            color=colors[1],
        )
        # plt.plot(dates_formatted, daily_summary['avg_questions_per_user'], label='Avg Questions per User', color=colors[2])

        # Set labels and title
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.title("Daily Summary Over Time")
        plt.legend()
        plt.grid(True)

        # Save the plot to a temporary file
        plt.tight_layout()
        daily_summary_plot_path = "daily_summary_plot.png"
        plt.savefig(daily_summary_plot_path)
        daily_sheet.insert_image("G2", daily_summary_plot_path)

        # Plot pie chart for category_summary
        plt.figure(figsize=(6, 6))
        plt.pie(
            category_summary["percentage_of_total"],
            labels=category_summary["Category"],
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        plt.title("Category Distribution")

        # Save the pie chart to a temporary file
        plt.tight_layout()
        category_pie_chart_path = "category_pie_chart.png"
        plt.savefig(category_pie_chart_path)
        category_sheet.insert_image("G2", category_pie_chart_path)

        # Write the side-by-side category labels, counts, and related with table outlines and headers
        write_side_by_side(writer, "All Categories Summary", clustered_results)

        # Now add the category-specific sheets and apply column formatting
        for category, result_df in clustered_results.items():
            # Write each DataFrame to a separate sheet in the Excel file
            result_df[["label", "related", "question_count", "Sentence"]].to_excel(
                writer, sheet_name=category, index=False
            )

            # Access the worksheet object for the current category sheet
            category_sheet = writer.sheets[category]

            # Apply custom formatting (column width and wrap text)
            format_category_sheet(category_sheet)

    # Remove the temporary plot images after inserting them into the Excel file
    os.remove(daily_summary_plot_path)
    os.remove(category_pie_chart_path)

    return output_file


# Streamlit App
st.title("Chatbot Data Processor")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    output_excel = process_file(uploaded_file)
    st.success(f"Processing complete. Download your file below.")
    st.download_button(
        label="Download Excel",
        data=open(output_excel, "rb").read(),
        file_name="processed_output.xlsx",
    )
