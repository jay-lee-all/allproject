import streamlit as st

st.set_page_config(page_title="Home", page_icon=":house:", layout="wide")

st.title("Home")

st.markdown(
    """
이 페이지에서는 다양한 데이터를 처리하고 분석하는 도구를 제공합니다. 아래에서 각 페이지의 기능을 확인하고, 필요한 분석을 수행해 보세요.
"""
)

# `clean.py`에 대한 설명
st.subheader("데이터 정리 도구")
st.markdown(
    """
데이터 정리 도구는 사용자, 에이전트, 그리고 봇 간의 대화 데이터를 기본 또는 페어링으로 처리하는 기능을 제공합니다. 
- **기본 처리**: 선택한 유형(사용자, 에이전트, 봇)의 메시지를 필터링하고 지정된 날짜 범위 내에서 데이터를 정리합니다.
- **페이링 처리**: 사용자와 봇 간의 쌍을 이루는 대화를 중심으로 데이터를 정리하고 날짜 범위 필터링을 제공합니다.
"""
)

# `cluster.py`에 대한 설명
st.subheader("사용자 데이터 클러스터링 도구")
st.markdown(
    """
사용자 데이터 클러스터링 도구는 업로드된 엑셀 파일을 사용하여 텍스트 데이터를 분석하고 그룹화하는 기능을 제공합니다.
- **HDBSCAN 클러스터링**: 텍스트 데이터의 유사성을 바탕으로 데이터를 클러스터링하고, 노이즈와 클러스터의 개수를 제공합니다.
- **주제 및 관련성 평가**: 각 클러스터 내의 질문들에 대해 주제를 생성하고, 지정된 주제와의 관련성을 평가합니다.
- **결과 시각화**: 클러스터 결과를 트리맵으로 시각화하여 각 클러스터의 크기와 주제를 한눈에 볼 수 있습니다.
"""
)

st.subheader("대량 테스트 도구")
st.markdown(
    """
Intent Classifier Tester는 사용자가 제공한 질문과 정답 데이터를 기반으로, 알리 스킬 API를 이용하여 실제 응답을 비교하고 정확도를 평가합니다.
- **API Key 및 Skill ID 입력**: API Key와 Skill ID를 입력하여 알리 스킬 API와 연결합니다.
- https://docs.allganize.ai/api_documentation/ko-1/untitled#executing-a-skill
- **엑셀 파일 업로드**: 제공된 샘플 형식에 맞는 엑셀 파일을 업로드하여 테스트를 진행합니다.
- **결과 확인 및 다운로드**: 테스트가 완료되면 정확도 정보를 화면에서 확인하고, 결과 파일을 다운로드할 수 있습니다.
- Sheet 1: Questions and comparison, Sheet 2: Accuracy, Sheet 3: Wrong answer list
"""
)

# 페이지 링크 안내
st.markdown(
    """
페이지에 대한 자세한 정보는 왼쪽 사이드바에서 확인하실 수 있습니다. 필요한 데이터를 업로드하고 분석을 시작하세요!
"""
)
