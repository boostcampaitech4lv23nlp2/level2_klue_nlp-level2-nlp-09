import argparse
import pickle as pickle

import pandas as pd
import streamlit as st

from utils import connect_remote, download_model, get_filtered_result, test

st.set_page_config(page_icon="❄️", page_title="Into the RE (Remote.ver)", layout="wide")
st.title("Into the Re")


def app(args):
    """Run streamlit app"""

    test_df = pd.read_csv(args.valid_data_path)

    result_df = test(args)
    filtered_df = get_filtered_result(result_df, test_df)

    st.dataframe(filtered_df)
    st.text(f"전체 {len(test_df)} 중 {len(filtered_df)}개를 틀렸습니다.")
    st.text("실제 정답 분포")
    st.bar_chart(filtered_df["answer"].value_counts())
    st.text("예측 라벨 분포")
    st.bar_chart(filtered_df["pred_label"].value_counts())


parser = argparse.ArgumentParser()

parser.add_argument("--model_name", default="klue/bert-base", type=str)
parser.add_argument(
    "--model_dir",
    # default="src/best_model",
    default="dashboard/download_model",
    type=str,
)
parser.add_argument(
    "--valid_data_path",
    default="dataset/train/dev.csv",
    type=str,
)


def model_selector():
    option = st.selectbox("모델을 선택하세요!!", connect_remote())

    st.write("선택한 모델: ", option)
    if st.button("이 모델을 다운로드하여 학습을 진행할까요??"):
        download_model(option)
        return True
    else:
        return False


# args = parser.parse_args()

# app(args)

if model_selector():
    args = parser.parse_args()
    app(args)
else:
    st.write("버튼을 누르면 바로 실행됩니다!")
