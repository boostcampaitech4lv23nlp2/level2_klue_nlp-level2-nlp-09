import argparse
import os
import pickle as pickle
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from dashboard.utils import connect_remote, download_model, get_filtered_result, test

st.set_page_config(page_icon="❄️", page_title="Into the RE (Remote.ver)", layout="wide")

st.title("Into the Re")


@st.experimental_memo
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")


def app(args):
    """Run streamlit app"""
    valid_data_df = pd.read_csv(args.valid_file_path)

    outputs, metrics = test(args)
    filtered_df = get_filtered_result(outputs, valid_data_df)
    csv = convert_df(filtered_df)

    st.dataframe(filtered_df)

    st.download_button("Download", csv, "result.csv", "text/csv")
    st.text(f"전체 {len(valid_data_df)} 중 {len(filtered_df)}개를 틀렸습니다.")
    st.text(
        f"micro_f1_score: {metrics['eval_micro_f1_score']} eval_auprc: {metrics['eval_auprc']} eval_accuracy: {metrics['eval_accuracy']}"
    )
    st.text("실제 정답 분포")
    st.bar_chart(filtered_df["answer"].value_counts())
    st.text("예측 라벨 분포")
    st.bar_chart(filtered_df["pred_label"].value_counts())


parser = argparse.ArgumentParser()

parser.add_argument("--model_name", default="klue/bert-base", type=str)
parser.add_argument(
    "--model_dir",
    default="dashboard/download_model",
    type=str,
)
parser.add_argument(
    "--valid_file_path",
    default="dataset/train/valid.csv",
    type=str,
)
parser.add_argument(
    "--seed",
    default=404,
    type=int,
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
