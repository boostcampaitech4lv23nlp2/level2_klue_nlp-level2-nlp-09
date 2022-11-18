import argparse
import pickle as pickle

import pandas as pd
import streamlit as st
from utils import get_filtered_result, test


def app(args):
    """Run streamlit app"""
    test_df = pd.read_csv(args.valid_data_path)

    st.set_page_config(page_icon="❄️", page_title="Into the RE", layout="wide")

    st.title("Into the Re")

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
    default="src/best_model",
    type=str,
)
parser.add_argument(
    "--valid_data_path",
    default="dataset/train/dev.csv",
    type=str,
)


args = parser.parse_args()

app(args)
