import argparse
import pickle as pickle

import pandas as pd
import streamlit as st

from utils import get_filtered_result, test


@st.experimental_memo
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")


def app(args):
    """Run streamlit app"""
    valid_data_df = pd.read_csv(args.valid_file_path)

    st.set_page_config(page_icon="❄️", page_title="Into the RE", layout="wide")

    st.title("Into the Re")

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
    default="src/best_model",
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

args = parser.parse_args()

app(args)
