"""
GCVS + ASAS-SN 데이터로 머신러닝 모델을 학습하는 실행 파일

전체 흐름
1. gcvs5.txt 읽기
2. asassn_catalog_full.csv 읽기
3. 두 데이터를 하나로 합치기
4. 연구 대상 6개 그룹의 학습 데이터 생성
5. 학습 데이터와 테스트 데이터 분리
6. Random Forest 모델 학습
7. 테스트 데이터로 성능 평가
8. 예측 확률 Top 3 결과 저장
"""

import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split

from config import (
    GCVS_PATH,
    ASASSN_PATH,
    MODEL_DIR,
    RESULT_DIR,
    MODEL_PATH,
    ALL_FEATURES,
    RANDOM_SEED,
)
from gcvs_parser import parse_gcvs_txt
from asassn_parser import parse_asassn_csv
from ml_pipeline import (
    make_training_data,
    build_model_pipeline,
    make_top3_probability_table,
)


def make_run_result_dir():
    """
    실행할 때마다 결과 폴더를 새로 생성.
    """
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = RESULT_DIR / now
    os.makedirs(path, exist_ok=True)

    return path


def print_compact_top3_examples(test_prob_df, n=10):
    """
    테스트 예측 예시를 터미널에 보기 좋게 출력.

    저장되는 CSV에는 top1_prob, top2_prob 같은 원확률도 그대로 남기고,
    화면 출력에서는 percent 컬럼만 보여준다.
    """
    show_cols = [
        "name",
        "true_label",
        "top1_label",
        "top1_percent",
        "top2_label",
        "top2_percent",
        "top3_label",
        "top3_percent",
        "other_percent",
    ]

    show_cols = [col for col in show_cols if col in test_prob_df.columns]

    print(f"\n테스트 예측 예시 {n}개:")
    print(test_prob_df[show_cols].head(n).to_string(index=False))


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    result_dir = make_run_result_dir()

    print("=" * 70)
    print("[1] GCVS + ASAS-SN 파일 읽기")

    print(f"GCVS 파일: {GCVS_PATH}")
    df_gcvs = parse_gcvs_txt(GCVS_PATH)

    if os.path.exists(ASASSN_PATH):
        print(f"ASAS-SN 파일: {ASASSN_PATH}")
        df_asassn = parse_asassn_csv(ASASSN_PATH)
        df_raw = pd.concat([df_gcvs, df_asassn], ignore_index=True)
    else:
        print(f"ASAS-SN 파일 없음: {ASASSN_PATH}")
        print("GCVS만 사용해서 학습 진행")
        df_raw = df_gcvs

    parsed_path = result_dir / "parsed_gcvs_asassn_all.csv"
    df_raw.to_csv(parsed_path, index=False, encoding="utf-8-sig")

    print(f"전체 파싱 데이터 수: {len(df_raw):,}")
    print(f"저장: {parsed_path}")

    print("\n데이터 출처별 개수:")
    print(df_raw["source"].value_counts(dropna=False).to_string())

    print("\n" + "=" * 70)
    print("[2] 학습 데이터 생성")

    df_ml = make_training_data(df_raw)

    training_path = result_dir / "training_dataset_grouped.csv"
    df_ml.to_csv(training_path, index=False, encoding="utf-8-sig")

    print(f"학습 사용 데이터 수: {len(df_ml):,}")
    print(f"저장: {training_path}")

    print("\n라벨별 데이터 수:")
    label_counts = df_ml["label"].value_counts().sort_index()

    for label, count in label_counts.items():
        print(f"- {label:12s}: {count:,}")

    label_counts_path = result_dir / "label_counts.csv"
    label_counts.to_csv(label_counts_path, header=["count"], encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("[3] 학습 데이터 / 테스트 데이터 분리")

    X = df_ml[ALL_FEATURES]
    y = df_ml["label"]

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X,
        y,
        df_ml.index,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print(f"학습 데이터: {len(X_train):,}")
    print(f"테스트 데이터: {len(X_test):,}")
    print("설명: 테스트 데이터는 모델이 학습 중에 보지 않은 데이터로, 성능 확인에 사용")

    print("\n" + "=" * 70)
    print("[4] Random Forest 모델 학습")

    model = build_model_pipeline()
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)

    print(f"모델 저장: {MODEL_PATH}")

    print("\n" + "=" * 70)
    print("[5] 테스트 데이터로 성능 평가")

    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, digits=4)

    print("\n[Classification Report]")
    print(report)

    report_path = result_dir / "classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    labels_sorted = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)

    cm_csv_path = result_dir / "confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 7))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sorted)
    display.plot(ax=ax, xticks_rotation=45, values_format="d")
    plt.title("Confusion Matrix - GCVS + ASAS-SN Classifier")
    plt.tight_layout()

    cm_png_path = result_dir / "confusion_matrix.png"
    plt.savefig(cm_png_path, dpi=200)
    plt.close()

    print(f"평가 리포트 저장: {report_path}")
    print(f"Confusion Matrix CSV 저장: {cm_csv_path}")
    print(f"Confusion Matrix 그림 저장: {cm_png_path}")

    print("\n" + "=" * 70)
    print("[6] 테스트 데이터 예측 확률 Top 3 저장")

    test_objects = df_ml.loc[test_idx].reset_index(drop=True)
    X_test_reset = X_test.reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)

    test_prob_df = make_top3_probability_table(
        model=model,
        X=X_test_reset,
        names=test_objects["name"],
        true_labels=y_test_reset,
    )

    test_prob_path = result_dir / "test_prediction_top3.csv"
    test_prob_df.to_csv(test_prob_path, index=False, encoding="utf-8-sig")

    print(f"테스트 예측 Top 3 저장: {test_prob_path}")

    print_compact_top3_examples(test_prob_df, n=10)

    print("\n" + "=" * 70)
    print("[완료]")
    print(f"결과 폴더: {result_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()