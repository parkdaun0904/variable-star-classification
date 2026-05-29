# make_sample_new_data.py
"""
새 데이터 예측 연습용 CSV 파일 생성

아직 실제 관측 데이터가 없을 때,
predict_new.py가 제대로 작동하는지 확인하기 위한 샘플 파일을 만듦.

생성되는 파일
new_data/sample_new_objects.csv

주의
- 아래 값들은 실제 관측값이 아니라 예시용 값
- 각 변광성 타입의 대략적인 특징을 흉내 낸 가상 데이터
- 모델 실행 흐름 확인용으로 사용
"""

import os
import pandas as pd

from config import NEW_DATA_DIR


def main():
    os.makedirs(NEW_DATA_DIR, exist_ok=True)

    sample_data = pd.DataFrame([
        {
            "name": "sample_RR_Lyrae_like",
            "period": 0.55,
            "mag_max": 12.10,
            "mag_min": 13.00,
            "amplitude": 0.90,
            "epoch": 53000.0,
            "rise_time": 35.0,
            "mag_code": "V",
        },
        {
            "name": "sample_Cepheid_like",
            "period": 8.20,
            "mag_max": 10.50,
            "mag_min": 11.20,
            "amplitude": 0.70,
            "epoch": 53000.0,
            "rise_time": 40.0,
            "mag_code": "V",
        },
        {
            "name": "sample_Eclipsing_like",
            "period": 1.80,
            "mag_max": 11.00,
            "mag_min": 13.20,
            "amplitude": 2.20,
            "epoch": 53000.0,
            "rise_time": 15.0,
            "mag_code": "V",
        },
        {
            "name": "sample_Mira_like",
            "period": 310.00,
            "mag_max": 8.50,
            "mag_min": 15.20,
            "amplitude": 6.70,
            "epoch": 53000.0,
            "rise_time": 45.0,
            "mag_code": "V",
        },
        {
            "name": "sample_SemiRegular_like",
            "period": 150.00,
            "mag_max": 9.50,
            "mag_min": 11.20,
            "amplitude": 1.70,
            "epoch": 53000.0,
            "rise_time": 40.0,
            "mag_code": "V",
        },
        {
            "name": "sample_DeltaScuti_like",
            "period": 0.12,
            "mag_max": 9.20,
            "mag_min": 9.45,
            "amplitude": 0.25,
            "epoch": 53000.0,
            "rise_time": 40.0,
            "mag_code": "V",
        },
    ])

    output_path = os.path.join(NEW_DATA_DIR, "sample_new_objects.csv")
    sample_data.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"가상 새 데이터 파일 생성 완료: {output_path}")


if __name__ == "__main__":
    main()