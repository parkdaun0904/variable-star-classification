# GCVS 기반 변광성 머신러닝 분류 프로젝트

================================================================================================
1. 프로젝트 목적

---

이 프로젝트는 GCVS와 ASAS-SN 변광성 카탈로그 데이터를 이용해
대표적인 변광성 타입을 머신러닝으로 분류하는 코드이다.

연구 전체 주제는 불규칙하거나 제한된 광도곡선 데이터에서
전통적인 주기 분석 방법과 머신러닝 기반 분류 방법을 비교하는 것이다.

이 코드 묶음은 그중 머신러닝 분류 부분을 담당한다.

모델은 다음과 같은 feature를 이용해 학습한다.

- period
- mag_max
- mag_min
- mean_mag
- amplitude
- epoch
- rise_time
- mag_code

학습에는 Random Forest 모델을 사용하며,
새로운 관측 대상에 대해 각 변광성 타입일 확률을 출력한다.

또한 conclusion_lightcurves 폴더의 txt 광도곡선 파일을 읽어
자동으로 feature를 추출하고 예측할 수 있다.

================================================================================================
2. 전체 파일 구조

코드 모음/
├─ project/
│  ├─ train_model.py
│  ├─ predict_new.py
│  ├─ gcvs_parser.py
│  ├─ asassn_parser.py
│  ├─ lightcurve_features.py
│  ├─ ml_pipeline.py
│  ├─ label_mapping.py
│  ├─ config.py
│  ├─ gcvs5.txt
│  └─ asassn_catalog_full.csv
│
├─ conclusion_lightcurves/
│  ├─ TARGET_INFO.TXT
│  └─ *.txt
│
├─ models/
├─ results/
└─ new_data/

================================================================================================

3. 사용 데이터

[1] GCVS

General Catalogue of Variable Stars.

전통적인 변광성 카탈로그이며
주기, 등급, 변광성 타입 등의 정보를 포함한다.

[2] ASAS-SN

All-Sky Automated Survey for Supernovae.

대량의 변광성 데이터를 포함하며
class_probability 값이 낮은 데이터는 제외한다.

현재 설정:

ASASSN_MIN_CLASS_PROBABILITY = 0.5

즉, 모델이 어느 정도 확신한 데이터만 학습에 사용한다.

================================================================================================

4. 현재 분류 대상

현재 코드에서는 아래 6개 그룹만 사용한다.

- RR_Lyrae
- Cepheid
- Eclipsing
- Mira
- SemiRegular
- DeltaScuti

GCVS에는 훨씬 더 많은 세부 타입이 존재하지만,
이번 연구에서는 대표적인 타입 위주로 단순화하였다.

================================================================================================

5. 학습 데이터 필터링 조건

다음 숫자 feature 중 최소 2개 이상 값이 있어야 학습에 사용한다.

- period
- mag_max
- mag_min
- amplitude
- epoch
- rise_time
또한 한 라벨의 데이터 수가 30개 미만이면 제외한다.

이유:
- 데이터가 너무 적으면 모델이 특징을 안정적으로 배우기 어려움
- 테스트 결과가 크게 흔들릴 수 있음

================================================================================================

6. 실행 방법

[1] 모델 학습

python train_model.py

생성 결과 예시:

models/gcvs_asassn_random_forest.joblib

results/날짜시간/
├─ parsed_gcvs_asassn_all.csv
├─ training_dataset_grouped.csv
├─ classification_report.txt
├─ confusion_matrix.csv
├─ confusion_matrix.png
└─ test_prediction_top3.csv

[2] CSV 파일 예측

python predict_new.py --input ../new_data/sample_new_objects.csv

[3] txt 광도곡선 폴더 예측

python predict_new.py --folder ../conclusion_lightcurves

================================================================================================

7. conclusion_lightcurves 사용 방법

conclusion_lightcurves 안에는 다음 파일들을 넣는다.

- TARGET_INFO.TXT
- 광도곡선 txt 파일들

예시:

object_name=V799 Aur
expected_type=
period=
brightness_mode=differential_mag
filters=all
time_column=1
mag_column=2
error_column=3

brightness_mode 설명:

differential_mag

차등등급 또는 상대 밝기값.

이 경우:

- mag_max
- mag_min
- mean_mag

값은 사용하지 않고 amplitude 중심으로 예측한다.

================================================================================================

8. 자동 주기 추정

txt 광도곡선 예측 시
period 값이 비어 있으면 Lomb-Scargle 기반으로 자동 추정한다.

생성 파일:

period_candidates.csv

예시:

rank,period,power
1,0.07597,0.82
2,0.15190,0.61

현재 관측 데이터에서 가장 가능성이 높은 주기 후보를 의미한다.

================================================================================================

9. 위상(Phase) 정보

광도곡선 txt 예측 시
추정된 period와 epoch를 이용해 위상 범위도 계산한다.

예시:

phase_text = 0.132 ~ 0.287

의미:

전체 주기를 0~1로 정규화했을 때
현재 관측 데이터는 약 13.2% ~ 28.7% 구간을 관측했다는 뜻이다.

즉:
- phase_span이 작으면 전체 주기의 일부만 관측한 상태
- phase_span이 크면 더 넓은 주기 구간을 관측한 상태

================================================================================================

10. 예측 결과 해석

출력 예:

top1_label
top1_prob
top2_label
top2_prob
top3_label
top3_prob

예:

top1_label = DeltaScuti
top1_percent = 86.48%

의미:

모델이 해당 데이터를 Delta Scuti형일 가능성이 가장 높다고 판단했다는 뜻이다.

================================================================================================

11. 같은 대상에 대한 결과 통합

같은 대상(object_name)에 대한 여러 txt 파일이 존재할 경우
각 파일별 결과뿐 아니라 전체 경향도 비교 가능하다.

예:
- B/V/R/I 필터별 결과 비교
- 날짜별 예측 안정성 비교
- 특정 필터에서의 혼동 경향 확인

현재 V799 Aur 데이터에서는 대부분 DeltaScuti로 안정적으로 분류되었으며,
일부 구간에서는 Eclipsing 또는 RR_Lyrae와 부분적으로 혼동되는 결과가 나타났다.

이는:
- 관측 구간 부족
- 짧은 주기 변광성 간 feature overlap
- differential magnitude 사용
등의 이유 때문일 수 있다.

================================================================================================

12. 현재 코드의 한계

이 코드는 광도곡선 이미지를 직접 넣어 CNN처럼 분류하는 구조가 아니다.

현재 구조:

광도곡선
    > feature 추출
    > 머신러닝 모델 입력
    > 변광성 타입 확률 출력

즉, feature 기반 머신러닝 모델이다.

================================================================================================

13. 참고

tempCodeRunnerFile.py는 VS Code 실행 중 생성되는 임시 파일일 가능성이 크다.

프로젝트 핵심 파일은 아니므로 제출용에서는 제외 가능하다.
