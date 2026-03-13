# 바벨 전략 백테스팅 (Barbell Strategy Backtesting)

이 프로젝트는 바벨 전략의 백테스팅 결과를 분석합니다. 데이터 전처리 과정과 백테스팅 결과를 포함합니다.

## 백테스팅 결과

### 전략 성과
![백테스팅 성과](backtest_performance.png)

### 레짐 타임라인
![레짐 타임라인](regime_timeline.png)

### VIX 신호 비교
![VIX 신호 비교](signal_comparison_vix.png)

## 데이터 전처리 과정

- 데이터 소스: FRED API 등을 사용한 경제 지표 및 자산 가격 데이터
- 전처리: 결측치 처리, 정규화, 레짐 분류 등
- 자세한 코드는 `바벨_VIX_추가.ipynb` 노트북을 참조하세요.

## 파일 설명

- `바벨_VIX_추가.ipynb`: 메인 백테스팅 노트북
- 기타 노트북 파일들: 이전 버전 및 분석
- 엑셀 파일들: 상세 성과 보고서

## 실행 방법

1. Python 환경 설정
2. 필요한 패키지 설치 (pandas, matplotlib, etc.)
3. 노트북 실행