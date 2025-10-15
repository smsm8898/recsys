# 📊 MovieLens 100K 데이터셋 분석 요약

## 1. 사용자-아이템 상호작용 (`u.data`)
| 항목 | 값 | 참고 사항 |
|------|----|-----------|
| 사용자 수 | 943명 | - |
| 영화 수 | 1,682편 | - |
| 상호작용 수 | 100,000건 (평점) | 희소 행렬(Sparse Matrix) 형태 |
| 최소 평점 수 | 최소 20개 | 모든 사용자가 20개 이상의 영화를 시청함 |

---

## 2. 사용자 특징 (`u.user`)
| 항목 | 통계량 / 분포 | 주요 특징 및 직관적 해석 |
|------|---------------|------------------------|
| 연령 (Age) | Max: 73, Min: 7, Mean: 34, Median: 31 | 22~46세 사이가 가장 많아 경제력이 있는 젊은 층 시청이 많음을 시사. 분포는 거의 정규 분포 형태 |
| 성별 (Gender) | 남성: 670명 (71%), 여성: 273명 (29%) | 남성 시청자가 여성보다 압도적으로 많음 |
| 직업 (Occupation) | 총 21가지 | `student`가 가장 많음. `other`(105), `none`(9) 등 알 수 없는 직업 존재 → 전처리 시 묶음 처리 필요 |

<details>
<summary><strong>직업 목록 (총 21가지) 펼쳐보기</strong></summary>

- administrator, artist, doctor, educator, engineer, entertainment, executive, healthcare, homemaker, lawyer, librarian, marketing, none, other, programmer, retired, salesman, scientist, student, technician, writer

</details>

---

## 3. 아이템 특징 (`u.item`)
| 항목 | 통계량 / 분포 | 주요 특징 및 참고 사항 |
|------|---------------|------------------------|
| 결측치 | movie_id 267 | title, release_date 등 주요 값이 모두 NaN 처리됨 |
| IMDB URL | 모두 접속 불가 | 데이터 수집 이후 URL 변경 또는 삭제 |
| 비디오 출시일 | 모두 NaN | 분석에 활용 불가 |
| 개봉 연도 (Year) | 1922년 ~ 1998년 | 대부분 1980~2000년 사이 개봉, 평균에 몰려 있는 정규 분포 형태 |
| 개봉 월 (Month) | 1월 (68%) | 상반기 개봉작 집중 |
| 개봉 요일 (Day) | 금(40%) > 토(16%) > 일(15%) | 주말 개봉 압도적, 수요일(13%)도 의미 있음 |
| 장르 (Genre) | 총 19가지 | Drama, Comedy가 전체의 약 42.5%를 차지 |

### 장르 분포 상세 (상위 5개)
| 장르 | 비율 |
|------|-----|
| Drama | 25.06% |
| Comedy | 17.46% |
| Action | 8.68% |
| Thriller | 8.68% |
| Romance | 8.54% |

<details>
<summary><strong>전체 장르 분포 펼쳐보기</strong></summary>

| 장르 | 비율 (%) | 장르 | 비율 (%) |
|------|----------|------|----------|
| Drama | 25.06 | Film-Noir | 0.83 |
| Comedy | 17.46 | Fantasy | 0.76 |
| Action | 8.68 | unknown | 0.07 |
| Thriller | 8.68 | Animation | 1.45 |
| Romance | 8.54 | Documentary | 1.73 |
| Adventure | 4.67 | Musical | 1.94 |
| Children | 4.22 | Mystery | 2.11 |
| Crime | 3.77 | War | 2.45 |
| Sci-Fi | 3.49 | Horror | 3.18 |

</details>
