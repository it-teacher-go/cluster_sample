# main.py — 영화 유형 나누기: 정답 없이 비슷한 영화끼리 묶는다
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")
st.caption("정답 열 없이, 내가 고른 속성이 비슷한 영화끼리 묶습니다.")

MOVIES = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"

기호 = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]

속성 = [
    "스크린 수(로그)",
    "누적 관객(로그)",
    "10위권 일수",
    "롱런 지수"
]

원래단위 = [
    "스크린 수",
    "누적 관객",
    "10위권 일수",
    "롱런 지수"
]


@st.cache_data
def load_data():
    df = pd.read_csv(
        MOVIES,
        dtype={"movieCd": str}
    )

    df = df.sort_values("movieCd").reset_index(drop=True)

    전체편수 = len(df)

    # 네 속성을 만들 수 없는 영화는 제외
    쓸열 = [
        "first_scrn",
        "total_audi",
        "days_in_top10",
        "first_week_audi"
    ]

    df = df.dropna(
        subset=쓸열 + ["movieNm"]
    )

    df = df[
        (df["first_scrn"] > 0)
        & (df["total_audi"] > 0)
        & (df["first_week_audi"] > 0)
    ]

    df = df.reset_index(drop=True)

    return df, 전체편수


df, 전체편수 = load_data()


# -------------------------------------------------
# 1. 군집화에 사용할 네 가지 속성 만들기
# -------------------------------------------------

df["스크린 수"] = df["first_scrn"]
df["누적 관객"] = df["total_audi"]
df["10위권 일수"] = df["days_in_top10"]

df["롱런 지수"] = (
    df["total_audi"]
    / df["first_week_audi"]
).clip(upper=20)

# 값 차이가 큰 두 속성은 로그 변환
df["스크린 수(로그)"] = np.log10(
    df["first_scrn"]
)

df["누적 관객(로그)"] = np.log10(
    df["total_audi"]
)


st.info(
    f"전체 {전체편수:,}편 가운데 "
    f"네 속성을 모두 만들 수 있는 "
    f"{len(df):,}편으로 묶었습니다."
)


# -------------------------------------------------
# 2. 사용할 속성 선택
# -------------------------------------------------

고른속성 = st.multiselect(
    "묶는 데 사용할 속성 (둘 이상)",
    속성,
    default=속성
)

if len(고른속성) < 2:
    st.warning("속성을 둘 이상 골라 주세요.")
    st.stop()


# -------------------------------------------------
# 3. 표준화
# -------------------------------------------------

scaler = StandardScaler()

Xs = scaler.fit_transform(
    df[고른속성]
)


# 표준화된 값을 DataFrame으로 다시 만든다
표준화df = pd.DataFrame(
    Xs,
    columns=고른속성,
    index=df.index
)


# -------------------------------------------------
# 4. K-means 군집화
# -------------------------------------------------

k = 3

번호 = KMeans(
    n_clusters=k,
    n_init=10,
    random_state=42
).fit_predict(Xs)


# -------------------------------------------------
# 5. 묶음 이름 정하기
# 누적 관객 평균이 큰 묶음부터 ㉮, ㉯, ㉰
# -------------------------------------------------

순서 = (
    pd.Series(
        df["누적 관객"].to_numpy()
    )
    .groupby(번호)
    .mean()
    .sort_values(ascending=False)
    .index
)

자리 = {
    g: i
    for i, g in enumerate(순서)
}

df["묶음"] = [
    기호[자리[g]]
    for g in 번호
]


# 표준화된 그래프에서도 영화 이름과 묶음을 사용할 수 있도록 추가
표준화df["묶음"] = df["묶음"]
표준화df["movieNm"] = df["movieNm"]


# -------------------------------------------------
# 6. 표준화 설명
# -------------------------------------------------

st.info(
    "아래 묶음 지도는 표준화된 값으로 그립니다. "
    "각 속성에서 0은 평균, 양수는 평균보다 큰 값, "
    "음수는 평균보다 작은 값입니다."
)


# -------------------------------------------------
# 7. 2차원 묶음 지도
# -------------------------------------------------

st.subheader("묶음 지도 · 2차원")

c1, c2 = st.columns(2)

x축 = c1.selectbox(
    "가로축",
    고른속성,
    index=0
)

y축 = c2.selectbox(
    "세로축",
    고른속성,
    index=min(
        1,
        len(고른속성) - 1
    )
)


fig = px.scatter(
    표준화df,
    x=x축,
    y=y축,
    color="묶음",
    hover_name="movieNm",
    category_orders={
        "묶음": 기호[:k]
    }
)

fig.update_traces(
    marker=dict(
        size=9,
        opacity=0.8
    )
)

fig.update_layout(
    height=460
)


# 평균 위치 0을 기준선으로 표시
fig.add_vline(
    x=0,
    line_dash="dash"
)

fig.add_hline(
    y=0,
    line_dash="dash"
)


st.plotly_chart(
    fig,
    width="stretch"
)

st.caption(
    "표준화된 값으로 그린 그래프입니다. "
    "0은 해당 속성의 평균입니다. "
    "+1은 평균보다 약 1표준편차 큰 위치, "
    "-1은 평균보다 약 1표준편차 작은 위치입니다."
)


# -------------------------------------------------
# 8. 3차원 묶음 지도
# -------------------------------------------------

st.subheader("묶음 지도 · 3차원")

if len(고른속성) < 3:

    st.info(
        "속성을 셋 이상 고르면 "
        "3차원 그림이 표시됩니다."
    )

else:

    d1, d2, d3 = st.columns(3)

    x3 = d1.selectbox(
        "x축",
        고른속성,
        index=0,
        key="x3"
    )

    y3 = d2.selectbox(
        "y축",
        고른속성,
        index=1,
        key="y3"
    )

    z3 = d3.selectbox(
        "z축",
        고른속성,
        index=2,
        key="z3"
    )


    fig3 = px.scatter_3d(
        표준화df,
        x=x3,
        y=y3,
        z=z3,
        color="묶음",
        hover_name="movieNm",
        category_orders={
            "묶음": 기호[:k]
        }
    )


    fig3.update_traces(
        marker=dict(
            size=2,
            opacity=0.7
        )
    )

    fig3.update_layout(
        height=560,
        legend=dict(
            orientation="h"
        )
    )


    st.plotly_chart(
        fig3,
        width="stretch"
    )

    st.caption(
        "세 축 모두 표준화된 값입니다. "
        "각 축에서 0이 평균입니다. "
        "마우스로 끌면 그래프를 회전할 수 있습니다."
    )


# -------------------------------------------------
# 9. 묶음별 특성
# -------------------------------------------------

st.subheader(
    "묶음별 편수와 네 속성의 평균"
)

요약 = (
    df
    .groupby("묶음")[원래단위]
    .mean()
    .round(2)
)

요약.insert(
    0,
    "편수",
    df.groupby("묶음").size()
)

요약 = 요약.reindex(
    기호[:k]
)


st.dataframe(
    요약,
    width="stretch",
    column_config={

        "편수":
            st.column_config.NumberColumn(
                "편수",
                format="%,d편"
            ),

        "스크린 수":
            st.column_config.NumberColumn(
                "스크린 수",
                format="%.1f개"
            ),

        "누적 관객":
            st.column_config.NumberColumn(
                "누적 관객",
                format="%,.0f명"
            ),

        "10위권 일수":
            st.column_config.NumberColumn(
                "10위권 일수",
                format="%.1f일"
            ),

        "롱런 지수":
            st.column_config.NumberColumn(
                "롱런 지수",
                format="%.2f배"
            ),
    }
)

st.caption(
    "이 표의 평균은 로그 변환이나 표준화를 하기 전의 "
    "원래 단위로 표시합니다. "
    "롱런 지수는 누적 관객을 첫 주 관객으로 나눈 값이며 "
    "20을 넘는 값은 20으로 제한했습니다."
)


# -------------------------------------------------
# 10. 묶음마다 관객이 많은 영화
# -------------------------------------------------

st.subheader(
    "묶음마다 관객이 많은 다섯 편"
)

열 = st.columns(k)

for i, 이름 in enumerate(
    기호[:k]
):

    해당묶음전체 = df[
        df["묶음"] == 이름
    ]

    묶음 = (
        해당묶음전체
        .sort_values(
            "누적 관객",
            ascending=False
        )
        .head(5)
    )

    열[i].markdown(
        f"**{이름} 묶음 "
        f"({len(해당묶음전체):,}편)**"
    )

    열[i].dataframe(
        pd.DataFrame(
            {
                "영화":
                    묶음[
                        "movieNm"
                    ].to_numpy(),

                "누적 관객":
                    묶음[
                        "누적 관객"
                    ].to_numpy()
            }
        ),

        width="stretch",
        hide_index=True,

        column_config={

            "누적 관객":
                st.column_config.NumberColumn(
                    "누적 관객",
                    format="%,d명"
                )
        }
    )
