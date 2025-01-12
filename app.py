import streamlit as st
import pickle
import numpy as np

# 学習済みモデルをロード
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlitアプリのタイトル
st.title("住宅価格予測アプリ")
st.write("区、部屋の広さ、築年数、駅徒歩分数、バス分数を入力してください。")

# 地域マッピング情報
area_mapping = {
    "千代田区": "都心部", "中央区": "都心部", "港区": "都心部", "新宿区": "都心部", "文京区": "都心部", "渋谷区": "都心部",
    "台東区": "東部", "墨田区": "東部", "荒川区": "東部", "足立区": "東部", "江東区": "東部", "葛飾区": "東部", "江戸川区": "東部",
    "品川区": "南部", "目黒区": "南部", "世田谷区": "南部", "大田区": "南部",
    "中野区": "西部", "杉並区": "西部", "練馬区": "西部",
    "豊島区": "北部", "北区": "北部", "板橋区": "北部"
}

# 地域情報を整形して表示
region_info = {
    "北部": ["豊島区", "北区", "板橋区"],
    "南部": ["品川区", "目黒区", "世田谷区", "大田区"],
    "東部": ["台東区", "墨田区", "荒川区", "足立区", "江東区", "葛飾区", "江戸川区"],
    "西部": ["中野区", "杉並区", "練馬区"],
    "都心部": ["千代田区", "中央区", "港区", "新宿区", "文京区", "渋谷区"]
}

# サイドバーに地域情報を表示
st.sidebar.header("地域情報")
for region, wards in region_info.items():
    st.sidebar.write(f"**{region}**: {', '.join(wards)}")

# サイドバーで入力フォームを作成
st.sidebar.header("物件情報の入力")

# 入力項目
区 = st.sidebar.selectbox("区", ["北部", "南部", "東部", "西部", "都心部"])
部屋の広さ = st.sidebar.number_input("部屋の広さ (㎡)", min_value=10.0, max_value=300.0, value=70.0, step=1.0)
築年数 = st.sidebar.number_input("築年数 (年)", min_value=0, max_value=100, value=20, step=1)
駅徒歩分数 = st.sidebar.number_input("駅徒歩分数 (分)", min_value=0, max_value=60, value=10, step=1)
バス分数 = st.sidebar.number_input("バス分数 (分)", min_value=0, max_value=60, value=0, step=1)

# 区をOne-Hot Encoding
区_dict = {
    "北部": [1, 0, 0, 0, 0],
    "南部": [0, 1, 0, 0, 0],
    "東部": [0, 0, 1, 0, 0],
    "西部": [0, 0, 0, 1, 0],
    "都心部": [0, 0, 0, 0, 1],
}
区_encoded = 区_dict[区]  # 選択された区をOne-Hot Encoding形式に変換

# 入力データを9個の特徴量に変換
# 順序: [専有面積, 築年数, 駅徒歩分数, バス分数] + [エリア_北部, エリア_南部, エリア_東部, エリア_西部, エリア_都心部]
input_data = np.array(
    [部屋の広さ, 築年数, 駅徒歩分数, バス分数] + 区_encoded
).reshape(1, -1)

# 価格予測ボタン
if st.button("価格を予測する"):
    try:
        # モデルで予測
        predicted_price = model.predict(input_data)[0]

        # 結果を表示
        st.subheader("予測結果")
        st.write(f"この物件の予測価格は **{predicted_price:,.0f}万円** です。")
    except Exception as e:
        # エラーが発生した場合の処理
        st.error(f"予測中にエラーが発生しました: {e}")
