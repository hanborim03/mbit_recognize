from flask import Flask, render_template, request, redirect, url_for, session
from multimodal_predict import (
    predict_mbti_binary_with_probs,
    generate_image_description,
    clean_generated_description,
    extract_keywords,
    STOPWORDS,
)
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

app = Flask(__name__)
app.secret_key = "5004"

USERS = {
    "testuser": "testpass"
}

def save_mbti_bar_chart(axis_probs, save_path):
    axes = ['IE', 'NS', 'TF', 'JP']
    values = [axis_probs[a][3] for a in axes]  # 신뢰도(예측된 축의 softmax 값)
    plt.figure(figsize=(10, 2))
    plt.barh(axes, values, color='skyblue')
    plt.xlim(0, 1)
    plt.xlabel('reliability')
    plt.title('MBTI')
    for i, v in enumerate(values):
        plt.text(v + 0.02, i, f'{v:.2f}', va='center')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_path = None
    summary = None
    mbti_chart_url = None
    username = session.get("username")
    if request.method == "POST":
        if not username:
            return redirect(url_for("login"))
        text = request.form.get("text")
        image = request.files.get("image")
        if image:
            static_dir = os.path.join(os.path.dirname(__file__), "static")
            if not os.path.exists(static_dir):
                os.makedirs(static_dir)
            abs_image_path = os.path.join(static_dir, "uploaded_image.jpg")
            image.save(abs_image_path)
            image_path = "static/uploaded_image.jpg"
        if text and image_path:
            # 1. 텍스트 기반 MBTI 예측
            mbti_binary, conf_text, axis_probs = predict_mbti_binary_with_probs(text)
            # 2. 신뢰도 그래프 생성
            mbti_chart_path = os.path.join(static_dir, "mbti_bar.png")
            save_mbti_bar_chart(axis_probs, mbti_chart_path)
            mbti_chart_url = "/static/mbti_bar.png"
            # 3. 이미지 설명 주요 키워드만 추출
            img_desc_raw = generate_image_description(abs_image_path, "이 이미지의 분위기를 설명해줘")
            img_desc = clean_generated_description(img_desc_raw)
            keywords = [w for w in extract_keywords(img_desc, topk=5, stopwords=STOPWORDS) if len(w) > 1]
            summary = (
                f"<div class='result-mbti'>텍스트에서 분석한 MBTI: <b>{mbti_binary}</b> "
                f"(평균 신뢰도 {conf_text:.2f})</div>"
                f"<div class='result-bar'><img src='{mbti_chart_url}' style='width:80%;max-width:700px'></div>"
                f"<div class='result-keywords'>이미지 주요 키워드: <b>{', '.join(keywords)}</b></div>"
            )
            result = {
                "mbti_text": mbti_binary,
                "conf_text": conf_text,
                "img_keywords": keywords
            }
    return render_template("index.html", summary=summary, result=result, image_path=image_path, username=username)

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username in USERS and USERS[username] == password:
            session["username"] = username
            return redirect(url_for("index"))
        else:
            error = "아이디 또는 비밀번호가 올바르지 않습니다."
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
