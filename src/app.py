from flask import Flask, render_template, request, redirect, url_for, session
from multimodal_predict import (
    predict_mbti_multiclass,
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

app = Flask(__name__)
app.secret_key = "5004"

USERS = {
    "testuser": "testpass"
}

def save_mbti_bar_chart(prob_list, label_list, save_path):
    plt.figure(figsize=(10, 4))
    plt.bar(label_list, prob_list, color='skyblue')
    plt.ylim(0, 1)
    plt.xlabel('MBTI')
    plt.ylabel('Probability')
    plt.title('MBTI Softmax Probability')
    for i, v in enumerate(prob_list):
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def save_binary_bar_chart(axis_probs, save_path):
    axes = ['IE', 'NS', 'TF', 'JP']
    values = [axis_probs[a][3] for a in axes]
    plt.figure(figsize=(6, 2))
    plt.barh(axes, values, color='salmon')
    plt.xlim(0, 1)
    plt.xlabel('reliability')
    plt.title('MBTI Binary (축별 신뢰도)')
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
    binary_chart_url = None
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
            # 1. 전체 MBTI 예측
            mbti_pred, conf_text, prob_list, label_list = predict_mbti_multiclass(
                text,
                model_dir=os.path.abspath("result/bert_multiclass/final_model"),
                tokenizer_dir=os.path.abspath("result/bert_multiclass/final_tokenizer"),
                label_encoder_path=os.path.abspath("result/bert_multiclass/label_encoder.pkl"),
                return_label_list=True
            )
            static_dir = os.path.join(os.path.dirname(__file__), "static")
            mbti_chart_path = os.path.join(static_dir, "mbti_bar.png")
            save_mbti_bar_chart(prob_list, label_list, mbti_chart_path)
            mbti_chart_url = "/static/mbti_bar.png"

            # 2. 이진 분류 MBTI 예측
            mbti_binary, conf_binary, axis_probs = predict_mbti_binary_with_probs(text)
            binary_chart_path = os.path.join(static_dir, "binary_bar.png")
            save_binary_bar_chart(axis_probs, binary_chart_path)
            binary_chart_url = "/static/binary_bar.png"

            # 3. 이미지 설명 주요 키워드 추출
            img_desc_raw = generate_image_description(abs_image_path, "이 이미지의 분위기를 설명해줘")
            img_desc = clean_generated_description(img_desc_raw)
            keywords = [w for w in extract_keywords(img_desc, topk=5, stopwords=STOPWORDS) if len(w) > 1]

            # 4. 결과 summary
            summary = (
                f"<div class='result-mbti'>[전체 MBTI] 예측: <b>{mbti_pred}</b> (신뢰도 {conf_text:.2f})</div>"
                f"<div class='result-bar'><img src='{mbti_chart_url}' style='width:80%;max-width:700px'></div>"
                f"<div class='result-mbti'>[이진 분류 MBTI] 예측: <b>{mbti_binary}</b> (평균 신뢰도 {conf_binary:.2f})</div>"
                f"<div class='result-bar'><img src='{binary_chart_url}' style='width:60%;max-width:400px'></div>"
                f"<div class='result-keywords'>이미지 주요 키워드: <b>{', '.join(keywords)}</b></div>"
            )
            result = {
                "mbti_text": mbti_pred,
                "conf_text": conf_text,
                "mbti_binary": mbti_binary,
                "conf_binary": conf_binary,
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
