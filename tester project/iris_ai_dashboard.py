# ==============================================================
# 🌸 IRIS AI DASHBOARD — GRADIO 2026 STYLE
# Modern UI • Dark Mode • Live Accuracy • Animated Cards
# ==============================================================

import numpy as np
import pandas as pd
import gradio as gr
import plotly.express as px

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# ==============================================================
# LOAD DATASET
# ==============================================================

iris = datasets.load_iris()

X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================================================
# MODELS
# ==============================================================

models = {
    "🌳 Decision Tree": DecisionTreeClassifier(
        criterion="entropy", random_state=42
    ),
    "📍 K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=5
    ),
    "📊 Naive Bayes": GaussianNB()
}

trained_models = {}
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results[name] = round(acc * 100, 2)

# ==============================================================
# CHART
# ==============================================================

def generate_chart():
    df = pd.DataFrame({
        "Model": list(results.keys()),
        "Accuracy": list(results.values())
    })

    fig = px.bar(
        df,
        x="Model",
        y="Accuracy",
        text="Accuracy",
        color="Accuracy",
        template="plotly_dark",
        height=450
    )

    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(
        title="🚀 Model Performance Comparison",
        xaxis_title="Algorithm",
        yaxis_title="Accuracy %",
        yaxis_range=[0, 110]
    )

    return fig

# ==============================================================
# PREDICTION ENGINE
# ==============================================================

def predict_flower(model_name, sl, sw, pl, pw):
    model = trained_models[model_name]

    sample = np.array([[sl, sw, pl, pw]])
    pred = model.predict(sample)[0]

    species = target_names[pred]
    accuracy = results[model_name]

    emoji = {
        "setosa": "🌼",
        "versicolor": "🌺",
        "virginica": "🌷"
    }

    return (
        f"{emoji[species]} Predicted Species: {species.upper()}",
        f"🎯 Historical Accuracy: {accuracy} %"
    )

# ==============================================================
# CUSTOM CSS
# ==============================================================

custom_css = """
body {
    background: #0f172a;
}

.gradio-container {
    max-width: 1300px !important;
    margin: auto;
}

.card {
    border-radius: 20px;
    padding: 20px;
    background: linear-gradient(145deg,#111827,#1e293b);
    box-shadow: 0 8px 25px rgba(0,0,0,.35);
}

h1,h2,h3 {
    text-align:center;
}
"""

# ==============================================================
# UI
# ==============================================================

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="violet",
        neutral_hue="slate"
    ),
    css=custom_css,
    title="Iris AI Dashboard"
) as app:

    gr.Markdown("""
    # 🌸 IRIS AI DASHBOARD 2026
    ### Next Generation Machine Learning Tester
    """)

    with gr.Row():

        with gr.Column(scale=1):

            gr.Markdown("## ⚙️ Input Panel")

            model_choice = gr.Dropdown(
                choices=list(models.keys()),
                value="🌳 Decision Tree",
                label="Choose Model"
            )

            sl = gr.Slider(4, 8, value=5.1, label="Sepal Length")
            sw = gr.Slider(2, 5, value=3.5, label="Sepal Width")
            pl = gr.Slider(1, 7, value=1.4, label="Petal Length")
            pw = gr.Slider(0.1, 3, value=0.2, label="Petal Width")

            predict_btn = gr.Button(
                "🚀 Analyze Flower",
                variant="primary",
                size="lg"
            )

        with gr.Column(scale=1):

            gr.Markdown("## 📋 Prediction Result")

            output1 = gr.Textbox(
                label="Species Result",
                lines=2
            )

            output2 = gr.Textbox(
                label="Model Accuracy",
                lines=2
            )

            gr.Markdown("""
            ### 🧠 AI Notes

            - Decision Tree unggul dalam aturan logika.
            - KNN unggul pada data jarak dekat.
            - Naive Bayes cepat & efisien.
            """)

    gr.Markdown("---")

    gr.Markdown("## 📊 Live Analytics Dashboard")

    chart = gr.Plot(value=generate_chart())

    predict_btn.click(
        fn=predict_flower,
        inputs=[model_choice, sl, sw, pl, pw],
        outputs=[output1, output2]
    )

# ==============================================================
# RUN
# ==============================================================

app.launch(
    share=True,
    debug=True
)