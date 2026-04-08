# 🌾 Yield Prediction Web App

A Machine Learning-powered web application that predicts agricultural yield based on input parameters. This project integrates a trained ML model with a Flask backend and a simple frontend interface.

---

## 🚀 Live Demo

👉 https: https://yield-prediction-vwqh.onrender.com

---

## 📌 Features

* 🌱 Predict crop yield using a trained ML model
* ⚡ Fast and interactive web interface
* 🧠 Model trained using Scikit-learn
* 🌐 Deployed using Render
* 📊 Clean UI with HTML, CSS, and Flask

---

## 🛠️ Tech Stack

* **Frontend:** HTML, CSS
* **Backend:** Flask (Python)
* **Machine Learning:** Scikit-learn
* **Deployment:** Render
* **Version Control:** Git & GitHub

---

## 📂 Project Structure

```
Yield_Prediction/
│
├── static/
│   └── css/
│       └── disease_style.css
│
├── templates/
│   └── index.html
│
├── app.py
├── yield_prediction_model.pkl
├── requirements.txt
├── Procfile
└── .gitignore
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/yield-prediction.git
cd yield-prediction
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the App

```bash
python app.py
```

👉 Open in browser:

```
http://127.0.0.1:5000/
```

---

## 🚀 Deployment (Render)

1. Push code to GitHub
2. Go to Render → New Web Service
3. Connect GitHub repo
4. Add:

   * **Build Command:**

     ```
     pip install -r requirements.txt
     ```
   * **Start Command:**

     ```
     gunicorn app:app
     ```
5. Deploy 🚀

---

## 🧠 Model Information

* Model trained using **Scikit-learn**
* Saved as `.pkl` file using **pickle**
* Loaded in Flask backend for predictions

---

## 📸 Screenshots

*Add your app screenshots here*

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork this repo and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Raj Sirohi**

* 💼 Aspiring AI & Data Engineer
* 🔗 LinkedIn: https://linkedin.com/in/your-profile

---

## ⭐ Show Your Support

If you like this project, give it a ⭐ on GitHub!
