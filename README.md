# Classification app

## 🌟 About the Project
A Classification App is a software application designed to categorize input data into predefined classes or 
labels using machine learning models (TabNet TabTransformer FT-Transformer XGBClassifier RandomForestClassifier).

## 📦 Installation
To run the project locally, follow these steps:

1) Clone the repo :
```bash
git clone (project git url)
cd (project folder)
```

2) Create a Virtual Environment :
```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

3) Install dependencies from requirements.txt :
```bash
pip install -r requirements.txt
```

4) Run backend service:
```bash
cd back
uvicorn main:app --port 8000 --reload
```

5) Run frontend service:
```bash
cd front
python main.py
```

