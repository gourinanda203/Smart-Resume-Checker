# Smart-Resume-Checker
Smart Resume Checker
   Smart Resume Checker is a machine learning and NLP–powered web app that predicts how well a resume matches a job description. The app helps job seekers and recruiters        quickly assess resume relevance.

🔹 Features

. Paste Resume and Job Description in the app.

. Cleans and preprocesses text using NLP techniques.

. Uses TF-IDF vectorization and Logistic Regression for prediction.

. Shows match probability and indicates if it’s a good match.

. Simple and interactive Streamlit interface.

🔹 Project Structure
smart-resume-checker/
│
├── app.py                    # Streamlit web app
├── train_model.py            # ML model training pipeline
├── utils.py                  # Text cleaning functions
├── job_applicant_dataset.csv # Dataset for training
├── model.pkl                 # Trained ML model
├── vectorizer.pkl            # TF-IDF vectorizer
└── smart resume checker ss.png  # App screenshot

🔹 How It Works

. Data Cleaning: Removes punctuation, stopwords, and normalizes text.

. Text Vectorization: Converts text to TF-IDF vectors (max_features=5000).

. Model Training: Logistic Regression model trained on resume + job description data.

Prediction:

Computes match probability (%)

Indicates GOOD or NOT GOOD match

🔹 Installation & Usage

1. Install Dependencies
pip install streamlit scikit-learn pandas joblib
2. Train the Model (Optional)
python train_model.py
3. Run the Web App
streamlit run app.py

🔹 App Screenshot
<img width="560" height="389" alt="smart resume checker web app ss" src="https://github.com/user-attachments/assets/cad83808-5243-49f9-87da-d691f70379fb" />



🔹 Tech Stack

. Python

. Pandas for data handling

. Scikit-learn for ML

. TF-IDF for NLP vectorization

. Streamlit for web interface

🔹 Use Cases

. Job seekers optimizing resumes for specific jobs

. Recruiters screening candidate resumes

. Resume ATS scoring systems
