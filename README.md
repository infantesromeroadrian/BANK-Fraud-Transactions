# 🕵️ Fraud Detection in Bank Transactions

## Overview 👀
This project implements a machine learning model 🧠 to identify potentially fraudulent transactions 💸 in banking systems 🏦. It leverages Apache Spark 🚀 to process large volumes of data and utilizes PySpark for feature engineering and MLflow for model tracking and serving. The predictive model is then exposed through a Streamlit application that allows users to input transaction details and receive instant predictions on whether a transaction is likely to be fraudulent.

## Features 🌟
- Data Ingestion and Processing with PySpark ✨.
- Feature Engineering and Scaling 🔍.
- Model Training and Evaluation using Spark MLlib 📊.
- Model Tracking and Management with MLflow 📈.
- A Streamlit Web Application for real-time fraud prediction 🖥️.

## Project Structure 📁
The project is organized as follows:
- `data_ingestion.py`: Script for loading and preprocessing the data 📚.
- `feature_engineering.py`: Script for transforming the data into a suitable format for modeling 🔧.
- `model_training.py`: Script for training the fraud detection model 🏋️‍♂️.
- `model_evaluation.py`: Script for evaluating the model's performance 📝.
- `streamlit_app.py`: Streamlit application script for the interactive fraud prediction interface 🎛️.
- `/model`: Directory containing the saved ML models 📦.
- `requirements.txt`: A text file listing the project's dependencies 📄.

## Installation 🛠️
To set up the project environment:
1. Clone the repository to your local machine 🖥️.
2. Ensure that Python 3.6+ is installed 🐍.
3. Install the required Python packages using `poetry install` 📌.

## Usage 🚀
To run the Streamlit app locally:
1. Navigate to the project directory 🗂️.
2. Execute `streamlit run streamlit_app.py` 🖥️.
3. Open your web browser to the address provided in the command line output (typically `http://localhost:8501`) 🌐.

## Model Training 🏋️
To train the model, run the `model_training.py` script. This will process the data, train the model, and save it to the `/model` directory.

## Contributing 🤝
Contributions to this project are welcome. Please follow the standard fork-branch-PR workflow 🔄.

## License 📜
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact 📬
- Your Name - your.email@example.com 📧
- Project Link: [https://github.com/infantesromeroadrian/BankFraudTransactions.git) 🔗

## Acknowledgments 💖
- Thanks to the team for insights and contributions 🙏.
- Special thanks to open-source projects that made this project possible 🎉.
