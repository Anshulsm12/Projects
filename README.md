# Stock Market Price Prediction

## Setup Instructions

1. **Clone the repository**:
   ```sh
   git clone https://github.com/Anshulsm12/Stock-Market_Price-Prediction.git
   cd Stock-Market_Price-Prediction
   ```

2. **Create and activate a virtual environment**:
   ```sh
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**:
   ```sh
   streamlit run prediction1.py
   ```

## Project Description

This project predicts stock market prices using machine learning models. The main script `prediction1.py` fetches stock data, preprocesses it, trains a model, and makes predictions which are visualized using Streamlit.