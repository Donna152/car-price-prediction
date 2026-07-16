# 🚗 Car Price Predictor

A machine learning web app that predicts the resale price of a used car based on its name, company, manufacturing year, kilometers driven, and fuel type. Built with scikit-learn and deployed as an interactive Streamlit app.

**🔗 Live Demo:** [donna7-car-price-predictor.streamlit.app](https://donna7-car-price-predictor.streamlit.app)

---

## 📌 Overview

This project builds a regression pipeline that estimates a used car's price from real-world classified-ad data. Messy, inconsistently formatted scraped data is cleaned and structured, categorical features are encoded, and a Linear Regression model is trained to predict price. The entire pipeline (encoding + model) is bundled into a single deployable object.

## ✨ Features

- 🚘 Select the car name, company, manufacturing year, kilometers driven, and fuel type
- 💰 Get an instant estimated resale price
- 🧹 Model trained on a cleaned, outlier-filtered version of real Quikr car listing data
- 🔗 End-to-end scikit-learn `Pipeline` (encoding + model) for consistent predictions

## 🖼️ Screenshots

Screenshots of the app in action are available in the [`Results`](./Results) folder.

## 🧠 How It Works

### 1. Dataset

The project uses **`quikr_car.csv`** (in the [`dataset`](./dataset) folder) — a raw dataset of used car listings scraped from Quikr, containing:

| Column | Description |
|---|---|
| `name` | Full car name/model |
| `company` | Car manufacturer |
| `year` | Manufacturing year |
| `Price` | Listed price (target variable) |
| `kms_driven` | Kilometers driven |
| `fuel_type` | Fuel type (Petrol, Diesel, LPG, etc.) |

### 2. Data Cleaning

The raw dataset was messy and required significant cleaning:

- **`year`** — contained non-year junk values; filtered to numeric-only entries and cast to `int`
- **`Price`** — contained `"Ask For Price"` placeholder rows; these were dropped, commas were stripped, and the column was cast to `int`
- **`kms_driven`** — contained values like `"45,000 kms"`; the `"kms"` suffix and commas were stripped, non-numeric rows dropped, and the column cast to `int`
- **`fuel_type`** — rows with missing values were removed
- **`name`** — trimmed to the first 3 words (e.g. `"Maruti Suzuki Swift Dzire VXi"` → `"Maruti Suzuki Swift"`) to reduce cardinality
- The index was reset after row removal, and the cleaned dataset was exported to `cleaned_car.csv`

### 3. Outlier Removal

Boxplots of `year`, `Price`, and `kms_driven` revealed extreme outliers (e.g. a `Price` value in the millions, far beyond the rest of the distribution). Rows with `Price >= 6,000,000` were removed to keep the model from being skewed by unrealistic listings.

### 4. Feature Encoding

- **Categorical features** (`name`, `company`, `fuel_type`) are transformed using **One-Hot Encoding** (`OneHotEncoder`)
- **Numerical features** (`year`, `kms_driven`) are passed through unchanged
- A `ColumnTransformer` (`make_column_transformer`) applies this encoding consistently to any new input data

### 5. Model Building

- **Algorithm:** Linear Regression
- The column transformer and the regression model are combined into a single **`Pipeline`** (`make_pipeline`), so encoding and prediction happen in one step
- The data was split 80/20 into train and test sets
- Model performance was evaluated using **R² score** on the test set

### 6. Deployment Preparation

The final trained pipeline is serialized with `pickle`:

| File | Contents |
|---|---|
| `LinearRegression.pkl` | Full trained pipeline (One-Hot Encoding + Linear Regression model) |
| `car.pkl` | Cleaned car dataframe, used to populate dropdown options (car names, companies, years, fuel types) in the app |

The Streamlit app loads both files at startup — `car.pkl` to populate the input dropdowns, and `LinearRegression.pkl` to make predictions directly from raw input (no manual encoding needed, since it's baked into the pipeline).

## 🛠️ Tech Stack

- **Python**
- **pandas / numpy** — data cleaning and processing
- **scikit-learn** — `OneHotEncoder`, `ColumnTransformer`, `LinearRegression`, `Pipeline`
- **matplotlib / seaborn** — outlier visualization (boxplots)
- **Streamlit** — web app frontend
- **pickle** — model/data serialization

## 📁 Project Structure

```
car-price-predictor/
├── app.py                       # Streamlit frontend
├── car_price_predictor.ipynb    # Data cleaning, EDA & model-building notebook
├── LinearRegression.pkl         # Serialized trained pipeline (encoder + model)
├── car.pkl                      # Serialized cleaned car dataframe (for dropdown options)
├── dataset/                     # Raw dataset (quikr_car.csv)
├── Results/                     # Screenshots of app results
├── requirements.txt             # Python dependencies
├── setup.sh                     # Streamlit config setup (for Heroku-style deploys)
├── Procfile                     # Process file for deployment
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/car-price-predictor.git
   cd car-price-predictor
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure `LinearRegression.pkl` and `car.pkl` are present in the project root. If not, run through `car_price_predictor.ipynb` to regenerate them from `dataset/quikr_car.csv`.

4. Run the app locally:
   ```bash
   streamlit run app.py
   ```

5. Open the URL shown in your terminal (typically `http://localhost:8501`).

## 🌐 Deployment

This app is deployed on **Streamlit Community Cloud**:
👉 **[donna7-car-price-predictor.streamlit.app](https://donna7-car-price-predictor.streamlit.app)**

To deploy your own copy:
1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.
3. Select the repository, branch, and `app.py` as the entry point.
4. Deploy — Streamlit Cloud will install dependencies from `requirements.txt` automatically.

## 📓 Notebook

`car_price_predictor.ipynb` contains the full, step-by-step pipeline used to build this system:
- Dataset loading and exploration
- Data cleaning (fixing `year`, `Price`, `kms_driven` formatting, handling missing values, trimming `name`)
- Outlier detection and removal via boxplots
- Feature/target split and train-test split
- One-Hot Encoding setup with `ColumnTransformer`
- Linear Regression model training inside a scikit-learn `Pipeline`
- Model evaluation (R² score)
- Serialization of the final `LinearRegression.pkl` and `car.pkl` used by the Streamlit app

## 🔮 Possible Improvements

- Try more advanced regression models (Random Forest, Gradient Boosting, XGBoost) and compare R² scores
- Add more features if available (mileage, transmission type, number of owners, engine capacity)
- Apply regularization (Ridge/Lasso) to reduce overfitting from high-cardinality one-hot encoded features
- Add input validation in the app to prevent unrealistic combinations (e.g. car name unmatched to company)

## 📄 License

This project is intended for educational and portfolio purposes. The dataset is sourced from used car listings on [Quikr](https://www.quikr.com/).
