from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


st.set_page_config(
    page_title="AI Salary Predictor",
    page_icon=":bar_chart:",
    layout="wide",
)

DATA_PATH = Path(__file__).parent / "ai_job_dataset.csv"


def get_skill_vocabulary(data: pd.DataFrame) -> list[str]:
    return sorted(
        {
            skill.strip()
            for values in data["required_skills"].dropna()
            for skill in values.split(",")
        }
    )


def add_features(
    data: pd.DataFrame,
    skill_vocabulary: list[str] | None = None,
) -> pd.DataFrame:
    """Create the same date and skill features used during notebook training."""
    frame = data.copy()
    frame["posting_date"] = pd.to_datetime(frame["posting_date"])
    frame["application_deadline"] = pd.to_datetime(frame["application_deadline"])
    frame["posting_year"] = frame["posting_date"].dt.year
    frame["posting_month"] = frame["posting_date"].dt.month
    frame["days_to_deadline"] = (
        frame["application_deadline"] - frame["posting_date"]
    ).dt.days

    skills = skill_vocabulary or get_skill_vocabulary(frame)
    for skill in skills:
        column_name = "skill_" + skill.lower().replace(" ", "_")
        frame[column_name] = frame["required_skills"].fillna("").apply(
            lambda value, selected_skill=skill: int(
                selected_skill in [item.strip() for item in value.split(",")]
            )
        )

    return frame.drop(
        columns=[
            "job_id",
            "salary_usd",
            "required_skills",
            "posting_date",
            "application_deadline",
        ],
        errors="ignore",
    )


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def train_model(data: pd.DataFrame):
    features = add_features(data, get_skill_vocabulary(data))
    target = data["salary_usd"]
    categorical_features = features.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = features.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("selector", SelectKBest(score_func=f_regression, k="all")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )
    model.fit(features, target)
    return model


def selectbox_options(data: pd.DataFrame, column: str) -> list[str]:
    return sorted(data[column].dropna().astype(str).unique().tolist())


def main() -> None:
    data = load_data()
    model = train_model(data)
    skill_vocabulary = get_skill_vocabulary(data)

    st.title("AI Job Salary Predictor")
    st.write("Estimate the expected salary in USD from job and company details.")

    with st.sidebar:
        st.subheader("Dataset")
        st.metric("Job records", f"{len(data):,}")
        st.metric("Salary range", f"${data.salary_usd.min():,.0f} - ${data.salary_usd.max():,.0f}")
        st.caption("The model is trained from ai_job_dataset.csv when the app starts.")

    st.subheader("Job details")
    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            job_title = st.selectbox("Job title", selectbox_options(data, "job_title"))
            experience_level = st.selectbox(
                "Experience level", selectbox_options(data, "experience_level")
            )
            employment_type = st.selectbox(
                "Employment type", selectbox_options(data, "employment_type")
            )
            education_required = st.selectbox(
                "Education required", selectbox_options(data, "education_required")
            )
            industry = st.selectbox("Industry", selectbox_options(data, "industry"))
            company_size = st.selectbox(
                "Company size", selectbox_options(data, "company_size")
            )
        with right:
            salary_currency = st.selectbox(
                "Salary currency", selectbox_options(data, "salary_currency")
            )
            company_location = st.selectbox(
                "Company location", selectbox_options(data, "company_location")
            )
            employee_residence = st.selectbox(
                "Employee residence", selectbox_options(data, "employee_residence")
            )
            company_name = st.selectbox(
                "Company name", selectbox_options(data, "company_name")
            )
            remote_ratio = st.slider("Remote work (%)", 0, 100, 50, step=50)
            years_experience = st.number_input(
                "Years of experience", min_value=0, max_value=50, value=3, step=1
            )

        col_one, col_two, col_three = st.columns(3)
        with col_one:
            description_length = st.number_input(
                "Job description length", min_value=1, value=1200, step=50
            )
        with col_two:
            benefits_score = st.number_input(
                "Benefits score", min_value=0.0, max_value=10.0, value=7.0, step=0.1
            )
        with col_three:
            required_skills = st.multiselect(
                "Required skills",
                sorted(
                    {
                        skill.strip()
                        for values in data.required_skills.dropna()
                        for skill in values.split(",")
                    }
                ),
                default=["Python"],
            )

        posting_date = st.date_input("Posting date", value=pd.Timestamp.today().date())
        deadline = st.date_input(
            "Application deadline",
            value=(pd.Timestamp.today() + pd.Timedelta(days=30)).date(),
        )
        submitted = st.form_submit_button("Predict salary", type="primary")

    if submitted:
        if deadline < posting_date:
            st.error("Application deadline must be on or after the posting date.")
            return

        row = pd.DataFrame(
            [
                {
                    "job_id": "PREDICTION",
                    "job_title": job_title,
                    "salary_currency": salary_currency,
                    "experience_level": experience_level,
                    "employment_type": employment_type,
                    "company_location": company_location,
                    "company_size": company_size,
                    "employee_residence": employee_residence,
                    "remote_ratio": remote_ratio,
                    "required_skills": ", ".join(required_skills),
                    "education_required": education_required,
                    "years_experience": years_experience,
                    "industry": industry,
                    "posting_date": posting_date,
                    "application_deadline": deadline,
                    "job_description_length": description_length,
                    "benefits_score": benefits_score,
                    "company_name": company_name,
                }
            ]
        )
        prediction = float(model.predict(add_features(row, skill_vocabulary))[0])
        st.success(f"Estimated salary: ${prediction:,.0f} USD")
        st.caption("This is an estimate based on patterns in the supplied dataset.")


if __name__ == "__main__":
    main()
