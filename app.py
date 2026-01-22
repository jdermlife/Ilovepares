import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

# -----------------------------------------------------------------------------
# CONFIGURATION & UTILS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Credit Risk Engine", layout="wide")

def main():
    st.title("🏦 Credit Risk Engine: Scorecard & ECL")
    st.markdown("""
    This application allows you to:
    1. **Develop a Credit Scorecard:** Train a logistic regression model to predict probability of default (PD).
    2. **Compute ECL:** Calculate Estimated Credit Loss based on PD, LGD, and EAD.
    """)

    # Sidebar Navigation
    page = st.sidebar.radio("Navigation", ["1. Data Upload", "2. Scorecard Development", "3. ECL Computation"])

    # Session State for Data Persistence
    if 'data' not in st.session_state:
        st.session_state['data'] = None
    if 'model' not in st.session_state:
        st.session_state['model'] = None
    if 'scored_data' not in st.session_state:
        st.session_state['scored_data'] = None

    # -------------------------------------------------------------------------
    # PAGE 1: DATA UPLOAD
    # -------------------------------------------------------------------------
    if page == "1. Data Upload":
        st.header("📂 Data Upload")
        
        # Option to generate sample data
        if st.button("Generate Sample Data (Demo)"):
            st.session_state['data'] = generate_sample_data()
            st.success("Sample data generated!")
        
        uploaded_file = st.file_uploader("Or upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            st.session_state['data'] = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")

        # Display Data
        if st.session_state['data'] is not None:
            st.write("### Data Preview")
            st.dataframe(st.session_state['data'].head())
            
            st.write("### Data Statistics")
            st.write(st.session_state['data'].describe())

    # -------------------------------------------------------------------------
    # PAGE 2: SCORECARD DEVELOPMENT
    # -------------------------------------------------------------------------
    elif page == "2. Scorecard Development":
        st.header("🛠️ Scorecard Development")
        
        df = st.session_state['data']
        if df is None:
            st.warning("Please upload or generate data first.")
            return

        # 1. Variable Selection
        st.subheader("1. Variable Configuration")
        cols = df.columns.tolist()
        
        target_col = st.selectbox("Select Target Variable (0=Good, 1=Default)", cols, index=cols.index("default") if "default" in cols else 0)
        feature_cols = st.multiselect("Select Independent Variables (Features)", [c for c in cols if c != target_col], default=[c for c in cols if c != target_col and c not in ['id', 'EAD', 'LGD']])

        if st.button("Train Model"):
            if not feature_cols:
                st.error("Please select at least one feature.")
            else:
                # Preprocessing
                X = df[feature_cols].fillna(0) # Simple imputation for demo
                y = df[target_col]

                # Train Test Split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                # Logistic Regression
                model = LogisticRegression(class_weight='balanced', max_iter=1000)
                model.fit(X_train, y_train)
                
                # Save model to session
                st.session_state['model'] = model
                st.session_state['features'] = feature_cols
                st.session_state['target'] = target_col

                # Predictions
                y_prob = model.predict_proba(X_test)[:, 1]
                
                # Metrics
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)

                # ----------------------------------
                # SCORING LOGIC (Standard Scaling)
                # Score = Offset + Factor * ln(odds)
                # Target Score = 600 at 50:1 odds, PDO = 20
                # ----------------------------------
                PDO = 20
                Base_Score = 600
                Base_Odds = 50
                Factor = PDO / np.log(2)
                Offset = Base_Score - (Factor * np.log(Base_Odds))
                
                # Calculate Scores for the whole dataset
                all_probs = model.predict_proba(X[feature_cols])[:, 1]
                # Avoid log(0)
                epsilon = 1e-6
                all_odds = (1 - all_probs) / (all_probs + epsilon) # Odds of being Good
                st.session_state['data']['PD_Predicted'] = all_probs
                st.session_state['data']['Credit_Score'] = Offset + (Factor * np.log(all_odds))
                st.session_state['data']['Credit_Score'] = st.session_state['data']['Credit_Score'].astype(int)
                st.session_state['scored_data'] = st.session_state['data']

                # ----------------------------------
                # DISPLAY RESULTS
                # ----------------------------------
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### Model Performance")
                    st.metric("ROC AUC Score", f"{roc_auc:.4f}")
                    
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('Receiver Operating Characteristic')
                    ax.legend(loc="lower right")
                    st.pyplot(fig)

                with col2:
                    st.write("### Feature Importance (Coefficients)")
                    coef_df = pd.DataFrame({
                        'Feature': feature_cols,
                        'Coefficient': model.coef_[0]
                    }).sort_values(by='Coefficient', ascending=False)
                    st.dataframe(coef_df)

                st.success("Model Trained and Scores Generated! Go to the ECL tab.")

    # -------------------------------------------------------------------------
    # PAGE 3: ECL COMPUTATION
    # -------------------------------------------------------------------------
    elif page == "3. ECL Computation":
        st.header("📉 Estimated Credit Loss (ECL)")
        
        df = st.session_state.get('scored_data')
        
        if df is None:
            st.warning("Please train the scorecard first to generate PDs, or upload a file with PD columns.")
            return
            
        st.subheader("1. ECL Mapping")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pd_col = st.selectbox("Select PD Column", df.columns, index=df.columns.get_loc("PD_Predicted") if "PD_Predicted" in df.columns else 0)
        
        with col2:
            # Check if LGD exists in data, else ask for manual input
            if 'LGD' in df.columns:
                lgd_input_type = st.radio("LGD Source", ["From Column", "Fixed Value"])
                if lgd_input_type == "From Column":
                    lgd_val = st.selectbox("Select LGD Column", df.columns, index=df.columns.get_loc("LGD"))
                    lgd_is_col = True
                else:
                    lgd_val = st.number_input("Fixed LGD Value (e.g. 0.45)", 0.0, 1.0, 0.45)
                    lgd_is_col = False
            else:
                lgd_val = st.number_input("Fixed LGD Value (e.g. 0.45)", 0.0, 1.0, 0.45)
                lgd_is_col = False
                
        with col3:
            if 'EAD' in df.columns:
                ead_col = st.selectbox("Select EAD Column", df.columns, index=df.columns.get_loc("EAD"))
            else:
                st.error("Dataset must contain an Exposure at Default (EAD) column.")
                return

        if st.button("Calculate ECL"):
            # Calculation
            # ECL = PD * LGD * EAD
            
            pd_series = df[pd_col]
            ead_series = df[ead_col]
            
            if lgd_is_col:
                lgd_series = df[lgd_val]
            else:
                lgd_series = lgd_val
                
            df['ECL_Value'] = pd_series * lgd_series * ead_series
            
            total_ecl = df['ECL_Value'].sum()
            total_ead = df[ead_col].sum()
            avg_pd = df[pd_col].mean()
            
            # Summary Metrics
            st.write("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Portfolio Exposure", f"${total_ead:,.2f}")
            m2.metric("Total Expected Credit Loss", f"${total_ecl:,.2f}")
            m3.metric("Coverage Ratio (ECL/Exposure)", f"{(total_ecl/total_ead)*100:.2f}%")
            
            # Visualizations
            st.write("### ECL Distribution by Credit Score")
            
            # Binning Scores for visualization
            df['Score_Bin'] = pd.cut(df['Credit_Score'], bins=5)
            grouped = df.groupby('Score_Bin')[['ECL_Value', ead_col]].sum().reset_index()
            grouped['Risk_Density'] = grouped['ECL_Value'] / grouped[ead_col]
            
            st.dataframe(grouped.style.format({
                'ECL_Value': '${:,.2f}', 
                ead_col: '${:,.2f}',
                'Risk_Density': '{:.2%}'
            }))
            
            # Chart
            fig, ax1 = plt.subplots(figsize=(10,6))
            sns.barplot(data=grouped, x='Score_Bin', y='ECL_Value', alpha=0.6, ax=ax1, color='blue')
            ax2 = ax1.twinx()
            sns.lineplot(data=grouped, x='Score_Bin', y='Risk_Density', marker='o', ax=ax2, color='red')
            
            ax1.set_ylabel("Total ECL Amount ($)", color='blue')
            ax2.set_ylabel("Risk Density (ECL/EAD)", color='red')
            ax1.set_title("ECL and Risk Density per Score Bucket")
            ax1.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
            
            # Download Result
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Results CSV",
                csv,
                "ecl_results.csv",
                "text/csv",
                key='download-csv'
            )

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def generate_sample_data(n=1000):
    """Generates a dummy credit dataset for demonstration."""
    np.random.seed(42)
    data = pd.DataFrame({
        'id': range(1, n+1),
        'income': np.random.normal(50000, 15000, n),
        'age': np.random.randint(21, 70, n),
        'debt_to_income': np.random.uniform(0.1, 0.8, n),
        'loan_amount': np.random.randint(1000, 50000, n),
        'credit_utilization': np.random.uniform(0, 1, n),
        'EAD': np.random.randint(1000, 50000, n)  # Using loan amount as EAD for simplicity
    })
    
    # Create a synthetic default relationship
    # Higher debt, utilization, and lower income -> Higher log odds of default
    log_odds = -4 + (3 * data['credit_utilization']) + (2 * data['debt_to_income']) - (0.00005 * data['income'])
    probability = 1 / (1 + np.exp(-log_odds))
    data['default'] = np.random.binomial(1, probability)
    
    # Add an LGD column (e.g., Unsecured loans have higher LGD)
    data['LGD'] = np.random.choice([0.45, 0.75], n, p=[0.7, 0.3])
    
    return data

if __name__ == "__main__":
    main()