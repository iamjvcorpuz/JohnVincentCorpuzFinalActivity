import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

# Title of the app
st.title("Hotel Bookings Analysis Dashboard")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your hotel bookings CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    try:
        data = pd.read_csv(uploaded_file)

        # Check for required columns
        required_columns = ['reservation_id', 'reserved_room_type', 'adr']
        if all(column in data.columns for column in required_columns):
            # Display the first 15 columns of the dataset
            st.write("Data Preview (First 15 Columns):")
            st.write(data.iloc[:, :15])  # Show first 15 columns

            # Process the data for association rule mining
            basket = data.groupby(['reservation_id', 'reserved_room_type'])['adr'].sum().unstack().reset_index().fillna(0)
            basket = (basket > 0).astype(int)  # Convert to binary

            # Frequent Itemsets Mining
            frequent_itemsets = apriori(basket, min_support=0.1, use_colnames=True)

            # Rule Generation
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
            st.write("Association Rules:")
            st.write(rules)

            # Dynamic Visualizations
            if not rules.empty:
                # Bar chart for support
                plt.figure(figsize=(10, 5))
                sns.barplot(x='support', y=rules.index, data=rules, palette='viridis')
                plt.title('Support of Association Rules')
                plt.xlabel('Support')
                plt.ylabel('Rule Index')
                st.pyplot(plt)

                # Heatmap for rule metrics
                plt.figure(figsize=(8, 6))
                heatmap_data = rules.pivot_table(values='lift', index='antecedents', columns='consequents', fill_value=0)
                sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt=".2f")
                plt.title('Lift Heatmap')
                st.pyplot(plt)

            # Filter and Search
            st.sidebar.header("Filter Options")
            support_filter = st.sidebar.slider("Min Support", 0.0, 1.0, 0.1)
            confidence_filter = st.sidebar.slider("Min Confidence", 0.0, 1.0, 0.5)

            filtered_rules = rules[(rules['support'] >= support_filter) & (rules['confidence'] >= confidence_filter)]
            st.write("Filtered Association Rules:")
            st.write(filtered_rules)

            # Recommendations
            if not filtered_rules.empty:
                st.write("Recommended Product Bundles:")
                st.write(filtered_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

        else:
            st.error("The uploaded dataset is missing one or more required columns.")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
