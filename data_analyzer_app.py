# csv_dummies_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import io
import random


def main():
    # st.set_page_config(layout="wide") 
    st.title("KMeans Clustering...")
    
    
    # File uploader widget
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        st.write("Filename:", uploaded_file.name)

        col1, col2 = st.columns(2)

        # with col1:
            # Read CSV data into a DataFrame
        df = pd.read_csv(uploaded_file)
        st.write("File content:")
        st.dataframe(df)  # Display DataFrame as a table

        tempDf = []

        for col in df.columns:
            tempDf.append([col, df[col].isna().any(), df[col].isna().mean().round(5) * 100])

        tempdf1 = pd.DataFrame(tempDf, columns=["Column Name","Contains Missing Data","Percentage of Missing Data"])
        tempdf1["Contains Missing Data"].replace({False:"No",True:"Yes"}, inplace=True)

        st.write(tempdf1)

        replaceNull = st.checkbox("Replace Missing values", value=False, label_visibility="visible")

        if replaceNull:
            selected_cols = st.multiselect("Select columns to replace null", df.columns, placeholder="Choose columns", label_visibility="visible")

            replaceMthd = st.selectbox('Replace Missing Data Using...', ('Mean', 'Median', 'Mode'), placeholder="Select method...", index=None)

            for col in selected_cols:

                if replaceMthd == "Mean":
                    df[col].fillna(value=round(df[col].mean()), inplace=True)
                
                if replaceMthd == "Median":
                    df[col].fillna(value=round(df[col].median()), inplace=True)

                if replaceMthd == "Mode":
                    df[col].fillna(value=round(df[col].mode().iat[0]), inplace=True)

            st.dataframe(df)

        createDummy = st.checkbox("Create dummy Variables", value=False, label_visibility="visible")

        if createDummy:
            # User input for creating dummy variables
            selected_column = st.selectbox("Select a column for dummy variables", df.columns, placeholder="Choose one...", index=None)

            if selected_column:

                df = pd.get_dummies(df, columns=[selected_column])
                st.write("New data with Dummy variables:")
                st.dataframe(df)

        dropCols = st.checkbox("Drop columns", value=False)

        if dropCols:
            selectedColumns = st.multiselect("Select colums to drop",  df.columns, placeholder="Choose columns", label_visibility="visible")
            df = df.drop(columns=selectedColumns, axis=1)
            st.dataframe(df)
            # st.write(selectedColumns)
                
        ready = st.checkbox("Ready for analysis?", value=False)
        
        if ready:

            tab1, tab2 = st.tabs(["Scatter", "Boxplot"])

            with tab1:
                selected_columns = st.multiselect("Select columns to see clustering", df.columns, placeholder="Choose columns", label_visibility="visible", max_selections=2)

                    # filtered_df = df[analyseClusters]

                filtered_df = df[selected_columns]

                colslist = []

                    # st.write(filtered_df)
                for col in filtered_df.columns:
                    colslist.append(col)

                    # st.write(colslist)

                    # with col2:
                if colslist:
                    numOfClusters = st.slider("Number of Clusters",2)
                    if numOfClusters:
                        kmeans = KMeans(n_clusters=numOfClusters,random_state=42).fit(filtered_df)
                                    # df["Cluster"] = kmeans.labels_
                        st.write("Silhouette Score: ", round(silhouette_score(filtered_df,kmeans.labels_),5))
                        # plt.style.use

                        fig, ax = plt.subplots()
                        ax.scatter(x=filtered_df[colslist[0]], y=filtered_df[colslist[1]],c=kmeans.labels_)
                        plt.xlabel(colslist[0])
                        plt.ylabel(colslist[1])
                        # plt.scatter(x=filtered_df[colslist[0]], y=filtered_df[colslist[1]],c=kmeans.labels_)
                        st.pyplot(fig)

                        st.header("Summary")

                        summary_df = []

                        kmeans_all = KMeans(n_clusters=numOfClusters,random_state=42).fit(df)

                        for sum in kmeans_all.cluster_centers_:
                            summary_df.append(sum)
                        st.dataframe(pd.DataFrame(summary_df, columns=df.columns))
                
            with tab2:
                box_columns = st.multiselect("Select columns to create Boxplot", df.columns, placeholder="Choose columns", label_visibility="visible")
                
                # df[box_columns].boxplot(figsize=(10,6))

                box_cols = df[box_columns]

                # for col in box_cols.columns:
                #     box_cols.append(col)

                if box_columns:
                    fig, ax = plt.subplots()
                    ax.boxplot(pd.DataFrame(df[box_columns].values))
                    ax.set_xticklabels(df[box_columns].keys())
                    plt.grid(alpha=0.3)
                    st.pyplot(fig)

                    box_summary = []

                    st.header("Data Descriptor:")
                    for col in box_cols.columns:
                        box_summary.append(df[col].describe())
                    st.dataframe(pd.DataFrame(box_summary))                  


if __name__ == "__main__":
    main()