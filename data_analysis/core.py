import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2_contingency
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import io

# Handling Colab-specific imports gracefully
try:
    from google.colab import files
    from IPython.display import HTML, display
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


class PlottingMethods:
    """Handles granular, modular chart generation returning HTML-wrapped figures."""
    
    def display_image(self, result_dict):
        """Helper to render HTML plots in Colab."""
        if IN_COLAB and 'html' in result_dict:
            display(HTML(result_dict['html']))
        elif not IN_COLAB:
            print("Display function requires Google Colab environment.")
            
    def plot_bar_chart(self, x, y, data, color=None, barmode='group'):
        """Generates a bar chart."""
        if data is None or data.empty:
            return {'status': 'error', 'message': 'Data is empty'}
        
        fig = px.bar(data, x=x, y=y, color=color, barmode=barmode, title=f"Bar Chart: {y} by {x}")
        return {'status': 'success', 'html': fig.to_html(full_html=False, include_plotlyjs='cdn')}

    def plot_pie_chart(self, names, values, data, hole=0.0, title=''):
        """Generates a pie/donut chart."""
        if data is None or data.empty:
            return {'status': 'error', 'message': 'Data is empty'}
            
        fig = px.pie(data, names=names, values=values, hole=hole, title=title)
        return {'status': 'success', 'html': fig.to_html(full_html=False, include_plotlyjs='cdn')}
        
    def plot_histogram(self, x, data, bins=None, title=''):
        """Generates a histogram."""
        if data is None or data.empty:
            return {'status': 'error', 'message': 'Data is empty'}
            
        fig = px.histogram(data, x=x, nbins=bins, title=title)
        return {'status': 'success', 'html': fig.to_html(full_html=False, include_plotlyjs='cdn')}


class DataInspector:
    """End-to-end tool for CSV data ingestion, cleaning, and statistical visualization."""
    
    def __init__(self):
        self.df = None
        self.plotter = PlottingMethods()

    # --- 1. Data Ingestion & Sanitization ---
    def upload_data(self):
        """Handles local file uploads in Google Colab and cleans garbage strings."""
        if not IN_COLAB:
            print("Upload method is only supported in Google Colab.")
            return

        print("Please upload your CSV file:")
        uploaded = files.upload()
        if not uploaded:
            print("No file uploaded.")
            return
            
        filename = list(uploaded.keys())[0]
        
        # Auto-handle garbage strings by converting them to actual NaNs
        missing_values = ['?', 'n/a', 'NULL', ' ', 'N/A', 'NA', '--']
        self.df = pd.read_csv(io.BytesIO(uploaded[filename]), na_values=missing_values)
        
        # Auto-Type Correction: force numeric if possible without destroying the column
        for col in self.df.columns:
            converted = pd.to_numeric(self.df[col], errors='coerce')
            if converted.notna().sum() > 0:
                self.df[col] = converted
                
        print(f"Successfully loaded {filename} with {self.df.shape[0]} rows and {self.df.shape[1]} columns.")

    # --- 2. Structural Analysis & Cleaning ---
    def get_summary(self):
        """Displays row/column counts, previews 20 rows, breaks down types."""
        if self.df is None: return "No data loaded."
        
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        print(f"Dataset Shape: {self.df.shape[0]} Rows, {self.df.shape[1]} Columns")
        print(f"Numerical Columns ({len(num_cols)}): {num_cols}")
        print(f"Categorical Columns ({len(cat_cols)}): {cat_cols}")
        print("\nFirst 20 Rows:")
        display(self.df.head(20))

    def handle_missing_values(self, strategy='mean', constant_value=None):
        """Imputes missing values based on strategy (mean, median, mode, constant)."""
        if self.df is None: return
        
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns

        for col in num_cols:
            if strategy == 'mean':
                self.df[col] = self.df[col].fillna(self.df[col].mean())
            elif strategy == 'median':
                self.df[col] = self.df[col].fillna(self.df[col].median())
            elif strategy == 'constant' and constant_value is not None:
                self.df[col] = self.df[col].fillna(constant_value)
                
        for col in cat_cols:
            if strategy == 'mode':
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            elif strategy == 'constant' and constant_value is not None:
                self.df[col] = self.df[col].fillna(constant_value)
                
        print(f"Missing values handled using '{strategy}' strategy.")

    def remove_duplicates(self):
        """Removes exact row matches."""
        if self.df is None: return
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        print(f"Removed {initial_rows - len(self.df)} duplicate rows.")

    def handle_outliers(self, columns, find_and_delete=False):
        """IQR-based outlier detection."""
        if self.df is None: return
        
        for col in columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                print(f"Column '{col}' has {len(outliers)} outliers.")
                
                if find_and_delete:
                    self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
                    print(f"Outliers removed from '{col}'.")

    def delete_columns(self, columns_to_drop):
        """Removes specified columns based on a comma-separated string."""
        if self.df is None: return
        cols = [c.strip() for c in columns_to_drop.split(',')]
        self.df = self.df.drop(columns=[c for c in cols if c in self.df.columns], errors='ignore')
        print(f"Columns dropped. Current shape: {self.df.shape}")

    def delete_rows(self, rows_to_drop):
        """Removes specified rows based on a comma-separated string of indices."""
        if self.df is None: return
        indices = [int(i.strip()) for i in rows_to_drop.split(',')]
        self.df = self.df.drop(index=indices, errors='ignore')
        print(f"Rows dropped. Current shape: {self.df.shape}")

    # --- 3. Feature Engineering Preparation ---
    def extract_normalized_numeric_data(self, method='standard'):
        """Supports minmax, standard, and robust scaling."""
        if self.df is None: return pd.DataFrame()
        
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0: return pd.DataFrame()
        
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
            
        scaled_data = scaler.fit_transform(self.df[num_cols])
        return pd.DataFrame(scaled_data, columns=num_cols)

    def extract_normalized_categorical_data(self, method='onehot'):
        """Supports onehot, ordinal, and uniform encoding."""
        if self.df is None: return pd.DataFrame()
        
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns
        if len(cat_cols) == 0: return pd.DataFrame()
        
        if method == 'ordinal' or method == 'uniform':
            encoder = OrdinalEncoder()
            encoded_data = encoder.fit_transform(self.df[cat_cols].astype(str))
            df_encoded = pd.DataFrame(encoded_data, columns=cat_cols)
            if method == 'uniform':
                # Scale the ordinal data 0-1
                scaler = MinMaxScaler()
                df_encoded = pd.DataFrame(scaler.fit_transform(df_encoded), columns=cat_cols)
            return df_encoded
        else: 
            return pd.get_dummies(self.df[cat_cols], drop_first=True)

    def create_normalized_data_df(self):
        """Merges scaled numeric data and encoded categorical data into one unified DataFrame."""
        if self.df is None: return pd.DataFrame()
        
        num_df = self.extract_normalized_numeric_data(method='standard')
        cat_df = self.extract_normalized_categorical_data(method='onehot')
        
        merged_df = pd.concat([num_df, cat_df], axis=1)
        print("Successfully merged normalized numeric and categorical data.")
        return merged_df

    # --- 4 & 5. Advanced Interactive Visualization & Deep Statistics ---
    def plot_numerical(self, column_names):
        """Generates a 3-panel subplot (Violin, Scatter, Histogram)."""
        if self.df is None: return
        
        for col in column_names:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                fig = make_subplots(rows=1, cols=3, subplot_titles=("Violin Plot", "Scatter Plot", "Histogram"))
                
                fig.add_trace(go.Violin(x=self.df[col], name=col, orientation='h'), row=1, col=1)
                fig.add_trace(go.Scatter(y=self.df[col], mode='markers', name=col), row=1, col=2)
                fig.add_trace(go.Histogram(x=self.df[col], name=col), row=1, col=3)
                
                fig.update_layout(title_text=f"Distribution Analysis for {col}", showlegend=False)
                fig.show()

    def plot_relationship(self, col1, col2):
        """Detects column types and chooses Scatter, Box, or Grouped Bar."""
        if self.df is None: return
        
        is_col1_num = pd.api.types.is_numeric_dtype(self.df[col1])
        is_col2_num = pd.api.types.is_numeric_dtype(self.df[col2])
        
        if is_col1_num and is_col2_num:
            # Num-Num: Scatter with OLS Trendline
            fig = px.scatter(self.df, x=col1, y=col2, trendline="ols", title=f"Scatter: {col1} vs {col2}")
        elif not is_col1_num and not is_col2_num:
            # Cat-Cat: Grouped Bar with percentages
            counts = self.df.groupby([col1, col2]).size().reset_index(name='Count')
            counts['Percentage'] = counts.groupby(col1)['Count'].transform(lambda x: (x / x.sum()) * 100)
            fig = px.bar(counts, x=col1, y="Count", color=col2, barmode='group', 
                         text=counts['Percentage'].apply(lambda x: f'{x:.1f}%'),
                         title=f"Bar: {col1} grouped by {col2}")
        else:
            # Cat-Num: Box plot with all data points
            x_col = col1 if not is_col1_num else col2
            y_col = col2 if not is_col1_num else col1
            fig = px.box(self.df, x=x_col, y=y_col, points="all", title=f"Box: {y_col} by {x_col}")
            
        fig.show()

    def plot_all_associations_heatmap(self):
        """Visualizes relationships across all data types (Pearson & Cramér's V)."""
        if self.df is None: return
        
        cols = self.df.columns
        corr_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
        
        for col1 in cols:
            for col2 in cols:
                if col1 == col2:
                    corr_matrix.loc[col1, col2] = 1.0
                    continue
                
                is_col1_num = pd.api.types.is_numeric_dtype(self.df[col1])
                is_col2_num = pd.api.types.is_numeric_dtype(self.df[col2])
                
                if is_col1_num and is_col2_num:
                    # Pearson for Numeric-Numeric
                    corr, _ = stats.pearsonr(self.df[col1].dropna(), self.df[col2].dropna())
                    corr_matrix.loc[col1, col2] = corr
                    
                elif not is_col1_num and not is_col2_num:
                    # Cramér's V for Categorical-Categorical
                    confusion_matrix = pd.crosstab(self.df[col1], self.df[col2])
                    chi2 = chi2_contingency(confusion_matrix)[0]
                    n = confusion_matrix.sum().sum()
                    phi2 = chi2 / n
                    r, k = confusion_matrix.shape
                    cramers_v = np.sqrt(phi2 / min((k-1), (r-1))) if min((k-1), (r-1)) > 0 else 0
                    corr_matrix.loc[col1, col2] = cramers_v
                    
                else:
                    # Mixed (Num-Cat) defaults to 0 to keep the visual clean
                    corr_matrix.loc[col1, col2] = 0.0 
                    
        fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", 
                        title="Unified Association Heatmap (Pearson & Cramér's V)",
                        color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        fig.show()
