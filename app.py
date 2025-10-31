from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, session
from werkzeug.utils import secure_filename
import os
import pandas as pd
from forms import FileUploadForm
from utils import allowed_file, clean_data, generate_chart #, generate_correlation_heatmap, generate_outlier_plot
from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS

app = Flask(__name__,
    template_folder=os.path.abspath("E:/FLASK/Data_Visulize_F/templates/"))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'bfvjesvjdsfew345'


@app.route('/', methods=['GET', 'POST'])
def index():
    form = FileUploadForm()
    if form.validate_on_submit():
        file = form.dataset.data
        filename = secure_filename(file.filename)
        if allowed_file(filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            file.save(filepath)

            # Save filepath in session
            session['uploaded_file_path'] = filepath

            flash("File Uploaded Successfully", 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Unsupported file type', 'error')
    return render_template('index.html', form=form)


def load_uploaded_df():
    file_path = session.get('uploaded_file_path')
    if not file_path or not os.path.exists(file_path):
        flash('Dataset not found.', 'error')
        return None
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        return pd.read_excel(file_path)


@app.route('/dashboard')
def dashboard():
    df = load_uploaded_df()
    if df is None:
        return redirect(url_for('index'))
    columns = list(df.columns)
    return render_template('dashboard.html', columns=columns)


@app.route('/generate_chart', methods=['POST'])
def generate_chart_route():
    df = load_uploaded_df()
    if df is None:
        return jsonify({'error': "Dataset not loaded."})

    x_col = request.form.get('x_col')
    y_col = request.form.get('y_col')
    chart_type = request.form.get('chart_type')

    if not x_col or not y_col:
        return jsonify({'error': "Please select valid columns."})

    chart_path = generate_chart(df, x_col, y_col, chart_type)
    return jsonify({'chart_url': '/' + chart_path})


@app.route('/correlation')
def correlation():
    df = load_uploaded_df()
    if df is None:
        return redirect(url_for('index'))

    numeric_df = df.select_dtypes(include=['number'])

    if numeric_df.empty:
        flash('No Numeric columns available.', 'error')
        return redirect(url_for('dashboard'))

    # Fix: Use numeric_df for correlation
    corr = numeric_df.corr()
    chart_path = generate_correlation_heatmap(corr)

    return render_template('correlation.html', chart_url='/' + chart_path)


@app.route('/regression', methods=['POST'])
def regression():
    df = load_uploaded_df()
    if df is None:
        return redirect(url_for('index'))

    x_col = request.form['x_col']
    y_col = request.form['y_col']

    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt

    X = df[[x_col]].values.reshape(-1, 1)
    Y = df[y_col].values

    # Fix: Remove rows where X or Y is NaN
    valid_mask = ~pd.isnull(X.flatten()) & ~pd.isnull(Y)
    X = X[valid_mask].reshape(-1, 1)
    Y = Y[valid_mask]

    model = LinearRegression()
    model.fit(X, Y)
    Y_pred = model.predict(X)

    path = 'static/images/regression_plot.png'
    plt.figure(figsize=(10,6))
    plt.scatter(X, Y, label='Data')
    plt.plot(X, Y_pred, color='red', label='Regression Line')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return render_template('regression.html', chart_url='/' + path)



@app.route('/trend')
def trend():
    df = load_uploaded_df()
    if df is None:
        return redirect(url_for('index'))

    time_col = request.args.get('time_col')
    value_col = request.args.get('value_col')

    # Safety Check: Columns exist in DataFrame
    if time_col not in df.columns or value_col not in df.columns:
        flash('Selected columns not found in dataset.', 'error')
        return redirect(url_for('dashboard'))

    # Convert time_col to datetime safely
    try:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    except Exception as e:
        flash(f'Error converting {time_col} to datetime: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

    # Drop rows where time_col or value_col are NaN
    df = df.dropna(subset=[time_col, value_col])

    if df.empty:
        flash('No valid data available for trend analysis after cleaning.', 'error')
        return redirect(url_for('dashboard'))

    df_sorted = df.sort_values(time_col)

    import matplotlib.pyplot as plt
    path = 'static/images/trend_plot.png'
    plt.figure(figsize=(12,6))
    plt.plot(df_sorted[time_col], df_sorted[value_col], marker='o')
    plt.xlabel('Time')
    plt.ylabel(value_col)
    plt.title(f'Trend Analysis: {value_col} over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return render_template('trend.html', chart_url='/' + path)

import matplotlib.pyplot as plt
import seaborn as sns
import os
import uuid  # To generate unique filenames

def generate_correlation_heatmap(corr):
    # Ensure the output directory exists
    output_dir = 'static/images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate a unique filename for every heatmap
    filename = f'corr_heatmap_{uuid.uuid4().hex}.png'
    filepath = os.path.join(output_dir, filename)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, square=True)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    return filepath  # Return relative path to be used in HTML

@app.route('/outliers')
def outliers():
    df = load_uploaded_df()
    if df is None:
        return redirect(url_for('index'))

    col = request.args.get('col')
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    outliers_df = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]

    import matplotlib.pyplot as plt
    path = 'static/images/outlier_plot.png'
    plt.figure(figsize=(10,6))
    plt.scatter(df.index, df[col], label='Data')
    plt.scatter(outliers_df.index, outliers_df[col], color='red', label='Outliers')
    plt.xlabel('Index')
    plt.ylabel(col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return render_template('outliers.html', chart_url='/' + path)


@app.route('/clustering')
def clustering():
    df = load_uploaded_df()
    if df is None:
        return redirect(url_for('index'))

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    df_numeric = df[numeric_cols].dropna()

    kmeans = KMeans(n_clusters=3)
    clusters = kmeans.fit_predict(df_numeric)

    pca = PCA(n_components=2)
    components = pca.fit_transform(df_numeric)

    path = 'static/images/clustering_pca.png'
    plt.figure(figsize=(10,6))
    plt.scatter(components[:, 0], components[:, 1], c=clusters, cmap='viridis')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Clustering + PCA Visualization')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return render_template('clustering.html', chart_url='/' + path)


if __name__ == '__main__':
    app.run(debug=True)
