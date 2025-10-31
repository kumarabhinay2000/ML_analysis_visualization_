import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import uuid

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in {'csv','xlsx'}

def clean_data(df):
    return df.dropna()

def generate_chart(df, x_col, y_col, chart_type='scatter'):
    plt.figure(figsize=(10,6))
    if chart_type == 'scatter':
        sns.scatterplot(data=df,x=x_col,y=y_col)
    elif chart_type=='line':
        sns.lineplot(data=df, x=x_col,y=y_col)
    elif chart_type=='bar':
        sns.barplot(data=df, x=x_col,y=y_col)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'{chart_type.title()} Chart of {y_col} vs {x_col} ')
    plt.tight_layout()

    chart_path='static/images/chart.png'
    plt.savefig(chart_path)
    filename=f"static/images/chart_{uuid.uuid4().hex}.png"
    plt.savefig(filename,bbox_inches='tight')
    plt.close()
    return filename