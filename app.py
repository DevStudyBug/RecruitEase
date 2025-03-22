import pickle
import os
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Create app with explicit template and static folder configuration
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'your_secret_key_here'  # Secret key for session management

# Ensure necessary static folders exist
os.makedirs('static/images', exist_ok=True)
os.makedirs('static/styles', exist_ok=True)
os.makedirs('static/script', exist_ok=True)

# Load dataset and models
try:
    df = pd.read_csv('HR_comma_sep.csv')
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError as e:
    print(f"Error loading required files: {e}")
    raise

# Data cleaning function
def reading_cleaning(df):
    df.drop_duplicates(inplace=True)
    df.columns = [x.lower() for x in df.columns]
    return df

df = reading_cleaning(df)

# Employee important info function
def employee_important_info(df):
    return (
        df['satisfaction_level'].mean(),
        df.groupby('department')['satisfaction_level'].mean().to_dict(),
        df.groupby('salary')['satisfaction_level'].mean().to_dict(),
        len(df[df['left'] == 1]),
        len(df[df['left'] == 0])
    )

# Plot generation functions
def save_plot(plot_func, filename):
    plt.figure(figsize=(15, 10))
    plot_func()
    save_path = os.path.join('static', 'images', filename)
    plt.savefig(save_path)
    plt.close()

def plots(df, col):
    save_plot(lambda: plt.pie(df[col].value_counts(), labels=df[col].unique(), autopct='%1.1f%%', startangle=40), f'{col}.png')

def distribution(df, col):
    save_plot(lambda: sns.countplot(x=df[col], hue=df['left'], palette='Set1', data=df), f'{col}_distribution.png')

def comparison(df, x, y):
    save_plot(lambda: sns.barplot(x=x, y=y, hue='left', data=df, ci=None), 'comparison.png')

def corr_with_left(df):
    save_plot(lambda: df.corr()['left'].drop('left').plot(kind='barh'), 'correlation.png')

def histogram(df, col):
    save_plot(lambda: sns.histplot(df[col], bins=20, hue=df['left']), f'{col}_histogram.png')

def prediction(sl_no, gender, ssc_p, hsc_p, degree_p, workex, etest_p, specialisation, mba_p):
    try:
        data = pd.DataFrame({
            'sl_no': [float(sl_no)],
            'gender': [1 if gender == 'Male' else 0],
            'ssc_p': [float(ssc_p)],
            'hsc_p': [float(hsc_p)],
            'degree_p': [float(degree_p)],
            'workex': [1 if workex == 'Yes' else 0],
            'etest_p': [float(etest_p)],
            'specialisation': [1 if specialisation == 'Mkt&HR' else 0],
            'mba_p': [float(mba_p)]
        })
        scaled_df = scaler.transform(data)
        return model.predict(scaled_df)[0]
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

# Routes
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/ana')
def ana():
    try:
        avg_sat, dept_sat, sal_sat, left_count, stay_count = employee_important_info(df)
        plots(df, 'left')
        plots(df, 'salary')
        plots(df, 'department')
        distribution(df, 'salary')
        distribution(df, 'department')
        comparison(df, 'department', 'satisfaction_level')
        corr_with_left(df)
        histogram(df, 'satisfaction_level')
        return render_template('ana.html', df=df.head(), average_satisfaction=avg_sat, department_satisfaction=dept_sat, salary_satisfaction=sal_sat, left_employees=left_count, stayed_employees=stay_count)
    except Exception as e:
        flash(f"Error generating analytics: {str(e)}", "danger")
        return redirect(url_for('index'))

@app.route("/placement", methods=['POST', 'GET'])
def placement():
    if request.method == 'POST':
        try:
            form_data = {k: request.form[k] for k in ['sl_no', 'gender', 'ssc_p', 'hsc_p', 'degree_p', 'workex', 'etest_p', 'specialisation', 'mba_p']}
            result = prediction(**form_data)
            if result is not None:
                pred = "Placed" if result == 1 else "Not Placed"
                rec = "Best candidate for your business" if result == 1 else "Not the best candidate"
                return render_template('job.html', result=pred, rec=rec)
            flash("Prediction failed", "danger")
        except Exception as e:
            flash(f"Error processing form: {str(e)}", "danger")
    return redirect(url_for('index'))

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

# Serve static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=True, host="0.0.0.0", port=port)
# if __name__ == "__main__":
#     app.run(debug=True)
