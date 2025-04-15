from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
from datetime import datetime
import logging
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import xgboost

app = Flask(__name__)

# Secret key for session management
app.secret_key = 'your_secret_key'


# MySQL configurations
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'Claims'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE username = %s AND password = %s', (username, password))
        account = cursor.fetchone()
        
        if account:
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            return redirect(url_for('dashboard'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('login.html', msg=msg)

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE username = %s OR email = %s', (username, email))
        account = cursor.fetchone()

        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
            cursor.execute('INSERT INTO users (username, password, email) VALUES (%s, %s, %s)', (username, password, email))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
    return render_template('register.html', msg=msg)

@app.route('/dashboard')
def dashboard():
    if 'loggedin' in session:
        return render_template('dashboard.html', username=session['username'])

# Load the pre-trained model
try:
    with open('ClaimClassifier.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully.")
    print(f"Loaded object type: {type(model)}")
    #print(f"Object contents: {model}")
    if isinstance(model, xgboost.sklearn.XGBClassifier):

        # get feature importance instead of feature weights
        feature_importance = model.feature_importances_
        print(feature_importance)
except FileNotFoundError:
    raise Exception("Model file not found. Ensure 'classification_model.pkl' exists.")
except Exception as e:
    raise Exception(f"Error loading model: {e}")

@app.route('/claim_classification',methods=['GET','POST'])
def claim_classification():
    if request.method == 'POST':
        try:
            # Log received form data
            logger.debug("Received form data: %s",request.form)

            # Get form data
            claim_id = request.form['claim_id']

            # Connect to MySQL
            cur = mysql.connection.cursor()

            # Get data from database
            cur.execute('SELECT * FROM medical_claims WHERE claim_id = %s',(claim_id))
            claim_data = cur.fetchone()

            if not claim_data:
                return render_template('claim_classification.html', error="Claim ID not found.")

            # map database row to column names
            column_names = ['claim_id','insured_name','age','gender','admission_date','discharge_date','amount_billed','diagnosis', 'treatment', 'created_on','status']
            claim_df = pd.DataFrame([claim_data], columns=column_names)

            processed_data = preprocess_claim_data(claim_df)

            # predict using the trained model
            prediction = model.predict(processed_data)
            prediction_label = "Fraud" if prediction[0] == 1 else "Legit"

            # Update status in databse
            cur.execute('UPDATE medical_claims SET status = %s WHERE claim_id = %s',(prediction_label, claim_id))
            mysql.connection.commit()

            return render_template('claim_classification.html', success = True, claim_id=claim_id, prediction=prediction_label)

        except Exception as e:
            logger.error(f"Error during classification: {e}")
            return render_template('claim_classification.html', error="An error occurred during classification")

        finally:
            if 'cur' in locals():
                cur.close()
    return render_template('claim_classification.html')

def preprocess_claim_data(claim_df):

    # Copy to avoid SettingWithCopyWarning
    #processed_df = claim_df.copy()

    # Drop unnecessary columns
    features_to_drop = ['claim_id','gender','admission_date','discharge_date','diagnosis','created_on','status']
    claim_df.drop(columns=features_to_drop, inplace=True, errors='ignore')

    # Rename columns to match mode's expected feature names
    claim_df.rename(columns={
        'insured_name': 'Patient ID',
        'age': 'Age',
        'amount_billed': 'Amount Billed',
        'treatment': 'Treatment' 
    }, inplace=True)

    # Adding default values for column Treatment
    #if 'Treatment' not in claim_df.columns:
        #claim_df['Treatment'] = 'Unknown'

    #Apply label encoding to the diagnosis colunm
    le = LabelEncoder()
    claim_df['Patient ID'] = le.fit_transform(claim_df['Patient ID'])
    claim_df['Treatment'] = le.fit_transform(claim_df['Treatment'])


    # Ensure correct column order
    expected_columns = ['Patient ID','Age', 'Amount Billed','Treatment']
    claim_df = claim_df.reindex(columns=expected_columns)

    return claim_df

@app.route('/medical_claim', methods=['GET', 'POST'])
def medical_claim():
    if request.method == 'POST':
        try:
            # Log received form data
            logger.debug("Received form data: %s", request.form)
            
            # Get form data
            insured_name = request.form['insured_name']
            age = request.form['age']
            gender = request.form['gender']
            admission_date = request.form['admission_date']
            discharge_date = request.form['discharge_date']
            amount_billed = request.form['amount_billed']
            diagnosis = request.form['diagnosis']
            treatment = request.form['treatment']
            
            # Validate dates
            admission = datetime.strptime(admission_date, '%Y-%m-%d')
            discharge = datetime.strptime(discharge_date, '%Y-%m-%d')
            if discharge < admission:
                return render_template('medical_claim.html', 
                                    error="Discharge date cannot be before admission date")
            
            # Connect to MySQL
            cur = mysql.connection.cursor()
            
            # Insert into database
            cur.execute("""
                INSERT INTO medical_claims 
                (insured_name, age, gender, admission_date, discharge_date, amount_billed, diagnosis, treatment, created_on)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (insured_name, age, gender, admission_date, discharge_date, amount_billed, diagnosis, treatment))
            
            mysql.connection.commit()
            claim_id = cur.lastrowid
            cur.close()
            
            logger.info(f"New claim submitted with ID: {claim_id}")
            return render_template('medical_claim.html', success=True, claim_id=claim_id)
            
        except KeyError as e:
            logger.error(f"Missing form field: {e}")
            return render_template('medical_claim.html', 
                                  error=f"Missing required field: {e}")
        except ValueError as e:
            logger.error(f"Invalid data format: {e}")
            return render_template('medical_claim.html', 
                                  error="Invalid date or number format")
        except Exception as e:
            logger.error(f"Database error: {e}")
            return render_template('medical_claim.html', 
                                  error="Failed to submit claim. Please try again.")
    
    # GET request - show empty form
    return render_template('medical_claim.html')


@app.route('/logout', methods=['GET'])
def logout():
    if 'loggedin' in session:
        logout_user()
        session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)


