from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3
import os
from g4f.client import Client

app = Flask(__name__)
app.secret_key = "your_secret_key"
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize database
def init_db():
    with sqlite3.connect("database.db") as conn:
        conn.execute("""
             CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                mobile_no TEXT,
                email TEXT,
                location TEXT,
                role TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                image_path TEXT,
                quantity INTEGER,
                brand TEXT,
                price INTEGER,
                details TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS payments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                products TEXT,
                total_amount INTEGER,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        conn.commit()

# Seed admin credentials
def seed_admin():
    with sqlite3.connect("database.db") as conn:
        try:
            conn.execute("INSERT INTO users (name, username, password, role) VALUES (?, ?, ?, ?)",
                         ("Admin", "admin", "admin123", "admin"))
            conn.commit()
        except sqlite3.IntegrityError:
            pass

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/about')
def about():
    return render_template('about.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Connect to the database and fetch the user with the given username and password
        with sqlite3.connect("database.db") as conn:
            conn.row_factory = sqlite3.Row  # Fetch results as dictionary-like objects
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
            user = cur.fetchone()

        # Check if the user exists and match the credentials
        if user:
            # Storing the user information in session
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            session['name'] = user['name']
            session['mobile_no'] = user['mobile_no']
            session['email'] = user['email']
            session['location'] = user['location']
            
            # Redirect the user to the appropriate page (products page in your case)
            return redirect(url_for('products'))
        else:
            # If invalid credentials, flash an error message
            flash("Invalid username or password", "danger")

    # Render the login page if the request method is GET or if the login failed
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        password = request.form['password']
        mobile_no = request.form['mobile_no']
        email = request.form['email']
        location = request.form['location']

        with sqlite3.connect("database.db") as conn:
            try:
                conn.execute("INSERT INTO users (name, username, password, mobile_no, email, location, role) VALUES (?, ?, ?, ?, ?, ?, ?)",
                             (name, username, password, mobile_no, email, location, "user"))
                conn.commit()
                flash("Registration successful! Please log in.", "success")
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash("Username already exists.", "danger")

    return render_template('register.html')

# Add Product (Admin Only)
@app.route('/add_product', methods=['GET', 'POST'])
def add_product():
    if session.get('role') != 'admin':
        return redirect(url_for('login'))
    if request.method == 'POST':
        name = request.form['name']
        quantity = int(request.form['quantity'])
        brand = request.form['brand']
        price = int(request.form['price'])
        details = request.form['details']
        image = request.files['image']

        # Ensure the upload folder exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(image_path)

        with sqlite3.connect("database.db") as conn:
            conn.execute("INSERT INTO products (name, image_path, quantity, brand, price, details) VALUES (?, ?, ?, ?, ?, ?)",
                         (name, image_path, quantity, brand, price, details))
            conn.commit()
        flash("Product added successfully!", "success")
    return render_template('add_product.html')

@app.route('/profile')
def profile():
    if 'username' not in session:  # Ensure correct session key
        flash("Please log in first.", "danger")
        return redirect(url_for('login'))

    username = session['username']

    with sqlite3.connect("database.db") as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT name, username, mobile_no, email, location FROM users WHERE username = ?", (username,))
        user = cur.fetchone()

    if user:
        return render_template('profile.html', user=user)
    else:
        flash("User not found.", "danger")
        return redirect(url_for('login'))

# View Products (User Only)
@app.route('/products', methods=['GET', 'POST'])
def products():
    if session.get('role') != 'user':
        return redirect(url_for('login'))
    selected_products = []
    total_cost = 0
    if request.method == 'POST':
        budget = int(request.form['budget'])
        with sqlite3.connect("database.db") as conn:
            products = conn.execute("SELECT * FROM products ORDER BY price ASC").fetchall()
        for product in products:
            if total_cost + product[5] <= budget:  # product[5] is price
                selected_products.append(product)
                total_cost += product[5]
        # Store payment details in the database
        with sqlite3.connect("database.db") as conn:
            conn.execute("INSERT INTO payments (user_id, products, total_amount) VALUES (?, ?, ?)",
                         (session['user_id'], ", ".join([p[1] for p in selected_products]), total_cost))
            conn.commit()
        flash("Payment successful!", "success")
    with sqlite3.connect("database.db") as conn:
        products = conn.execute("SELECT * FROM products").fetchall()
    return render_template('products.html', products=products, selected_products=selected_products, total=total_cost)

@app.route('/buy_product/<int:product_id>', methods=['POST'])
def buy_product(product_id):
    if session.get('role') != 'user':
        return redirect(url_for('login'))

    # Fetch the selected product from the database
    with sqlite3.connect("database.db") as conn:
        conn.row_factory = sqlite3.Row  # Set row_factory to sqlite3.Row
        product = conn.execute("SELECT * FROM products WHERE id = ?", (product_id,)).fetchone()

    if product is None:
        flash("Product not found.", "danger")
        return redirect(url_for('products'))

    # Get the quantity the user wants to buy from the form
    quantity_to_buy = int(request.form['quantity'])

    # Check if the quantity is available
    if product['quantity'] < quantity_to_buy:
        flash(f"Not enough stock for {product['name']}. Available quantity is {product['quantity']}.", "danger")
        return redirect(url_for('products'))

    # Store the selected product and quantity in the session to display on the payment page
    if 'selected_products' not in session:
        session['selected_products'] = []
    # Append the product and quantity to the session
    session['selected_products'].append({
        'id': product['id'],
        'name': product['name'],
        'quantity': quantity_to_buy,
        'price': product['price'],
        'total_price': product['price'] * quantity_to_buy
    })
    session.modified = True

    # Redirect to the payment page
    return redirect(url_for('payment'))


@app.route('/payment', methods=['GET', 'POST'])
def payment():
    if session.get('role') != 'user':
        return redirect(url_for('login'))

    selected_products = session.get('selected_products', [])
    total_cost = sum(product['price'] * product['quantity'] for product in selected_products)

    if request.method == 'POST':
        # Store payment details in the database
        with sqlite3.connect("database.db") as conn:
            products_names = ", ".join([p['name'] for p in selected_products])
            conn.execute("INSERT INTO payments (user_id, products, total_amount) VALUES (?, ?, ?)",
                         (session['user_id'], products_names, total_cost))

            # Update the product quantities after purchase
            for product in selected_products:
                conn.execute("UPDATE products SET quantity = quantity - ? WHERE id = ?",
                             (product['quantity'], product['id']))
            conn.commit()

        # Clear the session after payment
        session.pop('selected_products', None)
        flash("Payment successful! Thank you for your purchase.", "success")
        return redirect(url_for('products'))

    return render_template('payment.html', selected_products=selected_products, total_cost=total_cost)

@app.route('/view_stack', methods=['GET'])
def view_stack():
    if session.get('role') != 'admin':  # Ensure that only admins can view the stack
        flash("Access denied. You need admin privileges.", "danger")
        return redirect(url_for('login'))

    # Fetch all products from the database
    with sqlite3.connect("database.db") as conn:
        conn.row_factory = sqlite3.Row  # To get rows as dictionary-like objects
        products = conn.execute("SELECT * FROM products").fetchall()

    # Return the view with the products
    return render_template('view_stack.html', products=products)

import matplotlib.pyplot as plt
import io
import base64

# Route to display stock graph
@app.route('/stock_graph')
def stock_graph():
    with sqlite3.connect("database.db") as conn:
        conn.row_factory = sqlite3.Row
        products = conn.execute("SELECT name, quantity FROM products").fetchall()

    product_names = [product['name'] for product in products]
    quantities = [product['quantity'] for product in products]

    fig, ax = plt.subplots()
    ax.bar(product_names, quantities)
    ax.set_xlabel('Products')
    ax.set_ylabel('Stock Quantity')
    ax.set_title('Product Stock Levels')

    # Save it to a string buffer and encode it for use in HTML
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    return render_template('stock_graph.html', img_data=img_base64)


@app.route('/adminlogin', methods=['GET', 'POST'])
def adminlogin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Connect to the database to check the user's credentials
        with sqlite3.connect("database.db") as conn:
            conn.row_factory = sqlite3.Row  # Fetch results as dictionary-like objects
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
            user = cur.fetchone()

        # Check if the user exists and if their role is 'admin'
        if user and user['role'] == 'admin':
            # Store the admin information in the session
            session['username'] = user['username']
            session['role'] = user['role']
            session['user_id'] = user['id']  # Store user ID as well for tracking
            flash('Admin login successful!', 'success')
            return redirect(url_for('add_product'))
        else:
            flash('Invalid admin credentials or insufficient permissions.', 'danger')

    return render_template('adminlogin.html')


# Route to display user role distribution graph
@app.route('/user_roles_graph')
def user_roles_graph():
    with sqlite3.connect("database.db") as conn:
        conn.row_factory = sqlite3.Row
        roles_count = conn.execute("""
            SELECT role, COUNT(*) as count FROM users GROUP BY role
        """).fetchall()

    roles = [role['role'] for role in roles_count]
    counts = [role['count'] for role in roles_count]

    fig, ax = plt.subplots()
    ax.bar(roles, counts)
    ax.set_xlabel('User Roles')
    ax.set_ylabel('Number of Users')
    ax.set_title('User Role Distribution')

    # Save it to a string buffer and encode it for use in HTML
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    return render_template('user_roles_graph.html', img_data=img_base64)


# Chatbot route
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    client = Client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_message}],
        web_search=False
    )
    return jsonify({"response": response.choices[0].message.content})

@app.route('/view_users', methods=['GET'])
def view_users():
    if session.get('role') != 'admin':  # Ensure that only admins can view the users
        flash("Access denied. You need admin privileges.", "danger")
        return redirect(url_for('login'))

    # Fetch all users from the database
    with sqlite3.connect("database.db") as conn:
        conn.row_factory = sqlite3.Row  # To get rows as dictionary-like objects
        users = conn.execute("SELECT * FROM users").fetchall()

    # Return the view with the user data
    return render_template('view_users.html', users=users)

# Logout route
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import warnings
import io
import base64

# Suppress warnings
warnings.filterwarnings("ignore")


# Load and prepare data
df = pd.read_csv("medicine_sales.csv", parse_dates=["date"])
products = df['product'].unique()

@app.route("/demand", methods=["GET", "POST"])
def dindex():
    selected_product = None
    forecast_df = None
    plot_url = None

    if request.method == "POST":
        selected_product = request.form.get("product_name").strip()

        product_df = df[df['product'] == selected_product]
        if len(product_df) < 10:
            forecast_df = [{"Date": "Error", "Forecasted Quantity": "Not enough data"}]
        else:
            daily_sales = product_df.groupby('date')['quantity'].sum().asfreq('D').fillna(0)

            # Train ARIMA model
            stepwise_model = auto_arima(daily_sales, seasonal=False, suppress_warnings=True)
            model = ARIMA(daily_sales, order=stepwise_model.order)
            model_fit = model.fit()

            # Forecast next 7 days
            n_days = 7
            forecast = model_fit.forecast(steps=n_days)
            forecast_index = pd.date_range(start=daily_sales.index[-1] + pd.Timedelta(days=1), periods=n_days, freq='D')

            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                "Date": forecast_index.strftime('%Y-%m-%d'),
                "Forecasted Quantity": np.round(forecast).astype(int)
            }).to_dict(orient="records")

            # Plot
            plt.figure(figsize=(10, 4))
            plt.plot(daily_sales.index, daily_sales.values, label="Historical Demand")
            plt.plot(forecast_index, forecast, label="7-Day Forecast", color="red")
            plt.title(f"{selected_product} - 7-Day Demand Forecast")
            plt.xlabel("Date")
            plt.ylabel("Quantity")
            plt.legend()
            plt.tight_layout()

            # Convert plot to base64
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()

            plt.close()

    return render_template("dindex.html",
                           products=products,
                           selected_product=selected_product,
                           forecast_df=forecast_df,
                           plot_url=plot_url)

if __name__ == '__main__':
    init_db()
    seed_admin()
    app.run(debug=True)
