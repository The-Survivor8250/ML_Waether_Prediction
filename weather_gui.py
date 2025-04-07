import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

# Load dataset from file
def load_data():
    global data, X_train, X_test, y_train, y_test, model, y_pred, accuracy
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not file_path:
        return None
    try:
        df = pd.read_csv(file_path)
        if not {'precipitation', 'temp_max', 'temp_min', 'wind', 'weather'}.issubset(df.columns):
            messagebox.showerror("Error", "Dataset must contain precipitation, temp_max, temp_min, wind, and weather columns.")
            return None
        
        # Fill zero values with the column mean
        for col in ['precipitation', 'temp_max', 'temp_min', 'wind']:
            df[col] = df[col].replace(0, np.nan)
            df[col].fillna(df[col].mean(), inplace=True)
        
        data = df
        X = data[['precipitation', 'temp_max', 'temp_min', 'wind']]
        y = data['weather']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load dataset: {e}")

def predict_weather():
    try:
        precipitation = float(precip_entry.get())
        temp_max = float(temp_max_entry.get())
        temp_min = float(temp_min_entry.get())
        wind = float(wind_entry.get())
        
        prediction = model.predict([[precipitation, temp_max, temp_min, wind]])[0]
        messagebox.showinfo("Prediction", f"Predicted Weather Condition: {prediction}")
        show_prediction_chart(prediction)
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values.")

def show_prediction_chart(prediction):
    clear_graph()
    fig, ax = plt.subplots()
    categories = ['drizzle', 'rain', 'sun', 'snow', 'fog']
    values = [1 if cat == prediction else 0 for cat in categories]
    ax.bar(categories, values, color=['blue', 'green', 'yellow', 'white', 'gray'])
    ax.set_xlabel('Weather Condition')
    ax.set_ylabel('Prediction Confidence')
    ax.set_title(f'Predicted Weather: {prediction}')
    
    embed_graph(fig)

def show_accuracy():
    clear_graph()
    fig, ax = plt.subplots()
    categories = ['drizzle', 'rain', 'sun', 'snow', 'fog']
    counts = [list(y_pred).count(cat) for cat in categories]
    ax.bar(categories, counts, color=['blue', 'green', 'yellow', 'red', 'gray'])
    ax.set_xlabel('Weather Condition')
    ax.set_ylabel('Predicted Count')
    ax.set_title(f'Model Accuracy: {accuracy:.2%}')
    
    embed_graph(fig)

def embed_graph(fig):
    global canvas
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

def reset_all():
    precip_entry.delete(0, tk.END)
    temp_max_entry.delete(0, tk.END)
    temp_min_entry.delete(0, tk.END)
    wind_entry.delete(0, tk.END)
    clear_graph()

def clear_graph():
    global canvas
    for widget in frame.winfo_children():
        widget.destroy()

root = tk.Tk()
root.title("Weather Prediction")
root.geometry("600x700")
root.configure(bg="#f0f0f0")

# Styling
button_style = {"font": ("Arial", 12), "bg": "#4CAF50", "fg": "white", "bd": 3, "relief": "raised"}
label_style = {"font": ("Arial", 12), "bg": "#f0f0f0"}
entry_style = {"font": ("Arial", 12), "bd": 2, "relief": "solid"}

# GUI Elements
tk.Button(root, text="Load Dataset", command=load_data, **button_style).pack(pady=5)

tk.Label(root, text="Precipitation (mm):", **label_style).pack()
precip_entry = tk.Entry(root, **entry_style)
precip_entry.pack(pady=2)

tk.Label(root, text="Max Temperature (°C):", **label_style).pack()
temp_max_entry = tk.Entry(root, **entry_style)
temp_max_entry.pack(pady=2)

tk.Label(root, text="Min Temperature (°C):", **label_style).pack()
temp_min_entry = tk.Entry(root, **entry_style)
temp_min_entry.pack(pady=2)

tk.Label(root, text="Wind Speed (km/h):", **label_style).pack()
wind_entry = tk.Entry(root, **entry_style)
wind_entry.pack(pady=2)

# Buttons
tk.Button(root, text="Predict", command=predict_weather, **button_style).pack(pady=5)
tk.Button(root, text="Show Accuracy", command=show_accuracy, **button_style).pack(pady=5)
tk.Button(root, text="Reset", command=reset_all, **button_style).pack(pady=5)

# Frame for Graph Display
frame = tk.Frame(root, bg="#ffffff", bd=2, relief="sunken")
frame.pack(pady=10, fill="both", expand=True)

root.mainloop()
