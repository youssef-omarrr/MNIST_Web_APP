# 🧠 MNIST Drawing Classifier (Flask Web App)

This is a simple and interactive web application where you can **draw a digit (0–9)** using your mouse, and a **PyTorch-based neural network** will predict what digit you drew.

The model was trained on the **MNIST dataset** as well as additional **custom handwritten digit data** to improve real-world accuracy and generalization.
![alt text](img/empty.png)
![alt text](img/pred.png)
### NEW
![alt text](img/kepad.png)
## 🚀 Features

- Canvas for drawing digits
- Live prediction using a trained PyTorch model
- Visual display of class probabilities
- Backend powered by Flask
- Real-time input preview and prediction feedback
- [NEW] added feedback system to learn different handwritings.

---

## 📦 Installation & Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/mnist-drawing-classifier.git
cd mnist-drawing-classifier
````

### 2. Set up a Python virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

* **Windows:**

  ```bash
  venv\Scripts\activate
  ```

* **macOS/Linux:**

  ```bash
  source venv/bin/activate
  ```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

> ✅ Note: This project uses the **CPU-only version** of PyTorch to reduce system requirements.

### 5. Run the Flask app

```bash
python app.py
```

Now, open your browser and go to:
📍 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🧠 Model Training

* The digit classification model is built with **PyTorch**.
* It was trained on:

  * The standard **MNIST dataset** (`60,000+` training examples)
  * Additional **custom handwritten digit data** to improve robustness and performance on real user input.

You can modify or retrain the model in the `train_model.py` (if provided) with your own dataset.

---

## 📁 Project Structure

```
├── static/
│   ├── css/                    # Stylesheets
│   ├── js/                     # JavaScript files
│   └── received_input.png      # Image from canvas sent to the server
├── templates/
│   └── index.html              # Main UI template
├── app.py                      # Flask backend and model inference
├── model.pth                   # Trained PyTorch model
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 📌 Requirements

* Python 3.7+
* Flask
* Torch (CPU-only)
* Torchvision
* Pillow

---

## 🛠️ To Do / Future Work

* Deploy a flask hosting server

---