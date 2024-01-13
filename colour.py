import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from flask import Flask, render_template
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import base64
import threading
import time

app = Flask(__name__)

# Variables for communication between threads
color_lock = threading.Lock()
current_color = "#000000"

# Function to update color based on sound


def update_color():
    global current_color

    while True:
        with color_lock:
            # Replace this with your actual sound analysis and color mapping logic
            intensity = 0.5  # Simulated intensity
            hue = intensity * 360  # Mapping intensity to hue (0-360)
            current_color = plt.cm.hsv(hue / 360).to_rgba((0, 0, 0))[:3]

        time.sleep(0.1)  # Adjust as needed


# Start the color update thread
color_thread = threading.Thread(target=update_color)
color_thread.daemon = True
color_thread.start()


@app.route("/")
def index():
    return render_template("index.html", color=current_color)


@app.route("/plot.png")
def plot_png():
    with color_lock:
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.set_facecolor(current_color)

        canvas = FigureCanvas(fig)
        buf = io.BytesIO()
        canvas.print_png(buf)
        data = base64.b64encode(buf.getbuffer()).decode("utf-8")

    return f"data:image/png;base64,{data}"


if __name__ == "__main__":
    app.run(debug=True)


# Function to update the color based on audio features
# def update_color(frame,pos_scale):
#     if pos_scale > 0.5:
#         #if it is a positive song, change the colour to bright
