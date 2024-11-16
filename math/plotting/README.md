# Plotting Scripts in Python

This project consists of several Python scripts that demonstrate various types of data visualizations using `matplotlib` and `numpy`. Each script focuses on a specific kind of plot or visualization, showcasing the power of Python for data representation and analysis.

## Installation

Ensure you have Python 3 installed. Install the required dependencies using:

```bash
pip install -r requirements.txt
```

### Dependencies:

- matplotlib==3.8.3
- Pillow==10.3.0

For Ubuntu users, ensure python3-tk is installed for GUI support:

```bash
sudo apt-get install python3-tk
```

## Scripts Overview
1. `0-line.py`
Plots a simple line graph representing representing 
𝑦
=
𝑥
3
y=x 
3
 .

- X-axis range: 0 to 10
- Line style: Solid red line
- Usage: Run the script to visualize the plot.
2. `1-scatter.py`
Creates a scatter plot showing the relationship between men's height and weight.

- Data: Generated from a multivariate normal distribution.
- X-axis: Height (inches)
- Y-axis: Weight (lbs)
- Usage: Run the script to view the scatter plot.
3. `2-change_scale.py`
Visualizes exponential decay (C-14 decay) with a logarithmic y-axis.

- X-axis: Time (years)
- Y-axis: Fraction remaining (logarithmic scale)
- Usage: Run the script to see the decay process.
4. `3-two.py`
Plots two radioactive elements' exponential decay (C-14 and Ra-226).

- X-axis: Time (years)
- Y-axis: Fraction remaining
- Features: Two lines (C-14 dashed red, Ra-226 solid green) with a legend.
- Usage: Run the script to visualize the comparison.
5. `4-frequency.py`
Generates a histogram of student grades for "Project A".

- X-axis: Grades
- Y-axis: Number of students
- Bins: 10-grade intervals
- Usage: Run the script to display the histogram.
6. `5-all_in_one.py`
Combines multiple plots into a single figure:

- Line plot of 
𝑦
=
𝑥
3
y=x 
3
 
- Scatter plot of height vs. weight
- Exponential decay of C-14
- Comparison of decay (C-14 vs. Ra-226)
- Histogram of student grades
- Usage: Run the script to view all plots in a single window.
7. `6-bars.py`
- Creates a stacked bar graph showing fruit distribution among three individuals.

- People: Farrah, Fred, Felicia
- Fruits: Apples, Bananas, Oranges, Peaches
- Usage: Run the script to view the bar chart.
8. `100-gradient.py`
Plots a scatter plot with elevation data visualized using a color gradient.

- X/Y-axis: Coordinates
- Colorbar: Elevation
- Usage: Run the script to explore elevation changes.
9. 101-pca.py
Performs PCA on a dataset and visualizes the first three principal components in a 3D scatter plot.

- Data: data.npy and labels.npy
- Axes: Principal components (U1, U2, U3)
- Usage: Ensure data.npy and labels.npy are in the working directory, then run the script.

## Usage
Each script is standalone and can be executed using:

```bash
./<script_name>.py
```

Ensure the script has execute permissions:

```bash
chmod +x <script_name>.py
```

Or run directly via Python:

```bash
python3 <script_name>.py
```

Project Structure
```plaintext
├── 0-line.py
├── 1-scatter.py
├── 2-change_scale.py
├── 3-two.py
├── 4-frequency.py
├── 5-all_in_one.py
├── 6-bars.py
├── 100-gradient.py
├── 101-pca.py
├── data.npy
├── labels.npy
├── pca.npz
├── requirements.txt
├── README.md
```

## Author
This project is part of the Holberton School Machine Learning Curriculum. Each script demonstrates core concepts of data visualization in Python.

Happy plotting!
