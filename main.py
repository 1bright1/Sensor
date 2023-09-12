import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, entropy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay, roc_auc_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
import joblib
from sklearn.pipeline import Pipeline
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox



# Read CSV files
try:
    b_walking = pd.read_csv('Walkingdata_Bright.csv')
    h_walking = pd.read_csv('Walkingdata_Haley.csv')
    j_walking = pd.read_csv('Walkingdata_Johnny.csv')

    b_jump = pd.read_csv('Jumpingdata_Bright.csv')
    h_jump = pd.read_csv('Jumpingdata_Haley.csv')
    j_jump = pd.read_csv('Jumpingdata_Johnny.csv')
except FileNotFoundError:
    print("CSV file not found. Please ensure that the file path is correct and try again.")
    exit()

# Check for duplicates and ensure data consistency
try:
    b_data = pd.concat([b_walking, b_jump])
    b_data.drop_duplicates(inplace=True)

    h_data = pd.concat([h_walking, h_jump])
    h_data.drop_duplicates(inplace=True)

    j_data = pd.concat([j_walking, j_jump])
    j_data.drop_duplicates(inplace=True)
except ValueError:
    print("Error in concatenating data. Please check that the data is consistent and try again.")
    exit()

# Concatenate data
try:
    walking_data = pd.concat([b_walking, h_walking, j_walking], ignore_index=True)
    walking_data['type'] = 0
    walking_data.to_csv('walking_data.csv', index=False)
    jumping_data = pd.concat([b_jump, h_jump, j_jump], ignore_index=True)
    jumping_data['type'] = 1
    jumping_data.to_csv('jumping_data.csv', index=False)
    dataset = pd.concat([walking_data, jumping_data], ignore_index=True)
    dataset.to_csv('dataset.csv', index=False)  # save processed data
except ValueError:
    print("Error in concatenating data. Please check that the data is consistent and try again.")
    exit()

# Applying noise removal
for col in dataset.columns:
    dataset[col] = dataset[col].rolling(window=60).mean().fillna(method='bfill').fillna(method='ffill')


# Segment data
cols_of_interest = ['Acceleration x (m/s^2)', 'Acceleration x (m/s^2)', 'Acceleration x (m/s^2)']
window_size = 500
segments = []
for col in cols_of_interest:
    signal = dataset[col].to_numpy()
    num_windows = math.floor(len(signal) / window_size)
    for i in range(num_windows):
        start = i * window_size
        end = (i + 1) * window_size
        segment = signal[start:end]
        features = [
            np.mean(segment),
            np.median(segment),
            np.var(segment),
            np.std(segment),
            np.max(segment),
            np.min(segment),
            np.max(segment) - np.min(segment),
            skew(segment),
            entropy(segment),
            kurtosis(segment)
        ]
        segments.append(np.concatenate((features, [dataset.iloc[start, -1]])))

        # Print the extracted features for the current segment
        #print(f"Segment {i+1} features: {features}")

# Shuffle segments and split into train and test sets
segments = np.array(segments)
np.random.shuffle(segments)
X = segments[:, :-1]
y = segments[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


# Save train and test sets
# store each member's data and the test and train datas
try:
    with h5py.File('combinedData.hdf5', 'w') as f:
        data1 = f.create_group('Bright')
        data1.create_dataset('Walk/Jump', data=b_data.to_numpy())

        data2 = f.create_group('Haley')
        data2.create_dataset('Walk/Jump', data=h_data.to_numpy())

        data3 = f.create_group('John')
        data3.create_dataset('Walk/Jump', data=j_data.to_numpy())

        train_data = f.create_group('Train Data')
        train_data.create_dataset('Xtrain', data=X_train)
        train_data.create_dataset('ytest', data=y_train)

        test_data = f.create_group('Test Data')
        test_data.create_dataset('Xtest', data=X_test)
        test_data.create_dataset('ytest', data=y_test)
except ValueError:
    print("Error in saving data. Please check that the data is properly formatted and try again.")
    exit()
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# Read CSV file
walk = pd.read_csv('walking_data.csv')
jump = pd.read_csv('jumping_data.csv')

# # Plot the data
# plt.plot(walk['Time (s)'], walk['Acceleration x (m/s^2)'], color='blue', label='X')
# plt.plot(walk['Time (s)'], walk['Acceleration y (m/s^2)'], color='green', label='Y')
# plt.plot(walk['Time (s)'], walk['Acceleration z (m/s^2)'], color='red', label='Z')
# plt.title('Walking Data')
# plt.xlabel('Time (s)')
# plt.ylabel('Acceleration (m/s^2)')
# plt.legend()
# plt.show()
#
# # Plot the data
# plt.plot(jump['Time (s)'], jump['Acceleration x (m/s^2)'], color='blue', label='X')
# plt.plot(jump['Time (s)'], jump['Acceleration y (m/s^2)'], color='green', label='Y')
# plt.plot(jump['Time (s)'], jump['Acceleration z (m/s^2)'], color='red', label='Z')
# plt.title('Jumping Data')
# plt.xlabel('Time (s)')
# plt.ylabel('Acceleration (m/s^2)')
# plt.legend()
# plt.show()
# # plot acceleration vs time for each axis
# fig, axs = plt.subplots(3, 1, figsize=(10, 8))
# axs[0].plot(dataset['Time (s)'], dataset['Acceleration x (m/s^2)'], label='ax')
# axs[1].plot(dataset['Time (s)'], dataset['Acceleration y (m/s^2)'], label='ay')
# axs[2].plot(dataset['Time (s)'], dataset['Acceleration z (m/s^2)'], label='az')
#
# # set plot labels and title
# fig.suptitle('Acceleration vs Time')
# axs[0].set_ylabel('Acceleration (m/s^2)')
# axs[1].set_ylabel('Acceleration (m/s^2)')
# axs[2].set_ylabel('Acceleration (m/s^2)')
# axs[2].set_xlabel('Time (s)')
#
# # add legend
# for ax in axs:
#     ax.legend()
#
# #show plot
# plt.show()
# #######################################################################################################################
# fig11, ax11 = plt.subplots(figsize=(10, 10), layout="constrained")
# plt.title('Full Set')
# plt.xlabel('Time (s)', fontsize=15)
# plt.ylabel('Acceleration (m/s^2)', fontsize=15)
# ax11.scatter(dataset['Time (s)'], dataset['Absolute acceleration (m/s^2)'], label='Combined Data', color='red')
# plt.legend(loc="upper left")
# #########################
#
# thedatas = pd.concat([b_walking, h_walking, j_walking, b_jump, h_jump, j_jump]) #for plotting
# thedatas.to_csv('Thedatas.csv', index=False)
# start = 3000
# end = 3500
# smallStart = 3000
# smallEnd = 3300
# # print(combined_data)
# time = thedatas.iloc[start:end, 0]
# timeSmall = thedatas.iloc[smallStart: smallEnd, 0]
# fig2, ax2 = plt.subplots(figsize=(10, 10), layout="constrained")
# plt.title('Acceleration vs Time')
# plt.xlabel('Time (s)', fontsize=15)
# plt.ylabel('XYZ Acceleration (m/s^2)', fontsize=15)
# ax2.plot(time, b_walking.iloc[start: end, 1], label='X', color='red')
# ax2.plot(time, h_walking.iloc[start: end, 2], label='Y', color='blue')
# ax2.plot(time, j_walking.iloc[start: end, 3], label='Z', color='green')
# plt.show()
# ###############################plot for bright
# fig1, ax1 = plt.subplots(figsize=(10, 10), layout="constrained")
# plt.title('Brights data')
# plt.xlabel('Time (s)', fontsize=15)
# plt.ylabel('XYZ Acceleration (m/s^2)', fontsize=15)
# ax1.plot(time, b_walking.iloc[start: end, 1], label='X', color='red')
# ax1.plot(time, b_walking.iloc[start: end, 2], label='Y', color='blue')
# ax1.plot(time, b_walking.iloc[start: end, 3], label='Z', color='green')
#
# ax1.plot(time, b_jump.iloc[start: end, 1], label='Jump X', color='red', linestyle='dashed')
# ax1.plot(time, b_jump.iloc[start: end, 2], label='Jump Y', color='blue', linestyle='dashed')
# ax1.plot(time, b_jump.iloc[start: end, 3], label='Jump Z', color='green', linestyle='dashed')
# plt.legend(loc="upper left")
#
# plt.show()
# #####################################
# dataset.loc[dataset['type'] == 0, 'type'] = 0 #reading the dataset and binarizing the labels
# dataset.loc[dataset['type'] == 1, 'type'] = 1
# data = dataset.iloc[:, 1:-1]
# labels = dataset.iloc[:,-1]
# # assigning 10% of data to test set
# X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size= 0.1, shuffle = True, random_state=0)
####################################################################################################################################################
# Replace NaN and infinite values with 0
X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
# X_train_df = pd.DataFrame(X_train, columns=[features[0], features[1], features[2], features[3], features[4], features[5], features[6], features[7], features[8], features[9]])
# X_test_df = pd.DataFrame(X_test, columns=[features[0], features[1], features[2], features[3], features[4], features[5], features[6], features[7], features[8], features[9]])
#X_test_df.to_csv('x_test.csv', index = False)
# X_train_acc = X_train_df[['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)']]
# X_test_acc = X_test_df[['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)']]
# ...

scaler = StandardScaler() #define a standard scaler to normalize inputs
l_reg = LogisticRegression(max_iter=10000) #defining the classifier and the pipeline
clf = make_pipeline(scaler, l_reg)
clf.fit(X_train, y_train) #training
y_pred = clf.predict(X_test) #obtaining the predictions and the probability
y_clf_prob = clf.predict_proba(X_test)
# print('y_pred is: ', y_pred)
# print('y_clf_prob is: ', y_clf_prob)
# Assuming `clf` is your trained model
joblib.dump(clf, 'my_model.pkl')
acc = accuracy_score(y_test, y_pred) #obtaining the classification accuracy
print('accuracy is: ', acc)
recall = recall_score(y_test, y_pred) #obtaining the classification recall
# print('recall score is: ', recall)
#plotting the confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()
#plotting the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_clf_prob[:, 1], pos_label=clf.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.show()
#calculating the AUC
auc = roc_auc_score(y_test, y_clf_prob[:, 1])
# print('The AUC is: ', auc)

######################################
######################################
# # Read input csv file
# df = pd.read_csv("jump_motion.csv")
#
# # Calculate the mean acceleration in each axis
# x_mean = df['Acceleration x (m/s^2)'].mean()
# y_mean = df['Acceleration y (m/s^2)'].mean()
# z_mean = df['Acceleration z (m/s^2)'].mean()
#
# # Calculate the standard deviation of acceleration in each axis
# x_std = df['Acceleration x (m/s^2)'].std()
# y_std = df['Acceleration y (m/s^2)'].std()
# z_std = df['Acceleration z (m/s^2)'].std()
#
# # Determine if the motion is walking or jumping based on the mean and standard deviation of acceleration in each axis
# if x_mean > features[0] and y_mean > features[0] and z_mean > features[0] and x_std < features[3] and y_std < features[3] and z_std < features[3]:
#     print("The motion is walking")
# elif x_mean > features[0] and y_mean > features[0] and z_mean > features[0] and x_std > features[3] and y_std > features[3] and z_std > features[3]:
#     print("The motion is jumping")
# else:
#     print("The motion is unknown")


# # Create a function to handle the "Browse" button click event
# def browse_file():
#     # Show the file dialog box
#     file_path = filedialog.askopenfilename()
#
#     # Set the text of the file path label to the selected file path
#     file_path_label.config(text=file_path)
#
#
# # Create a function to handle the "Check motion" button click event
# def check_motion():
#     # Read the input CSV file
#     df = pd.read_csv(file_path_label.cget('text'))
#
#     # Calculate the mean acceleration in each axis
#     x_mean = df['Acceleration x (m/s^2)'].mean()
#     y_mean = df['Acceleration y (m/s^2)'].mean()
#     z_mean = df['Acceleration z (m/s^2)'].mean()
#
#     # Calculate the standard deviation of acceleration in each axis
#     x_std = df['Acceleration x (m/s^2)'].std()
#     y_std = df['Acceleration y (m/s^2)'].std()
#     z_std = df['Acceleration z (m/s^2)'].std()
#
#     # Determine if the motion is walking or jumping based on the mean and standard deviation of acceleration in each axis
#     if x_mean > features[0] and y_mean > features[0] and z_mean > features[0] and x_std < features[3] and y_std < \
#             features[3] and z_std < features[3]:
#         result_label.config(text="The motion is walking")
#     elif x_mean > features[0] and y_mean > features[0] and z_mean > features[0] and x_std > features[3] and y_std > \
#             features[3] and z_std > features[3]:
#         result_label.config(text="The motion is jumping")
#     else:
#         result_label.config(text="The motion is unknown")
#
#
# # Create the GUI
# root = tk.Tk()
# root.title("Motion Checker")
#
# # Create the "Browse" button and label
# browse_button = tk.Button(root, text="Browse", command=browse_file)
# browse_button.pack()
# file_path_label = tk.Label(root, text="")
# file_path_label.pack()
#
# # Create the "Check motion" button and label
# check_button = tk.Button(root, text="Check motion", command=check_motion)
# check_button.pack()
# result_label = tk.Label(root, text="")
# result_label.pack()
#
# # Start the GUI event loop
# root.mainloop()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# model = LogisticRegression(max_iter=10000) # Replace with your trained model
# scaler = StandardScaler() # Replace with your scaler
# clf = make_pipeline(scaler, model)
# clf.fit(X_train, y_train) #training
#
# # Define a function to preprocess the data
# # Define a function to preprocess the data
# def preprocess_data(data):
#     # Apply noise removal
#     for col in data.columns:
#         data[col] = data[col].rolling(window=60).mean().fillna(method='bfill').fillna(method='ffill')
#
#     # Segment data
#     window_size = 500
#     segments = []
#     for col in data.columns[:-1]:
#         signal = data[col].to_numpy()
#         num_windows = math.floor(len(signal) / window_size)
#         for i in range(num_windows):
#             start = i * window_size
#             end = (i + 1) * window_size
#             segment = signal[start:end]
#
#             # Check for and remove any infinite or too large values
#             if not np.isfinite(segment).all() or np.abs(segment).max() > 1e9:
#                 continue
#
#             features = [
#                 np.mean(segment),
#                 np.median(segment),
#                 np.var(segment),
#                 np.std(segment),
#                 np.max(segment),
#                 np.min(segment),
#                 np.max(segment) - np.min(segment),
#                 skew(segment),
#                 entropy(segment),
#                 kurtosis(segment)
#             ]
#             segments.append(np.concatenate((features, [data.iloc[start, -1]])))
#
#     # Standardize the data
#     segments = np.array(segments)
#     X = segments[:, :-1]
#     X = scaler.transform(X)
#
#     return X
#
# # Define a function to predict the motion
# def predict_motion(filepath):
#     try:
#         # Load the CSV file
#         data = pd.read_csv(filepath)
#
#         # Preprocess the data
#         X = preprocess_data(data)
#
#         # Make a prediction
#         y_pred = model.predict(X)
#
#         # Return the prediction
#         if np.mean(y_pred) < 0.5:
#             return "Walking"
#         else:
#             return "Jumping"
#
#     except Exception as e:
#         messagebox.showerror("Error", str(e))
#
# # Define a function to handle the button click event
# def browse_file():
#     filepath = filedialog.askopenfilename(initialdir="/", title="Select a File", filetypes=[("CSV files", "*.csv")])
#     if filepath:
#         motion = predict_motion(filepath)
#         messagebox.showinfo("Motion Prediction", f"The motion is {motion}.")
#
# # Create a GUI window
# window = tk.Tk()
# window.title("Motion Prediction")
# window.geometry("300x100")
#
# # Create a button to browse for a file
# button = tk.Button(text="Browse", command=browse_file)
# button.pack(pady=10)
#
# # Run the GUI loop
# window.mainloop()
#########################################################################################
# Load the CSV file
# df = pd.read_csv('jump_motion.csv')
#
# # Segment data
# window_size = 500
# segments = []
# for col in df.columns[:-1]:
#     signal = dataset[col].to_numpy()
#     num_windows = math.floor(len(signal) / window_size)
#     for i in range(num_windows):
#         start = i * window_size
#         end = (i + 1) * window_size
#         segment = signal[start:end]
#         feature = [
#             np.mean(segment),
#             np.median(segment),
#             np.var(segment),
#             np.std(segment),
#             np.max(segment),
#             np.min(segment),
#             np.max(segment) - np.min(segment),
#             skew(segment),
#             entropy(segment),
#             kurtosis(segment)
#         ]
#         segments.append(np.concatenate((feature, [dataset.iloc[start, -1]])))
#
#         # Print the extracted features for the current segment
#         #print(f"Segment {i+1} features: {features}")
#
# # Shuffle segments and split into train and test sets
# segments = np.array(segments)
# np.random.shuffle(segments)
# X_acc = segments[:, :-1]
# y_acc = segments[:, -1]
# X_train_acc, X_test_acc, y_train_acc, y_test_acc = train_test_split(X_acc, y_acc, test_size=0.1)
# X_train_acc = np.nan_to_num(X_train_acc, nan=0.0, posinf=1e6, neginf=-1e6)
# X_test_acc = np.nan_to_num(X_test_acc, nan=0.0, posinf=1e6, neginf=-1e6)
# # Extract the acceleration data
# #X = df[['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)']]
#
# # Load the trained logistic regression model
# clf = joblib.load('my_model.pkl')
#
# # Predict the activity (walking or jumping)
# y_pred = clf.predict(X_test_acc)
#
# # Print the predicted activity
# if y_pred[0] == 0:
#     print('The motion is walking')
# else:
#     print('The motion is jumping')
#444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444
#Define a function to create the GUI window
def create_window():
    global window
    window = tk.Tk()
    window.title("Activity Recognition")
    window.geometry("500x400")
    window.configure(bg='white')

    # Add a heading to the window
    tk.Label(window, text="Activity Recognition", font=("Arial", 16), fg="blue", bg="white").pack(pady=20)

    # Add a label to the window
    tk.Label(window, text="Select a CSV file containing accelerometer data", font=("Arial", 12), fg="black", bg="white").pack(pady=10)

    # Add a button to the window
    tk.Button(window, text="Select File", font=("Arial", 12), command=select_file, bg="blue", fg="white", padx=10, pady=5).pack()

    # Add a label to display the predicted activity
    global activity_label
    activity_label = tk.Label(window, font=("Arial", 14), fg="black", bg="white")
    activity_label.pack(pady=20)

    # Run the main loop for the window
    window.mainloop()

def select_file():
    global filename
    filename = filedialog.askopenfilename(initialdir='/', title='Select a CSV file', filetypes=[('CSV files', '*.csv')])
    if filename:
        # Load the CSV file
        df = pd.read_csv(filename)

        # Segment data
        window_size = 500
        cols_of_interest = ['Acceleration x (m/s^2)', 'Acceleration x (m/s^2)', 'Acceleration x (m/s^2)']
        segments = []
        for col in cols_of_interest:
            signal = df[col].to_numpy()
            num_windows = math.floor(len(signal) / window_size)
            for i in range(num_windows):
                start = i * window_size
                end = (i + 1) * window_size
                segment = signal[start:end]
                feature = [
                    np.mean(segment),
                    np.median(segment),
                    np.var(segment),
                    np.std(segment),
                    np.max(segment),
                    np.min(segment),
                    np.max(segment) - np.min(segment),
                    skew(segment),
                    entropy(segment),
                    kurtosis(segment)
                ]
                segments.append(np.concatenate((feature, [df.iloc[start, -1]])))

        # Shuffle segments and split into train and test sets
        segments = np.array(segments)
        np.random.shuffle(segments)
        X_acc = segments[:, :-1]
        y_acc = segments[:, -1]
        X_train_acc, X_test_acc, y_train_acc, y_test_acc = train_test_split(X_acc, y_acc, test_size=0.1)
        X_train_acc = np.nan_to_num(X_train_acc, nan=0.0, posinf=1e6, neginf=-1e6)
        X_test_acc = np.nan_to_num(X_test_acc, nan=0.0, posinf=1e6, neginf=-1e6)

        # Load the trained logistic regression model
        clf = joblib.load('my_model.pkl')

        # Predict the activity (walking or jumping)
        y_pred = clf.predict(X_test_acc)

        # Print the predicted activity
        if y_pred[0] == 0:
            messagebox.showinfo('Activity Prediction', 'The motion is walking')
        else:
            messagebox.showinfo('Activity Prediction', 'The motion is jumping')
        

# Add a button to the GUI window
# root = tk.Tk()
# root.title("Activity Recognition")
# root.geometry("400x300")
# # Add a label to the GUI window
# label = tk.Label(root, text="Select a CSV file to perform activity recognition")
# label.pack(pady=20)
# button = tk.Button(root, text="Select File", command=select_file)
# button.pack()
#
# root.mainloop()
# Create GUI window
##########
# root = tk.Tk()
# root.title("Activity Recognition")
# root.geometry("600x400")
# root.configure(background="#FFFFFF")
#
# # Add label to the GUI window
# label = tk.Label(root, text="Select a CSV file to perform activity recognition", font=("Arial", 16), fg="#000000", bg="#FFFFFF")
# label.pack(pady=30)
#
# # Add button to the GUI window
# button = tk.Button(root, text="Select File", command=select_file, font=("Arial", 14), fg="#FFFFFF", bg="#4CAF50", padx=20, pady=10, borderwidth=0, activebackground="#3e8e41", activeforeground="#FFFFFF")
# button.pack()
#
# root.mainloop()
create_window()


