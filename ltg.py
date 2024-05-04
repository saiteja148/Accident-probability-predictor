import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import geocoder

def get_location():
    try:
        location = geocoder.ip('me')
        if location.ok:
            return location.latlng 
        else:
            return None
    except Exception as e:
        print(f"Error getting location: {e}")
        return None

# Read the dataset
data = pd.read_csv(r"C:\Users\C KRISHNA\Documents\AccidentsBig.csv")
print(data.head(5))

# Drop rows with missing values in relevant columns
data.dropna(subset=['latitude', 'longitude', 'Accident_Severity', 'Time'], inplace=True)

# Convert 'Time' column to numerical features (hour and minute)
data['Hour'] = pd.to_datetime(data['Time']).dt.hour
data['Minute'] = pd.to_datetime(data['Time']).dt.minute

# Select features and target variable
X = data[['latitude', 'longitude', 'Hour', 'Minute']]  
y = data['Accident_Severity']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
cluster_labels = kmeans.labels_

# Add cluster labels as a new feature
data['Cluster_Labels'] = cluster_labels

# Visualize the clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x='longitude', y='latitude', hue='Cluster_Labels', data=data, palette='viridis')
plt.title('Accident Clusters')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[['latitude', 'longitude', 'Hour', 'Minute', 'Cluster_Labels']], y, test_size=0.2, random_state=42)

# Train the logistic regression model with cluster labels as features
model = LogisticRegression()
model.fit(X_train, y_train)

if __name__ == "__main__":
    # Get user's current location
    coordinates = get_location()
    if coordinates:
        user_latitude, user_longitude = coordinates
        print(f"Current Location - Latitude: {user_latitude}, Longitude: {user_longitude}")

        # Get current time
        current_time = datetime.datetime.now()

        # Extract hours and minutes
        user_hour = current_time.hour
        user_minute = current_time.minute

        # Predict probability of an accident at current location and time
        user_cluster = kmeans.predict(scaler.transform([[user_latitude, user_longitude, user_hour, user_minute]]))[0]
        input_data = [[user_latitude, user_longitude, user_hour, user_minute, user_cluster]]
        probability = model.predict_proba(input_data)
        print("Probability of accident at current coordinates and time:", probability)
    else:
        print("Failed to retrieve location.")
