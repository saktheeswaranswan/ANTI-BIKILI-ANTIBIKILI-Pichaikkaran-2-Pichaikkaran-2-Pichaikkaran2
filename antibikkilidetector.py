import cv2
import numpy as np
import time
import requests
import csv
from bs4 import BeautifulSoup

# Load the YOLOv3-tiny model
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")

# Load the COCO class names
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Create folders for storing cropped images
crop_folder = "CroppedImages"
anti_bikili_folder = f"{crop_folder}/AntiBikili"
bikili_folder = f"{crop_folder}/Bikili"
abqr_folder = f"{crop_folder}/ABQR"
bqr_folder = f"{crop_folder}/BQR"

for folder in [crop_folder, anti_bikili_folder, bikili_folder, abqr_folder, bqr_folder]:
    os.makedirs(folder, exist_ok=True)

# Create a list to store the detected objects
objects = []

# Create a dictionary to store the cheapest prices
prices = {}

# Create a list to store the search links
search_links = []

# Create a CSV file to store the results
csv_file = open("results.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Detected Class", "Shop Name", "Date", "Time", "Search Link"])

# Function to search for the cheapest price on a specific website
def search_cheapest_price(label):
    cheapest_price = None
    shop_name = ""
    search_link = ""
    
    for site in ["amazon", "flipkart", "myntra", "mesho"]:
        # Search for the product on the site
        url = f"https://{site}/search?q={label}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        # Get the prices and links of the products
        prices = soup.find_all("span", class_="a-price-whole")
        links = soup.find_all("a", class_="a-link-normal a-text-normal")

        # Find the cheapest price and the corresponding shop name and search link
        if len(prices) > 0 and len(links) > 0:
            current_price = float(prices[0].text)
            if cheapest_price is None or current_price < cheapest_price:
                cheapest_price = current_price
                shop_name = site.capitalize()
                search_link = f"https://{site}{links[0]['href']}"

    return cheapest_price, shop_name, search_link


# Create a video capture object
cap = cv2.VideoCapture(0)

while True:
    # Capture the frame from the video
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect objects in the frame
    detections = net.detectMultiScale(gray, 0.5, 0.5, 0, 200)

    # Iterate over the detected objects
    for detection in detections:
        # Get the bounding box of the object
        x, y, w, h = detection[0:4]

        # Crop the object from the frame
        crop = frame[y:y + h, x:x + w]

        # Get the class of the object
        label = classes[detection[5]]

        # Add the object to the list of detected objects
        objects.append([label, crop])

    # Iterate over the detected objects
    for object in objects:
        # Get the label of the object
        label = object[0]

        # Get the crop of the object
        crop = object[1]

        # Save the cropped object based on the detected class
        if label == "Anti Bikili":
            cv2.imwrite(f"{anti_bikili_folder}/{int(time.time())}.jpg", crop)
        elif label == "Bikili":
            cv2.imwrite(f"{bikili_folder}/{int(time.time())}.jpg", crop)
        elif label == "ABQR":
            cv2.imwrite(f"{abqr_folder}/{int(time.time())}.jpg", crop)
        elif label == "BQR":
            cv2.imwrite(f"{bqr_folder}/{int(time.time())}.jpg", crop)

        # Search for the cheapest price and retrieve the shop name and search link
        cheapest_price, shop_name, search_link = search_cheapest_price(label)

        # Add the cheapest price and search link to the respective dictionaries
        prices[label] = cheapest_price
        search_links.append(search_link)

        # Write the results to the CSV file
        csv_writer.writerow([label, shop_name, time.strftime("%Y-%m-%d"), time.strftime("%H:%M:%S"), search_link])

    # Display the frame
    cv2.imshow("Frame", frame)

    # Wait for a key press
    key = cv2.waitKey(1)

    # If the `q` key is pressed, stop the loop
    if key == ord("q"):
        break

# Release the video capture object
cap.release()

# Close the CSV file
csv_file.close()

