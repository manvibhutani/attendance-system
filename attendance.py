import cv2
import time
import face_recognition
import pandas as pd
from simplefacerec import SimpleFacerec

# Initialize face recognition model
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Load existing DataFrame or create a new one
excel_filename = "face_recognition_log.xlsx"
try:
    df = pd.read_excel(excel_filename)
except FileNotFoundError:
    columns = ["Name", "Timestamp"]
    df = pd.DataFrame(columns=columns)

cap = cv2.VideoCapture(0)
unknown_detected = False

while True:
    ret, frame = cap.read()
    face_locations, face_names = sfr.detect_known_faces(frame)

    for face_loc, name in zip(face_locations, face_names):
        y1, x1, y2, x2 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        cv2.putText(frame, name, ((x2 + 60), (y1 - 10)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 2)

        # Check if the face data already exists for the person on the current date
        today_date = time.strftime("%Y-%m-%d")
        person_data_exists = (df["Name"] == name) & (df["Timestamp"].str.startswith(today_date))

        if not person_data_exists.any():
            # Add data to DataFrame
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            df = df.append({"Name": name, "Timestamp": timestamp}, ignore_index=True)

        if name == "Unknown":
            unknown_detected = True
        else:
            unknown_detected = False

    if unknown_detected:
        # Sleep for 2 seconds and then break out of the loop
        time.sleep(2)
        break

    cv2.imshow("Smart Door Lock", frame)
    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Save DataFrame to Excel, appending new data
with pd.ExcelWriter(excel_filename, engine='openpyxl', mode='a') as writer:
    df.to_excel(writer, index=False, sheet_name='Sheet1', startrow=len(pd.read_excel(excel_filename)) + 1, header=False)

print(f"Face recognition log saved to {excel_filename}")
