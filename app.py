from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    redirect,
    url_for,
)
import os
from ultralytics import YOLO
import cv2
import numpy as np
import pyodbc
from datetime import datetime
import smtplib
from email.mime.text import MIMEText

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Paths for uploads and results
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
MISSING_PIC_FOLDER = "static/missing_pics"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER
app.config["MISSING_PIC_FOLDER"] = MISSING_PIC_FOLDER

# Load YOLOv8 model
model = YOLO("best1.pt")  # Replace with your model path

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(MISSING_PIC_FOLDER, exist_ok=True)

# SQL Server connection
conn_str = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=BABURAM\\SQLEXPRESS01;"  # Replace with your server name
    "DATABASE=MissingPersonsDB;"
    "Trusted_Connection=yes;"
)

# Email configuration (replace with your real credentials)
EMAIL_ADDRESS = "sagarphuyal44@gmail.com"  # Replace with your Gmail address
EMAIL_PASSWORD = "mhjo lljd vjss tsmo"  # Replace with your 16-character App Password
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Store notifications in memory
notifications = []


# Admin Page
@app.route("/admin")
def index():
    return render_template("admin.html", notifications=notifications)


# Drone Detection Upload (Admin-only)
@app.route("/admin/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file part", 400
    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)
    result_path, person_detected, user_email = process_file(file_path, file.filename)

    return render_template(
        "admin.html",
        uploaded_file=file.filename,
        result_file=os.path.basename(result_path),
        person_detected=person_detected,
        user_email=user_email,
        notifications=notifications,
    )


def process_file(file_path, filename):
    is_video = filename.lower().endswith((".mp4", ".avi", ".mov"))
    result_filename = f"result_{filename}"
    result_path = os.path.join(app.config["RESULT_FOLDER"], result_filename)
    person_detected = False
    user_email = None

    # Get the latest user email from the database
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT TOP 1 Email FROM MissingPersonReports ORDER BY SubmittedDate DESC"
    )
    result = cursor.fetchone()
    if result:
        user_email = result[0]
    cursor.close()
    conn.close()

    if is_video:
        cap = cv2.VideoCapture(file_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            result_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4)))
        )
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
            for r in results[0].boxes:
                if r.cls == 0 and r.conf > 0.3:  # Class 0 is 'person'
                    person_detected = True
                    break
        cap.release()
        out.release()
    else:
        img = cv2.imread(file_path)
        results = model(img)
        annotated_img = results[0].plot()
        cv2.imwrite(result_path, annotated_img)
        for r in results[0].boxes:
            if r.cls == 0 and r.conf > 0.3:  # Class 0 is 'person'
                person_detected = True
                break

    return result_path, person_detected, user_email


# Notification Page
@app.route("/admin/send_notification", methods=["GET", "POST"])
def send_notification():
    user_email = request.args.get("email")
    if request.method == "POST":
        to_email = request.form["to"]
        from_email = request.form["from"]
        message = request.form["message"]

        try:
            send_email(to_email, "Person Detected Notification", message)
            notifications.append(f"Email successfully sent to {to_email}")
        except Exception as e:
            notifications.append(f"Failed to send email to {to_email}: {str(e)}")

        return redirect(url_for("index"))

    return render_template(
        "send_notification.html", to_email=user_email, from_email=EMAIL_ADDRESS
    )


def send_email(to_email, subject, body):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = to_email

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)


# User Page (Report Missing Person)
@app.route("/", methods=["GET", "POST"])
def report_missing():
    if request.method == "POST":
        full_name = request.form["full_name"]
        address = request.form["address"]
        email = request.form["email"]
        phone = request.form["phone"]
        missing_full_name = request.form["missing_full_name"]
        missing_age = request.form["missing_age"]
        description = request.form["description"]
        missing_pic = request.files.get("missing_pic")

        pic_path = None
        if missing_pic and missing_pic.filename != "":
            pic_filename = f"{missing_full_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            pic_path = os.path.join(app.config["MISSING_PIC_FOLDER"], pic_filename)
            missing_pic.save(pic_path)
            pic_path = f"missing_pics/{pic_filename}"

        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO MissingPersonReports (FullName, Address, Email, PhoneNumber, MissingFullName, MissingAge, MissingPicPath, Description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                full_name,
                address,
                email,
                phone,
                missing_full_name,
                int(missing_age),
                pic_path,
                description,
            ),
        )
        conn.commit()
        cursor.close()
        conn.close()

        notifications.append(
            f"Missing person report submitted for {missing_full_name} by {full_name}"
        )
        return render_template("thank_you.html")

    return render_template("report_missing.html")


@app.route("/static/results/<filename>")
def send_result_file(filename):
    return send_from_directory(app.config["RESULT_FOLDER"], filename)


@app.route("/static/missing_pics/<filename>")
def send_missing_pic(filename):
    return send_from_directory(app.config["MISSING_PIC_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)
