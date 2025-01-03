from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import User
from .serializers import UserSerializer


class UserSignup(APIView):
    def post(self, request):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response({"message": "User signed up successfully"}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    

class UserLogin(APIView):
    def post(self, request):
        username=request.data.get('username','').strip()
        password=request.data.get('password','').strip()
        
        if not username or not password:
            return Response ({"error": "username and password are required"}, status=status.HTTP_400_BAD_REQUEST)
                    
        try:
            user = User.objects.get(username__iexact=username.strip())
        except User.DoesNotExist as e:
            print(f"Error: {str(e)}")
            return Response({"error": "Invalid username or password"}, status=status.HTTP_400_BAD_REQUEST)
        
        if password == user.password:
            return Response({"message": "Login successful"}, status=status.HTTP_200_OK)
        else:
            return Response({"error": "Invalid username or password"}, status=status.HTTP_400_BAD_REQUEST)




from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import cv2

class FaceInputView(APIView):
    def post(self, request):
        # Load Haar Cascade for face detection
        face_cap = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Open the camera
        video_cap = cv2.VideoCapture(0)
        if not video_cap.isOpened():
            return Response({"error": "Could not access the camera"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        face_details = []
        try:
            while True:
                ret, video_data = video_cap.read()
                if not ret:
                    return Response({"error": "Failed to capture video frame"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                # Convert the frame to grayscale
                col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

                # Detect faces
                faces = face_cap.detectMultiScale(
                    col,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                face_details = []
                for (x, y, w, h) in faces:
                    face_details.append({"x": x, "y": y, "width": w, "height": h})
                    cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.imshow("Face Detection", video_data)

                if cv2.waitKey(10) == ord("a"):
                    break

        finally:
            video_cap.release()
            cv2.destroyAllWindows()

        if face_details:
            return Response({"message": "Face(s) detected", "faces": face_details}, status=status.HTTP_200_OK)
        else:
            return Response({"message": "No face detected"}, status=status.HTTP_404_NOT_FOUND)



import cv2
import numpy as np
from django.core.files.storage import default_storage
from django.conf import settings
from .models import Attendance

class MarkAttendance(APIView):
    def post(self, request):
        camera_image = request.FILES.get('image')
        if not camera_image:
            return Response({"error": "Image not provided"}, status=status.HTTP_400_BAD_REQUEST)

        for user in User.objects.all():
            user_photo_path = default_storage.path(user.photo.name)
            user_photo = cv2.imread(user_photo_path)
            user_photo_gray = cv2.cvtColor(user_photo, cv2.COLOR_BGR2GRAY)
            
            camera_image_array = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
            camera_image_decoded = cv2.imdecode(camera_image_array, cv2.IMREAD_COLOR)
            camera_image_gray = cv2.cvtColor(camera_image_decoded, cv2.COLOR_BGR2GRAY)

            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            user_faces = face_cascade.detectMultiScale(user_photo_gray, scaleFactor=1.1, minNeighbors=5)
            camera_faces = face_cascade.detectMultiScale(camera_image_gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in camera_faces:
                for (ux, uy, uw, uh) in user_faces:
                    user_face_crop = user_photo_gray[uy:uy+uh, ux:ux+uw]
                    camera_face_crop = camera_image_gray[y:y+h, x:x+w]
                    user_face_resized = cv2.resize(user_face_crop, (100, 100))
                    camera_face_resized = cv2.resize(camera_face_crop, (100, 100))

                    diff = cv2.absdiff(user_face_resized, camera_face_resized)
                    if np.mean(diff) < 50:  # Threshold for similarity
                        Attendance.objects.create(user=user)
                        return Response({"message": "Attendance marked"}, status=status.HTTP_200_OK)

        return Response({"message": "Your attendance is not recorded"}, status=status.HTTP_400_BAD_REQUEST)







from django.utils.dateparse import parse_date

class AttendanceReport(APIView):
    def get(self, request):
        date = request.query_params.get('date')
        if not date:
            return Response({"error": "Date parameter is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            parsed_date = parse_date(date)
            attendance_records = Attendance.objects.filter(date=parsed_date).select_related('user')
            report = [
                {
                    "username": record.user.username,
                    "email": record.user.email,
                    "date": record.date,
                    "time": record.time,
                }
                for record in attendance_records
            ]
            return Response(report, status=status.HTTP_200_OK)
        except ValueError:
            return Response({"error": "Invalid date format"}, status=status.HTTP_400_BAD_REQUEST)


