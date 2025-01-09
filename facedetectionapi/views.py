from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import User
from .serializers import UserSerializer
import face_recognition
import base64
import numpy as np
from PIL import Image


class UserSignup(APIView):
    def post(self, request):
        serializer = UserSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        image_file = request.FILES.get("photo")
        if not image_file:
            return Response({"error": "Image file is required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            image = Image.open(image_file)
            image = image.convert("RGB") 
            image_array = np.array(image)

            face_encodings = face_recognition.face_encodings(image_array)
            if not face_encodings:
                return Response({"error": "No face detected in the image"}, status=status.HTTP_400_BAD_REQUEST)

            face_encoding = face_encodings[0]

            face_encoding_str = base64.b64encode(face_encoding).decode("utf-8")

            serializer.save(face_encoding=face_encoding_str)
            return Response({"message": "User signed up successfully"}, status=status.HTTP_201_CREATED)

        except Exception as e:
            return Response({"error": f"Error processing the image: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    
    

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



import cv2
import face_recognition
from .models import User
import numpy as np
import base64 
from datetime import datetime
from .models import Attendance


class FaceInputView(APIView):
    def post(self, request):
        face_cap = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        video_cap = cv2.VideoCapture(0)
        if not video_cap.isOpened():
            return Response({"error": "Could not access the camera"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        face_details = []
        try:
            while True:
                ret, video_data = video_cap.read()
                if not ret:
                    return Response({"error": "Failed to capture video frame"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                rgb_frame = cv2.cvtColor(video_data, cv2.COLOR_BGR2RGB)

                faces = face_cap.detectMultiScale(
                    cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY),
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
            face_locations = [(face['y'], face['x'] + face['width'], face['y'] + face['height'], face['x']) for face in face_details]
            camera_face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)

            if camera_face_encoding:  
                camera_face_encoding = camera_face_encoding[0]  
                          
                user_profiles = User.objects.all()
                for user_profile in user_profiles:
                    stored_face_encoding = user_profile.face_encoding

                    if not stored_face_encoding:
                        return Response({"error": f"User {user_profile.id} does not have a stored face encoding"}, status=status.HTTP_400_BAD_REQUEST)
                    
                    try:
                        # Decode the stored face encoding from Base64
                        stored_face_encoding = np.frombuffer(base64.b64decode(stored_face_encoding), dtype=np.float64)
                    except Exception as e:
                        return Response({"error": f"Error decoding stored face encoding: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

                    #Euclidean distance between stored image and camera image
                    face_distance = face_recognition.face_distance([stored_face_encoding], camera_face_encoding)[0]

                    threshold = 0.6

                    if face_distance < threshold:
                        try:
                            Attendance.objects.create(
                                user=user_profile,
                                date=datetime.now().date(),
                                time=datetime.now().time()
                            )
                            return Response({
                                "message": "Faces match",
                                "result": True,
                                "user_id": user_profile.id,
                                "distance": face_distance
                            }, status=status.HTTP_200_OK)
                        except Exception as e:
                            return Response({"error": f"Failed to record attendance: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                return Response({
                    "message": "Faces do not match with any stored user",
                    "result": False
                }, status=status.HTTP_400_BAD_REQUEST)

            else:
                return Response({"message": "No valid face encoding detected from camera"}, status=status.HTTP_404_NOT_FOUND)

        else:
            return Response({"message": "No face detected in the video frame"}, status=status.HTTP_404_NOT_FOUND)





from django.db.models import Q
from .models import Attendance
from .serializers import AttendanceSerializer

class AttendanceReportView(APIView):
    def get(self, request):
        user_id = request.query_params.get('userID')
        date_from = request.query_params.get('from')
        date_to = request.query_params.get('to')

    # WHERE user_id = userID AND date >= from AND date <= to
        filters = Q()
        if user_id:
            filters &= Q(user_id=user_id)
        if date_from:
            filters &= Q(date__gte=date_from)
        if date_to:
            filters &= Q(date__lte=date_to)

        attendance_records = Attendance.objects.filter(filters).select_related('user')

        serializer = AttendanceSerializer(attendance_records, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)







