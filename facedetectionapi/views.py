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




# from django.db.models import Q
# from .models import Attendance
from .serializers import AttendanceSerializer
from .pagination import CustomPagination

import csv
from django.http import HttpResponse
from django.db.models import Q
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

class AttendanceReportView(APIView):
    def get(self, request):
        download = request.query_params.get('download', None) == 'true'
        user_id = request.query_params.get('userID')
        date_from = request.query_params.get('from')
        date_to = request.query_params.get('to')

        filters = Q()
        if user_id:
            filters &= Q(user_id=user_id)
        if date_from:
            filters &= Q(date__gte=date_from)
        if date_to:
            filters &= Q(date__lte=date_to)

        # Filter attendance records
        attendance_records = Attendance.objects.filter(filters).select_related('user')

        # Check if download parameter is present
        if download:
            return self.generate_csv(attendance_records)

        # Default response for paginated data
        paginator = CustomPagination()
        paginated_records = paginator.paginate_queryset(attendance_records, request)
        serializer = AttendanceSerializer(paginated_records, many=True)
        return paginator.get_paginated_response(serializer.data)

    def generate_csv(self, attendance_records):
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="attendance_report.csv"'

        writer = csv.writer(response)
        writer.writerow(['User ID', 'User Name', 'Date', 'Time'])

        for record in attendance_records:
            writer.writerow([
                record.user.id,
                record.user.username,
                record.date,
                record.time
            ])

        return response


class AttendanceReportView(APIView):
    def get(self, request):
        user_id = request.query_params.get('userID')
        date_from = request.query_params.get('from')
        date_to = request.query_params.get('to')
        page_size = request.query_params.get('page_size')

    # WHERE user_id = userID AND date >= from AND date <= to
        filters = Q()
        if user_id:
            filters &= Q(user_id=user_id)
        if date_from:
            filters &= Q(date__gte=date_from)
        if date_to:
            filters &= Q(date__lte=date_to)

        attendance_records = Attendance.objects.filter(filters).select_related('user')
        
        paginator = CustomPagination()
        paginated_records = paginator.paginate_queryset(attendance_records, request)
        
        serializer = AttendanceSerializer(paginated_records, many=True)
        
        return paginator.get_paginated_response(serializer.data)
    
     
    
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from django.http import HttpResponse
from django.db.models import Q
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import AttendanceSerializer
from .models import Attendance
from .pagination import CustomPagination

class AttendancePDFReportView(APIView):
    def get(self, request):
        user_id = request.query_params.get('userID')
        date_from = request.query_params.get('from')
        date_to = request.query_params.get('to')
        page_size = request.query_params.get('page_size', 10)  

        filters = Q()
        if user_id:
            filters &= Q(user_id=user_id)
        if date_from:
            filters &= Q(date__gte=date_from)
        if date_to:
            filters &= Q(date__lte=date_to)

        attendance_records = Attendance.objects.filter(filters).select_related('user')

        paginator = CustomPagination()
        paginated_records = paginator.paginate_queryset(attendance_records, request)

        return self.generate_pdf(paginated_records)

    def generate_pdf(self, attendance_records):
        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="attendance_report.pdf"'

        p = canvas.Canvas(response, pagesize=letter)
        width, height = letter

        p.setFont("Helvetica-Bold", 16)
        p.drawString(100, height - 50, "Attendance Report")

        p.setFont("Helvetica-Bold", 12)
        y = height - 100
        p.drawString(50, y, "User ID")
        p.drawString(150, y, "User Name")
        p.drawString(300, y, "Date")
        p.drawString(450, y, "Time")

        p.setFont("Helvetica", 10)
        for record in attendance_records:
            y -= 20
            if y < 50:  
                p.showPage()
                p.setFont("Helvetica", 10)
                y = height - 50

            p.drawString(50, y, str(record.user.id))
            p.drawString(150, y, record.user.username)
            p.drawString(300, y, record.date.strftime("%Y-%m-%d"))
            p.drawString(450, y, record.time.strftime("%H:%M:%S"))

        p.save()
        return response
    


class UserDetails(APIView):
    def get(self, request):
        user_id = request.query_params.get('userID')
        
        if user_id:
            try:
                user=User.objects.get(id=user_id)
                serializer=UserSerializer(user)
                return Response(serializer.data, status=status.HTTP_200_OK)
            except User.DoesNotExist:
                return Response({"error":"User not found"},status=status.HTTP_404_NOT_FOUND)
        
        else:
            users=User.objects.all()
            serializer=UserSerializer(users,many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        

class UserUpdate(APIView):
    def put(self, request):
        user_id = request.query_params.get('userID')
        if not user_id:
            return Response(
                {"error":"userID is required"},
                status=status.HTTP_404_NOT_FOUND
            )
        try:
            user=User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response(
                {"error":"User not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        serializer=UserSerializer(user, data=request.data,partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        
        
        
    
            

        
        


    








