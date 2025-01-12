"""
URL configuration for facedetectionapi project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from .views import UserSignup,UserLogin,FaceInputView, AttendanceReportView,AttendancePDFReportView,UserDetails


urlpatterns = [
    path('admin/', admin.site.urls),
    path('signup/', UserSignup.as_view(), name='user_signup'),
    path('login/', UserLogin.as_view(), name='user_login'),
    path('face-input/', FaceInputView.as_view(), name='face-input'),
    path('attendance-report/', AttendanceReportView.as_view(), name='attendance-report'),
    path('attendance-report-pdf/', AttendancePDFReportView.as_view(), name='attendance-report-pdf'),
    path('user-details/', UserDetails.as_view(), name='user-details'),
]
