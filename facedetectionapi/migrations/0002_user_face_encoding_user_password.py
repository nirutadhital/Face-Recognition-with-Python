# Generated by Django 4.2.17 on 2025-01-05 14:11

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('facedetectionapi', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='face_encoding',
            field=models.TextField(blank=True, null=True),
        ),
        # migrations.AddField(
        #     model_name='user',
        #     name='password',
        #     field=models.CharField(default=django.utils.timezone.now, max_length=128),
        #     preserve_default=False,
        # ),
    ]
