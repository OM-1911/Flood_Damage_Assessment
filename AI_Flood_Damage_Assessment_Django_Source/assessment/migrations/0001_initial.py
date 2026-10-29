from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True
    dependencies = []
    operations = [
        migrations.CreateModel(
            name='FloodAssessment',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('uploaded_image', models.ImageField(upload_to='uploads/')),
                ('mask_image', models.ImageField(blank=True, null=True, upload_to='results/masks/')),
                ('result_image', models.ImageField(blank=True, null=True, upload_to='results/overlays/')),
                ('flood_percentage', models.FloatField(default=0.0)),
                ('severity', models.CharField(default='Unknown', max_length=30)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
            options={'ordering': ['-created_at']},
        ),
    ]
