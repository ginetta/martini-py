# Generated by Django 4.2.2 on 2023-06-29 06:34

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('documents', '0003_create_default_documentcollection'),
    ]

    operations = [
        migrations.AddField(
            model_name='unstructureddocument',
            name='has_embeddings',
            field=models.BooleanField(default=False),
        ),
        migrations.AlterField(
            model_name='unstructureddocument',
            name='name',
            field=models.CharField(default='Untitled', max_length=255),
        ),
    ]
