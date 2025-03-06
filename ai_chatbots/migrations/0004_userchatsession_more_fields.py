# Generated by Django 4.2.19 on 2025-03-04 19:42

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("ai_chatbots", "0003_tutorbotoutput"),
    ]

    operations = [
        migrations.AddField(
            model_name="userchatsession",
            name="dj_session_key",
            field=models.CharField(blank=True, db_index=True, max_length=512),
        ),
        migrations.AddField(
            model_name="userchatsession",
            name="object_id",
            field=models.CharField(blank=True, db_index=True, max_length=256),
        ),
        migrations.AlterField(
            model_name="userchatsession",
            name="agent",
            field=models.CharField(blank=True, db_index=True, max_length=128),
        ),
        migrations.AddIndex(
            model_name="djangocheckpoint",
            index=models.Index(
                fields=["thread_id", "checkpoint_ns", "checkpoint_id"],
                name="checkpoint_lookup_idx",
            ),
        ),
        migrations.AddIndex(
            model_name="djangocheckpoint",
            index=models.Index(fields=["thread_id"], name="thread_lookup_idx"),
        ),
    ]
