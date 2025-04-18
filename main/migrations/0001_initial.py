# Generated by Django 4.2.20 on 2025-04-07 12:11

from django.db import migrations, models


def set_initial_rate_limits(apps, schema_editor):
    """Populate initial rate limits for consumers"""
    ConsumerThrottleLimit = apps.get_model("main", "ConsumerThrottleLimit")
    for throttle_key in (
        "recommendation_bot",
        "syllabus_bot",
        "video_gpt_bot",
        "tutor_bot",
    ):
        ConsumerThrottleLimit.objects.get_or_create(
            throttle_key=throttle_key,
            defaults={
                "auth_limit": 1000,
                "anon_limit": 500,
                "interval": "day",
            },
        )


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="ConsumerThrottleLimit",
            fields=[
                (
                    "throttle_key",
                    models.CharField(max_length=255, primary_key=True, serialize=False),
                ),
                ("auth_limit", models.IntegerField(default=0)),
                ("anon_limit", models.IntegerField(default=0)),
                (
                    "interval",
                    models.CharField(
                        choices=[
                            ("minute", "minute"),
                            ("hour", "hour"),
                            ("day", "day"),
                            ("week", "week"),
                        ],
                        max_length=12,
                    ),
                ),
            ],
        ),
        migrations.RunPython(
            set_initial_rate_limits, reverse_code=migrations.RunPython.noop
        ),
    ]
