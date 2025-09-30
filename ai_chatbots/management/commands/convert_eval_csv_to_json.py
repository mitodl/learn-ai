"""Django management command to convert CSV evaluation data to JSON format."""

import csv
import json
from pathlib import Path

from django.core.management import BaseCommand


class Command(BaseCommand):
    """Convert CSV evaluation data to JSON format for RAG evaluations."""

    help = "Convert CSV evaluation data to JSON format suitable for RAG evaluations"

    def add_arguments(self, parser):
        """Add command arguments."""
        parser.add_argument(
            "--csv-file",
            dest="csv_file_path",
            required=True,
            help="Path to the input CSV file",
        )
        parser.add_argument(
            "--output-file",
            dest="output_file_path",
            required=True,
            help="Path for the output JSON file",
        )
        parser.add_argument(
            "--course-id",
            dest="course_id",
            required=True,
            help="Course ID to use in the JSON structure",
        )
        parser.add_argument(
            "--bot",
            dest="bot",
            required=True,
            choices=[
                "syllabus",
                "canvas_syllabus",
                "recommendation",
                "tutor",
                "videogpt",
            ],
            help="Bot type to use as entity name in JSON structure",
        )
        parser.add_argument(
            "--encoding",
            dest="encoding",
            default="utf-8",
            help="CSV file encoding (default: utf-8)",
        )

    def handle(self, **options):
        """Execute the command."""
        csv_file_path = Path(options["csv_file_path"])
        output_file_path = Path(options["output_file_path"])
        course_id = options["course_id"]
        encoding = options["encoding"]
        bot = options["bot"]

        # Validate input file exists
        if not csv_file_path.exists():
            self.stdout.write(self.style.ERROR(f"CSV file not found: {csv_file_path}"))
            return

        # Create output directory if it doesn't exist
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        syllabus_data = []

        try:
            with csv_file_path.open(encoding=encoding) as csvfile:
                reader = csv.reader(csvfile)

                # Skip the first row (headers)
                headers = next(reader)
                self.stdout.write(f"Headers: {headers}")

                # Process each row
                for row_num, row in enumerate(reader, start=2):
                    # Skip empty rows
                    min_columns = 5
                    if len(row) < min_columns or not row[2].strip():
                        continue

                    # Column C is index 2 (question)
                    question = row[2].strip()

                    # Column D is index 3 (answer)
                    answer = row[3].strip()

                    # Column E is index 4 (accuracy)
                    accuracy = row[4].strip()

                    # Determine expected_output based on accuracy
                    expected_output = answer if accuracy.lower() == "accurate" else ""

                    # Create JSON object
                    json_obj = {
                        "question": question,
                        "extra_state": {
                            "course_id": [course_id],
                            "collection_name": [None],
                            "exclude_canvas": [False]
                            if bot == "canvas_syllabus"
                            else [True],
                        },
                        "expected_output": expected_output,
                        "expected_tools": ["search_content_files"],
                    }

                    syllabus_data.append(json_obj)

                    # Print first few for verification
                    max_verification_rows = 5
                    if row_num <= max_verification_rows:
                        self.stdout.write(f"Row {row_num}:")
                        self.stdout.write(f"Question: {question[:50]}...")
                        self.stdout.write(f"Accuracy: '{accuracy}'")
                        self.stdout.write(
                            f"Has expected output: {bool(expected_output)}"
                        )

            # Create final JSON structure
            final_json = {bot: syllabus_data}

            # Write to file
            with output_file_path.open("w", encoding="utf-8") as jsonfile:
                json.dump(final_json, jsonfile, indent=2, ensure_ascii=False)

            self.stdout.write(
                self.style.SUCCESS(
                    f"Conversion complete! Created {len(syllabus_data)} entries."
                )
            )
            self.stdout.write(
                self.style.SUCCESS(f"Output saved to: {output_file_path}")
            )

        except FileNotFoundError:
            self.stdout.write(self.style.ERROR(f"File not found: {csv_file_path}"))
        except UnicodeDecodeError:
            self.stdout.write(
                self.style.ERROR(
                    f"Unable to decode file with encoding '{encoding}'. "
                    "Try a different encoding with --encoding option."
                )
            )
        except (OSError, ValueError) as e:
            self.stdout.write(self.style.ERROR(f"Error processing file: {e!s}"))
