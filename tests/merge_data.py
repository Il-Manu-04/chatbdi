import csv
import os

INPUT_DIR = os.path.join(os.path.dirname(__file__), "input")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "all_data.csv")

domains = [
    "booking",
    "car_control",
    "cooking",
    "domestic_robot",
    "employer_management",
    "games_platform",
    "house_builder",
    "onbanking",
    "robot_assistant",
    "tickets",
]

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as out_f:
    writer = None
    total = 0
    for domain in domains:
        path = os.path.join(INPUT_DIR, domain, "data.csv")
        with open(path, encoding="utf-8") as in_f:
            reader = csv.DictReader(in_f, delimiter=";")
            if writer is None:
                fieldnames = reader.fieldnames + ["domain"]
                writer = csv.DictWriter(out_f, fieldnames=fieldnames, delimiter=";")
                writer.writeheader()
            for row in reader:
                row["domain"] = domain
                writer.writerow(row)
                total += 1
        print(f"  {domain}: OK")

print(f"\nTotale righe: {total}")
print(f"Output: {OUTPUT_FILE}")
