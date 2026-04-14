import csv
import os
import re

BASE_DIR = os.path.dirname(__file__)
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_FILE = os.path.join(BASE_DIR, "all_literals.csv")

DOMAINS = [
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
    writer = csv.DictWriter(
        out_f,
        fieldnames=["literal", "functor", "domain"],
        delimiter=";",
    )
    writer.writeheader()

    total = 0
    for domain in DOMAINS:
        path = os.path.join(INPUT_DIR, domain, "literals.csv")
        with open(path, encoding="utf-8") as in_f:
            next(in_f)  # salta header "Literals"
            for line in in_f:
                literal = line.strip()
                if not literal:
                    continue

                match = re.match(r"^([a-zA-Z0-9_]+)", literal)
                functor = match.group(1) if match else ""

                writer.writerow({
                    "literal": literal,
                    "functor": functor,
                    "domain": domain,
                })
                total += 1

        print(f"  {domain}: OK")

print(f"\nTotale literal: {total}")
print(f"Output: {OUTPUT_FILE}")
