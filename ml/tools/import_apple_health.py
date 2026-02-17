"""Import Apple Health export.xml into NeuralDreamWorkshop.

Usage:
    1. On your iPhone: Settings → Health → Export All Health Data
    2. Send the ZIP to your Mac (AirDrop, email, etc.)
    3. Unzip it — you'll get a folder with 'export.xml'
    4. Run: python3 ml/tools/import_apple_health.py /path/to/export.xml

This parses the XML, extracts supported health metrics, and stores them
in our SQLite database for brain-health correlation analysis.
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from health.apple_health import HEALTHKIT_TYPE_MAP
from health.correlation_engine import HealthBrainDB


SUPPORTED_TYPES = set(HEALTHKIT_TYPE_MAP.keys())


def parse_datetime(date_str: str) -> float:
    """Parse Apple Health date string to unix timestamp."""
    # Format: "2024-01-15 08:30:00 -0500"
    for fmt in [
        "%Y-%m-%d %H:%M:%S %z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
    ]:
        try:
            return datetime.strptime(date_str, fmt).timestamp()
        except ValueError:
            continue
    return 0.0


def import_apple_health(xml_path: str, user_id: str = "default", max_days: int = 90):
    """Import Apple Health export.xml into our database.

    Args:
        xml_path: Path to export.xml file.
        user_id: User ID to store data under.
        max_days: Only import last N days of data (default 90).
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        print(f"Error: File not found: {xml_path}")
        sys.exit(1)

    # Check if it's a zip file
    if xml_path.suffix == ".zip":
        import zipfile
        import tempfile
        tmpdir = tempfile.mkdtemp()
        with zipfile.ZipFile(xml_path) as zf:
            zf.extractall(tmpdir)
        # Find export.xml inside
        candidates = list(Path(tmpdir).rglob("export.xml"))
        if not candidates:
            print("Error: No export.xml found in ZIP file")
            sys.exit(1)
        xml_path = candidates[0]
        print(f"Extracted: {xml_path}")

    print(f"Parsing {xml_path.name}...")
    print(f"  User ID: {user_id}")
    print(f"  Max days: {max_days}")

    cutoff = datetime.now().timestamp() - (max_days * 86400)

    db = HealthBrainDB()
    counts = {}
    total = 0
    skipped = 0

    # Stream-parse the XML (can be very large, 500MB+)
    context = ET.iterparse(str(xml_path), events=("end",))

    for event, elem in context:
        if elem.tag != "Record":
            elem.clear()
            continue

        record_type = elem.get("type", "")

        # Only import types we support
        if record_type not in SUPPORTED_TYPES:
            elem.clear()
            continue

        start_date = elem.get("startDate", "")
        end_date = elem.get("endDate", "")
        value_str = elem.get("value", "0")
        source_name = elem.get("sourceName", "Apple Health")
        unit = elem.get("unit", "")

        timestamp = parse_datetime(start_date)

        # Skip old data
        if timestamp < cutoff:
            skipped += 1
            elem.clear()
            continue

        # Parse value
        try:
            # Sleep analysis uses string values like "HKCategoryValueSleepAnalysisAsleepDeep"
            if record_type == "HKCategoryTypeIdentifierSleepAnalysis":
                sleep_value_map = {
                    "HKCategoryValueSleepAnalysisInBed": 0,
                    "HKCategoryValueSleepAnalysisAsleepUnspecified": 1,
                    "HKCategoryValueSleepAnalysisAsleep": 1,
                    "HKCategoryValueSleepAnalysisAwake": 2,
                    "HKCategoryValueSleepAnalysisAsleepCore": 3,
                    "HKCategoryValueSleepAnalysisAsleepDeep": 4,
                    "HKCategoryValueSleepAnalysisAsleepREM": 5,
                }
                value = float(sleep_value_map.get(value_str, 0))
            elif record_type == "HKCategoryTypeIdentifierMindfulSession":
                # Duration in minutes
                end_ts = parse_datetime(end_date)
                value = (end_ts - timestamp) / 60.0 if end_ts > timestamp else 0
            else:
                value = float(value_str)
        except (ValueError, TypeError):
            elem.clear()
            continue

        metric = HEALTHKIT_TYPE_MAP[record_type]
        end_timestamp = parse_datetime(end_date)

        sample = {
            "timestamp": timestamp,
            "end_timestamp": end_timestamp,
            "value": value,
            "source": source_name,
            "metadata": {"unit": unit},
        }

        db.store_health_samples(user_id, metric, [sample])

        counts[metric] = counts.get(metric, 0) + 1
        total += 1

        # Progress every 5000 records
        if total % 5000 == 0:
            print(f"  Imported {total:,} records...")

        elem.clear()

    print(f"\nDone! Imported {total:,} records ({skipped:,} skipped as older than {max_days} days)")
    print("\nMetrics imported:")
    for metric, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {metric}: {count:,} samples")

    # Show a quick summary
    print(f"\nDatabase: {db.db_path}")
    print("\nGenerating quick summary for today...")
    summary = db.get_daily_summary(user_id)
    if summary["health"]:
        print("\nToday's health data:")
        for metric, stats in summary["health"].items():
            print(f"  {metric}: avg={stats['average']}, min={stats['min']}, max={stats['max']} ({stats['count']} samples)")
    else:
        print("  No data for today (try checking recent dates)")

    return counts


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("Example:")
        print("  python3 ml/tools/import_apple_health.py ~/Downloads/export.xml")
        print("  python3 ml/tools/import_apple_health.py ~/Downloads/export.zip --user myname --days 30")
        sys.exit(1)

    xml_file = sys.argv[1]
    user = "default"
    days = 90

    # Simple arg parsing
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] in ("--user", "-u") and i + 1 < len(args):
            user = args[i + 1]
            i += 2
        elif args[i] in ("--days", "-d") and i + 1 < len(args):
            days = int(args[i + 1])
            i += 2
        else:
            i += 1

    import_apple_health(xml_file, user_id=user, max_days=days)
