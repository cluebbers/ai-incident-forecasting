"""Helper functions for processing MongoDB-backed incident data."""

from datetime import datetime
from typing import Any, Dict, List, Optional

import json
import pandas as pd

import os
from pymongo import MongoClient


MONGO_CONFIG = {
    'host': os.getenv('MONGO_HOST', 'localhost'),
    'port': int(os.getenv('MONGO_PORT', 27017)),
    'database': os.getenv('MONGO_DATABASE', 'aiidprod'),
    'incidents_collection': os.getenv('MONGO_INCIDENTS_COLLECTION', 'incidents'),
    'reports_collection': os.getenv('MONGO_REPORTS_COLLECTION', 'reports'),
    'classifications_collection': os.getenv('MONGO_CLASSIFICATIONS_COLLECTION', 'classifications'),
    'username': os.getenv('MONGO_USERNAME'),
    'password': os.getenv('MONGO_PASSWORD')
}

print("MongoDB Configuration:")
for key, value in MONGO_CONFIG.items():
    if 'password' in key.lower():
        print(f"  {key}: {'***' if value else 'Not set'}")
    else:
        print(f"  {key}: {value}")


def connect_to_mongodb():
    """Establish connection to MongoDB"""
    try:
        if MONGO_CONFIG['username'] and MONGO_CONFIG['password']:
            connection_string = (
                f"mongodb://{MONGO_CONFIG['username']}:{MONGO_CONFIG['password']}@"
                f"{MONGO_CONFIG['host']}:{MONGO_CONFIG['port']}/{MONGO_CONFIG['database']}"
            )
        else:
            connection_string = f"mongodb://{MONGO_CONFIG['host']}:{MONGO_CONFIG['port']}"

        client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("✅ Successfully connected to MongoDB")
        return client
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        return None


def load_incidents_from_mongo(client):
    """Load incidents from MongoDB"""
    try:
        db = client[MONGO_CONFIG['database']]
        incidents_collection = db[MONGO_CONFIG['incidents_collection']]

        incidents_cursor = incidents_collection.find()
        incidents_data = list(incidents_cursor)
        print(f"Loaded {len(incidents_data)} incidents from MongoDB")

        incidents_df = pd.DataFrame(incidents_data)

        if 'date' in incidents_df.columns:
            print("Converting incident dates...")
            incidents_df['date'] = incidents_df['date'].apply(safe_date_conversion)
            invalid_dates = incidents_df['date'].isna().sum()
            if invalid_dates > 0:
                print(f"Warning: Found {invalid_dates} incidents with invalid dates, removing them")
                incidents_df = incidents_df.dropna(subset=['date'])
        elif 'created_at' in incidents_df.columns:
            print("Converting incident created_at dates...")
            incidents_df['date'] = incidents_df['created_at'].apply(safe_date_conversion)
            invalid_dates = incidents_df['date'].isna().sum()
            if invalid_dates > 0:
                print(f"Warning: Found {invalid_dates} incidents with invalid dates, removing them")
                incidents_df = incidents_df.dropna(subset=['date'])

        return incidents_df
    except Exception as e:
        print(f"Error loading incidents: {e}")
        return pd.DataFrame()


def load_reports_from_mongo(client):
    """Load reports from MongoDB with robust date handling"""
    try:
        db = client[MONGO_CONFIG['database']]
        reports_collection = db[MONGO_CONFIG['reports_collection']]

        reports_cursor = reports_collection.find()
        reports_data = list(reports_cursor)
        print(f"Loaded {len(reports_data)} reports from MongoDB")

        reports_df = pd.DataFrame(reports_data)

        if 'date_published' in reports_df.columns:
            print("Converting report publication dates...")
            reports_df['date_published'] = reports_df['date_published'].apply(safe_date_conversion)
            invalid_dates = reports_df['date_published'].isna().sum()
            if invalid_dates > 0:
                print(
                    f"Warning: Found {invalid_dates} reports with invalid publication dates, removing them"
                )
                reports_df = reports_df.dropna(subset=['date_published'])

        date_fields = ['date_created', 'date_modified', 'date_downloaded', 'epoch_date_published']
        for date_field in date_fields:
            if date_field in reports_df.columns:
                print(f"Converting {date_field}...")
                reports_df[date_field] = reports_df[date_field].apply(safe_date_conversion)

        return reports_df
    except Exception as e:
        print(f"Error loading reports: {e}")
        return pd.DataFrame()


def load_classifications_from_mongo(client):
    """Load classifications from MongoDB"""
    try:
        db = client[MONGO_CONFIG['database']]
        classifications_collection = db[MONGO_CONFIG['classifications_collection']]

        classifications_cursor = classifications_collection.find()
        classifications_data = list(classifications_cursor)
        print(f"Loaded {len(classifications_data)} classifications from MongoDB")

        flattened_classifications = []
        for classification in classifications_data:
            flat_record = {
                '_id': classification.get('_id'),
                'namespace': classification.get('namespace'),
                'notes': classification.get('notes', ''),
                'incidents': classification.get('incidents', []),
                'reports': classification.get('reports', []),
                'publish': classification.get('publish', False)
            }

            attributes = classification.get('attributes', [])
            for attr in attributes:
                short_name = attr.get('short_name', '')
                value_json = attr.get('value_json', '')
                try:
                    parsed_value = json.loads(value_json)
                    flat_record[short_name] = parsed_value
                except Exception:
                    flat_record[short_name] = value_json

            flattened_classifications.append(flat_record)

        classifications_df = pd.DataFrame(flattened_classifications)

        if 'Beginning Date' in classifications_df.columns:
            print("Converting classification beginning dates...")
            classifications_df['beginning_date'] = classifications_df['Beginning Date'].apply(
                parse_classification_date
            )

        if 'Ending Date' in classifications_df.columns:
            print("Converting classification ending dates...")
            classifications_df['ending_date'] = classifications_df['Ending Date'].apply(
                parse_classification_date
            )

        return classifications_df
    except Exception as e:
        print(f"Error loading classifications: {e}")
        return pd.DataFrame()



def safe_date_conversion(date_value: Any) -> Optional[pd.Timestamp]:
    """Safely convert various date representations to pandas Timestamps."""
    if pd.isna(date_value) or date_value is None:
        return None

    try:
        date_str = str(date_value)

        if date_str.startswith("1") and len(date_str) >= 4:
            year_part = date_str[:4]
            if year_part.startswith("10") or year_part.startswith("11"):
                fixed_year = "20" + year_part[1:]
                date_str = fixed_year + date_str[4:]

        return pd.to_datetime(date_str, errors="coerce")
    except Exception as exc:  # pragma: no cover - defensive guard
        print(f"Warning: Could not parse date '{date_value}': {exc}")
        return None


def parse_classification_date(date_str: Any) -> Optional[pd.Timestamp]:
    """Parse classification date strings that may be partial or formatted."""
    if not date_str or pd.isna(date_str):
        return None

    try:
        date_text = str(date_str).strip('"')
        if "/" in date_text:
            parts = date_text.split("/")
            if len(parts) == 2:
                month, year = parts
                return pd.to_datetime(f"{year}-{month.zfill(2)}-01")
            if len(parts) == 3:
                return pd.to_datetime(date_text)
        return pd.to_datetime(date_text, errors="coerce")
    except Exception:
        return None


def parse_mongodb_list_field(field_value: Any) -> List[str]:
    """Normalize MongoDB list-like fields regardless of backing type."""
    if hasattr(field_value, "__len__") and not isinstance(field_value, (str, bytes)):
        if hasattr(field_value, "tolist"):
            return field_value.tolist()
        if isinstance(field_value, list):
            return field_value
        return list(field_value)

    try:
        if pd.isna(field_value):
            return []
    except (ValueError, TypeError):
        if field_value is None:
            return []

    if isinstance(field_value, str):
        try:
            return json.loads(field_value)
        except Exception:
            return [field_value]

    return [str(field_value)]


def merge_classifications_with_incidents(
    incidents_df: pd.DataFrame, classifications_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge incident records with their associated classification metadata."""
    if classifications_df.empty:
        print("No classifications data available for merging")
        return incidents_df

    print("Merging classifications with incidents...")
    incident_classifications: Dict[Any, List[Dict[str, Any]]] = {}

    for classification in classifications_df.to_dict("records"):
        for incident_id in classification.get("incidents", []) or []:
            incident_classifications.setdefault(incident_id, []).append(classification)

    classification_fields = [
        "Intent",
        "Severity",
        "Near Miss",
        "Technology Purveyor",
        "System Developer",
        "AI System Description",
        "AI Techniques",
        "AI Applications",
        "Data Inputs",
        "Harm Type",
        "Harm Distribution Basis",
        "Lives Lost",
        "Financial Cost",
        "Sector of Deployment",
        "Public Sector Deployment",
        "Nature of End User",
        "Level of Autonomy",
        "Relevant AI functions",
        "Physical System",
        "Problem Nature",
        "Full Description",
        "Short Description",
        "Location",
        "Named Entities",
        "Infrastructure Sectors",
        "Laws Implicated",
    ]

    for field in classification_fields:
        column_name = f"classification_{field.replace(' ', '_').lower()}"
        if column_name not in incidents_df.columns:
            incidents_df[column_name] = None

    matched_incidents = 0
    for idx, incident in incidents_df.iterrows():
        incident_id = incident.get("incident_id") or incident.get("_id")
        if incident_id is None and hasattr(idx, "__int__"):
            incident_id = int(idx)
        if incident_id is None:
            incident_id = idx + 1

        if incident_id in incident_classifications:
            classifications = incident_classifications[incident_id]
            matched_incidents += 1
            classification = classifications[0]

            for field in classification_fields:
                field_key = f"classification_{field.replace(' ', '_').lower()}"
                if field in classification:
                    incidents_df.at[idx, field_key] = classification[field]

    print(f"Successfully merged classifications for {matched_incidents} incidents")
    return incidents_df


def categorize_risk_domain_enhanced(row: pd.Series) -> str:
    """Infer a risk domain for an incident with classification context."""
    harm_basis = row.get("classification_harm_distribution_basis")
    if harm_basis:
        harm_basis_text = str(harm_basis).lower()
        if any(term in harm_basis_text for term in ["race", "gender", "age", "religion", "ethnicity"]):
            return "Discrimination and bias"

    harm_type = row.get("classification_harm_type")
    if harm_type:
        harm_type_text = str(harm_type).lower()
        if "privacy" in harm_type_text:
            return "Privacy"
        if any(term in harm_type_text for term in ["safety", "physical", "injury"]):
            return "AI system safety"

    ai_apps = row.get("classification_ai_applications")
    if ai_apps:
        ai_apps_text = str(ai_apps).lower()
        if any(term in ai_apps_text for term in ["facial recognition", "surveillance"]):
            return "Privacy"
        if any(term in ai_apps_text for term in ["autonomous", "self-driving"]):
            return "AI system safety"

    text = f"{row.get('title', '')} {row.get('description', '')}".lower()
    if any(term in text for term in ["bias", "discrimination", "fairness", "racist", "sexist"]):
        return "Discrimination and bias"
    if any(term in text for term in ["privacy", "surveillance", "tracking", "personal data"]):
        return "Privacy"
    if any(term in text for term in ["safety", "accident", "crash", "harm", "injury"]):
        return "AI system safety"
    if any(term in text for term in ["security", "hack", "attack", "breach"]):
        return "Security"
    if any(term in text for term in ["misinformation", "deepfake", "manipulation"]):
        return "Information integrity"
    return "Other"


def categorize_entity_type_enhanced(row: pd.Series) -> str:
    """Determine the entity type involved in an incident."""
    tech_purveyor = row.get("classification_technology_purveyor")
    if tech_purveyor:
        entity_text = " ".join(str(item).lower() for item in parse_mongodb_list_field(tech_purveyor))
        if any(term in entity_text for term in ["government", "police", "fbi", "agency", "department"]):
            return "Government"
        if any(term in entity_text for term in ["google", "microsoft", "amazon", "facebook", "apple", "yandex"]):
            return "Technology company"
        if any(term in entity_text for term in ["research", "university", "academic"]):
            return "Research institution"

    sys_dev = row.get("classification_system_developer")
    if sys_dev:
        entity_text = " ".join(str(item).lower() for item in parse_mongodb_list_field(sys_dev))
        if any(term in entity_text for term in ["government", "police", "fbi", "agency", "department"]):
            return "Government"
        if any(term in entity_text for term in ["google", "microsoft", "amazon", "facebook", "apple", "yandex"]):
            return "Technology company"
        if any(term in entity_text for term in ["research", "university", "academic"]):
            return "Research institution"

    deployers = row.get("deployer_list", [])
    developers = row.get("developer_list", [])
    entity_text = " ".join(str(item).lower() for item in deployers + developers)

    if any(term in entity_text for term in ["government", "police", "fbi", "agency", "department"]):
        return "Government"
    if any(term in entity_text for term in ["google", "microsoft", "amazon", "facebook", "apple", "tech"]):
        return "Technology company"
    if any(term in entity_text for term in ["research", "university", "academic"]):
        return "Research institution"
    return "Private sector"


def extract_region_enhanced(row: pd.Series) -> str:
    """Infer the deployment region of an incident."""
    location = row.get("classification_location")
    if location:
        location_text = str(location).lower()
        if location_text and location_text not in {'""', "null"}:
            if any(term in location_text for term in ["china", "chinese", "beijing", "shanghai", "japan", "korea"]):
                return "Asia"
            if any(term in location_text for term in ["europe", "uk", "germany", "france", "netherlands", "sweden", "russia"]):
                return "Europe"
            if any(term in location_text for term in ["usa", "america", "united states", "canada"]):
                return "North America"

    deployers = row.get("deployer_list", [])
    developers = row.get("developer_list", [])
    entity_text = " ".join(str(item).lower() for item in deployers + developers)

    if any(term in entity_text for term in ["china", "chinese", "beijing", "shanghai", "japan", "korea"]):
        return "Asia"
    if any(term in entity_text for term in ["europe", "uk", "germany", "france", "netherlands", "sweden"]):
        return "Europe"
    if any(term in entity_text for term in ["google", "microsoft", "amazon", "facebook", "tesla", "uber", "usa", "america"]):
        return "North America"
    return "Global"


def create_enhanced_description(
    incident_row: pd.Series, related_reports: Optional[List[Dict[str, Any]]] = None
) -> str:
    """Build a rich textual description combining incident and classification data."""
    title = str(incident_row.get("title", "Untitled incident"))
    description = str(incident_row.get("description", "No description available"))
    date_value = incident_row.get("date", datetime.now())

    context_parts = [
        f"Title: {title}",
        f"Date: {date_value.strftime('%Y-%m-%d') if hasattr(date_value, 'strftime') else str(date_value)}",
        f"Risk Domain: {incident_row.get('Risk Domain', 'Unknown')}",
        f"Risk Subdomain: {incident_row.get('Risk Subdomain', 'Unknown')}",
        f"Entity: {incident_row.get('Entity', 'Unknown')}",
        f"Intent: {incident_row.get('Intent', 'Unknown')}",
        f"Timing: {incident_row.get('Timing', 'Unknown')}",
        f"Region: {incident_row.get('region', 'Unknown')}",
    ]

    classification_data: List[str] = []
    key_classification_fields = {
        "classification_intent": "Intent Classification",
        "classification_severity": "Severity",
        "classification_near_miss": "Near Miss/Harm",
        "classification_ai_system_description": "AI System",
        "classification_ai_techniques": "AI Techniques",
        "classification_ai_applications": "AI Applications",
        "classification_harm_type": "Harm Type",
        "classification_harm_distribution_basis": "Harm Distribution",
        "classification_lives_lost": "Lives Lost",
        "classification_level_of_autonomy": "Autonomy Level",
        "classification_nature_of_end_user": "End User Type",
        "classification_sector_of_deployment": "Deployment Sector",
        "classification_problem_nature": "Problem Nature",
    }

    for field, label in key_classification_fields.items():
        try:
            if hasattr(incident_row.index, '__contains__') and field in incident_row.index:
                value = incident_row[field]
                if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
                    if hasattr(value, "__len__") and len(value) > 0:
                        stringified = [str(item) for item in value if str(item) not in {"", "null", "[]", '""'}]
                        if stringified:
                            classification_data.append(f"{label}: {', '.join(stringified)}")
                elif not pd.isna(value):
                    if isinstance(value, str):
                        if value not in {"", "[]", "null", '""'}:
                            classification_data.append(f"{label}: {value}")
                    elif isinstance(value, (int, float, bool)):
                        classification_data.append(f"{label}: {value}")
                    else:
                        string_value = str(value)
                        if string_value not in {"", "[]", "null", '""'}:
                            classification_data.append(f"{label}: {string_value}")
        except Exception:
            continue

    if classification_data:
        context_parts.append("\nClassification Details:")
        context_parts.extend([f"- {item}" for item in classification_data])

    deployers = incident_row.get("deployer_list", [])
    developers = incident_row.get("developer_list", [])
    harmed_parties = incident_row.get("harmed_parties_list", [])

    try:
        if hasattr(incident_row.index, '__contains__') and "classification_technology_purveyor" in incident_row.index:
            tech_purveyor = incident_row.get("classification_technology_purveyor")
            if not pd.isna(tech_purveyor):
                deployers.extend(parse_mongodb_list_field(tech_purveyor))
    except (ValueError, TypeError):
        pass

    try:
        if hasattr(incident_row.index, '__contains__') and "classification_system_developer" in incident_row.index:
            sys_dev = incident_row.get("classification_system_developer")
            if not pd.isna(sys_dev):
                developers.extend(parse_mongodb_list_field(sys_dev))
    except (ValueError, TypeError):
        pass

    try:
        if hasattr(incident_row.index, '__contains__') and "classification_named_entities" in incident_row.index:
            named_ent = incident_row.get("classification_named_entities")
            if not pd.isna(named_ent):
                named_entities = parse_mongodb_list_field(named_ent)
                if named_entities:
                    context_parts.append(f"Named Entities: {', '.join(named_entities)}")
    except (ValueError, TypeError):
        pass

    deployers = list(set(deployers)) if deployers else []
    developers = list(set(developers)) if developers else []

    if deployers:
        context_parts.append(f"Deployers: {', '.join(deployers)}")
    if developers:
        context_parts.append(f"Developers: {', '.join(developers)}")
    if harmed_parties:
        context_parts.append(f"Harmed parties: {', '.join(harmed_parties)}")

    full_desc = incident_row.get("classification_full_description")
    if full_desc and not pd.isna(full_desc):
        full_text = str(full_desc).strip('"')
        if full_text and full_text not in {description, "", "null"}:
            context_parts.append(f"\nDetailed Classification Description: {full_text}")

    short_desc = incident_row.get("classification_short_description")
    if short_desc and not pd.isna(short_desc):
        short_text = str(short_desc).strip('"')
        if short_text and short_text not in {description, "", "null"}:
            context_parts.append(f"Short Classification Description: {short_text}")

    if related_reports:
        context_parts.append("\nRelated Reports:")
        for report in related_reports[:3]:
            report_title = report.get("title", "Untitled report")
            report_text = report.get("text", report.get("description", ""))[:200]
            context_parts.append(f"- {report_title}: {report_text}...")

    enhanced_desc = "\n".join(context_parts)
    return f"{enhanced_desc}\n\nMain Description: {description}"


def get_related_reports(
    incident_row: pd.Series, reports_df: pd.DataFrame
) -> List[Dict[str, Any]]:
    """Retrieve reports linked to an incident via identifiers."""
    if reports_df.empty:
        return []

    report_ids = incident_row.get("reports", [])
    if report_ids and "report_number" in reports_df.columns:
        related = reports_df[reports_df["report_number"].isin(report_ids)]
        return related.to_dict("records")

    return []


__all__ = [
    "connect_to_mongodb",
    "load_incidents_from_mongo",
    "load_reports_from_mongo",
    "load_classifications_from_mongo",
    "safe_date_conversion",
    "parse_classification_date",
    "parse_mongodb_list_field",
    "merge_classifications_with_incidents",
    "categorize_risk_domain_enhanced",
    "categorize_entity_type_enhanced",
    "extract_region_enhanced",
    "create_enhanced_description",
    "get_related_reports",
    "MONGO_CONFIG",
]
