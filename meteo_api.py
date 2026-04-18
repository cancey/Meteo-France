import requests
from requests.auth import HTTPBasicAuth
import json
import logging
from typing import Optional, Dict, Any, List
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from collections import defaultdict
from matplotlib.patches import Patch
import yaml
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pytz
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
import pandas as pd
import time
####################################
## Auteur : Christophe Ancey      ##
## Date : juin 2025               ##
## Mise à jour : 3 septembre 2025 ##
####################################


# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Configuration (à adapter) ===
# CLIENT_ID = "LV4DCivq1F3zfDGcMNO4ODhnvS0a"
# CLIENT_SECRET = "POG5v7M41eCCyNu1FxfiRVMv458a"

# from pathlib import Path
# dossier = Path('home/ancey/XX')
# fichier_yaml = dossier  

# ======================
# SECTION 1: Config 
# ======================

dossier = '/home/ancey/météo'
fichier_yaml = dossier + r'\configuration_api_météo-france.yaml'

with open(fichier_yaml, 'r',encoding='utf-8') as file:
        codes = yaml.safe_load(file)
CLIENT_ID     = codes['CLIENT_ID']
CLIENT_SECRET = codes['CLIENT_SECRET']

# Pour obtenir le CLIENT_ID et le CLIENT_SECRET, il faut aller sur sa page pour générer un token OAuth2
# https://portail-api.meteofrance.fr/web/fr/token-generation
# Il y a une commande curl, . ex. :
# curl -k -X POST https://portail-api.meteofrance.fr/token -d "grant_type=client_credentials" -H "Authorization: Basic TFY0RENpdnExRjN6ZkRHY01OTzRPRGhudlMwYTpQT0c1djdNNDFlQ0N5TnUxRnhmaVJWTXY0NThh"
# ensuite dans un terminal bash, on copie le mot de passe codé situé après Basic, p. ex. ici on exécute la commande :
# echo TFY0RENpdnExRjN6ZkRHY01OTzRPRGhudlMwYTpQT0c1djdNNDFlQ0N5TnUxRnhmaVJWTXY0NThh | base64 -d
# ce qui affiche deux valeurs séparées par ":"
# LV4DCivq1F3zfDGcMNO4ODhnvS0a:POG5v7M41eCCyNu1FxfiRVMv458a
# La première valeur "LV4DCivq1F3zfDGcMNO4ODhnvS0a" est la clé "CLIENT_ID"
# La seconde  "POG5v7M41eCCyNu1FxfiRVMv458a" est la clé "CLIENT_SECRET"

BASE_URL = "https://public-api.meteofrance.fr/public/DPClim/v1"
TOKEN_URL = "https://portail-api.meteofrance.fr/token"

# Configuration des timeouts et retry
REQUEST_TIMEOUT = 30
MAX_RETRIES     = 3

# ----------
# Subsection 
# ---------- 
class MeteoFranceAPIError(Exception):
    """Exception personnalisée pour les erreurs de l'API Météo-France."""
    pass

# get token
def get_token(client_id: str = CLIENT_ID, client_secret: str = CLIENT_SECRET) -> str:
    """
    Récupère un token OAuth2 valide.
    
    Args:
        client_id: Identifiant client OAuth2
        client_secret: Secret client OAuth2
        
    Returns:
        Token d'accès
        
    Raises:
        MeteoFranceAPIError: En cas d'erreur d'authentification
    """
    try:
        response = requests.post(
            TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=HTTPBasicAuth(client_id, client_secret),
            timeout=REQUEST_TIMEOUT,
            verify=True  # Vérification SSL activée
        )
        response.raise_for_status()
        return response.json()["access_token"]
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur lors de l'obtention du token: {e}")
        raise MeteoFranceAPIError(f"Impossible d'obtenir le token d'accès: {e}")


def _make_api_request(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Effectue une requête API avec gestion d'erreurs et retry.
    
    Entrée :
        url: URL de l'endpoint
        params: Paramètres de la requête
        
    Sortie :
        Réponse JSON de l'API
        
    Exception :
        MeteoFranceAPIError: En cas d'erreur API
    """
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                url, 
                headers=headers, 
                params=params,
                timeout=REQUEST_TIMEOUT,
                verify=True  # Vérification SSL activée
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Tentative {attempt + 1}/{MAX_RETRIES} échouée: {e}")
            if attempt == MAX_RETRIES - 1:
                raise MeteoFranceAPIError(f"Erreur API après {MAX_RETRIES} tentatives: {e}")


def get_station_infra_horaire(département: str, parametre: str = "precipitation") -> Dict[str, Any]:
    """
    Récupère les stations avec données infra-horaires pour un département et un paramètre donné.
    
    Entrée :
        département: Code du département
        parametre: Paramètre météorologique (défaut: "precipitation")
        
    Sortie :
        Données des stations
    """
    url = f"{BASE_URL}/liste-stations/infrahoraire-6m"
    params = {"id-departement": département, "parametre": parametre}
    return _make_api_request(url, params)


def get_station_horaire(département: str, parametre: str = "precipitation") -> Dict[str, Any]:
    """
    Récupère les stations avec données horaires pour un département et un paramètre donné.
    
    Entrée :
        département: Code du département
        parametre: Paramètre météorologique (défaut: "precipitation")
        
    Sortie :
        Données des stations
    """
    url = f"{BASE_URL}/liste-stations/horaire"
    params = {"id-departement": département, "parametre": parametre}
    return _make_api_request(url, params)


def get_station_quotidienne(département: str, parametre: str = "precipitation") -> Dict[str, Any]:
    """
    Récupère les stations avec données quotidiennes pour un département et un paramètre donné.
    
    Entrée :
        département: Code du département
        parametre: Paramètre météorologique (défaut: "precipitation")
        
    Sortie :
        Données des stations
    """
    url = f"{BASE_URL}/liste-stations/quotidienne"
    params = {"id-departement": département, "parametre": parametre}
    return _make_api_request(url, params)


def get_station_info(station_id: str, token: Optional[str] = None) -> Dict[str, Any]:
    """
    Récupère les métadonnées pour une station météo donnée.
    
    Entrée :
        station_id : Identifiant de la station
        token : Token d'accès (optionnel, sera généré si non fourni)
        
    Sortie :
        Informations de la station
    """
    if token is None:
        token = get_token()
        
    url = f"{BASE_URL}/information-station"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"id-station": station_id}
    
    try:
        response = requests.get(
            url, 
            headers=headers, 
            params=params,
            timeout=REQUEST_TIMEOUT,
            verify=True
        )
        response.raise_for_status()
        data = response.json()
        return data[0] if isinstance(data, list) and len(data) == 1 else data
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur lors de la récupération des infos pour la station {station_id}: {e}")
        raise MeteoFranceAPIError(f"Impossible de récupérer les informations de la station {station_id}: {e}")





def save_json(data: Dict[str, Any], filename: str) -> None:
    """
    Sauvegarde les données au format JSON.
    
    Entrée :
        data : Données à sauvegarder
        filename : Nom du fichier de destination
    """
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Données sauvegardées dans {filename}")
    except IOError as e:
        logger.error(f"Erreur lors de la sauvegarde: {e}")
        raise


def print_station_summary(data: Dict[str, Any]) -> None:
    """
    Affiche un résumé des informations d'une station.
    
    Entrée :
        data: Données de la station
    """
    print(f"*  Station {data.get('id', 'N/A')} - {data.get('nom', 'N/A')}")
    print(f"* Lieu-dit   : {data.get('lieuDit', 'N/A')}")
    print(f"* Bassin     : {data.get('bassin', 'N/A')}")
    print(f"* Période    : {data.get('dateDebut', 'N/A')} → {data.get('dateFin', 'N/A')}")

    print("*  Positions :")
    positions = data.get("positions", [])
    for pos in positions:
        print(f"   - alt: {pos.get('altitude', 'N/A')} m, lat: {pos.get('latitude', 'N/A')}, lon: {pos.get('longitude', 'N/A')}")

    producteurs = [p.get("nom", "N/A") for p in data.get("producteurs", [])]
    print(f"* Producteurs : {', '.join(producteurs) if producteurs else 'N/A'}")


def print_parametres_table(data: Dict[str, Any]) -> None:
    """
    Affiche un tableau des paramètres disponibles.
    
    Entrée :
        data: Données de la station
    """
    parametres = data.get("parametres", [])
    if not parametres:
        print("* Aucun paramètre disponible.")
        return

    print("\n* Paramètres disponibles :")
    print(f"{'Nom':30} {'Début':12} {'Fin':12}")
    print("-" * 56)

    for p in parametres:
        nom = p.get("nom", "N/A")
        debut = p.get("dateDebut", "N/A")
        fin = p.get("dateFin", "N/A")
        print(f"{nom:30} {debut:12} {fin:12}")


def fournir_json(station_id: str) -> Optional[Dict[str, Any]]:
    """
    Récupère les données JSON d'une station.
    
    Entrées :
        station_id: Identifiant de la station
        
    Sortie :
        Données de la station ou None en cas d'erreur
    """
    try:
        token = get_token()
        data = get_station_info(station_id, token)
        return data
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données pour {station_id}: {e}")
        return None


def graphique_chronologie_paramètres(station_id: str):


    data = fournir_json(station_id)
    # Organize parameters by name and category
    param_dict = defaultdict(list)
    categories = {
        'Température': ['TEMPERATURE', 'TN', 'TX', 'TM', 'DJU'],
        'Précipitation': ['PRECIPITATION', 'RR', 'PLUIE', 'NEIGE', 'HAUTEUR', 'CUMUL'],
        'OTHER': []  # Everything else
    }

    for param in data['parametres']:
        param_name = param['nom']
        start = datetime.strptime(param['dateDebut'], '%Y-%m-%d %H:%M:%S')
        end = param['dateFin']
        end_date = datetime.strptime(end, '%Y-%m-%d %H:%M:%S') if end else datetime.now()
        
        # Determine category
        category = 'Autre'
        for cat, keywords in categories.items():
            if any(keyword in param_name for keyword in keywords):
                category = cat
                break
        
        param_dict[param_name].append({
            'start': start,
            'end': end_date,
            'category': category
        })

    # Prepare the plot
    fig = plt.figure(figsize=(15, 10))
    ax = plt.gca()

    # Color mapping for categories
    colors = {
        'Température': 'red',
        'Précipitation': 'blue',
        'Autre': 'gray'
    }

    # Plot each parameter
    for i, (param_name, periods) in enumerate(param_dict.items()):
        for period in periods:
            ax.barh(
                y=i,
                width=period['end'] - period['start'],
                left=period['start'],
                height=0.8,
                color=colors[period['category']],
                alpha=0.7
            )

    # Format the plot
    ax.set_yticks(range(len(param_dict)))
    ax.set_yticklabels([x.lower() for x in param_dict.keys()])
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.grid(True, axis='x', alpha=0.3)
    plt.title('Measurement Timeline of Meteorological Parameters', pad=20)

    # Create legend
    legend_elements = [Patch(facecolor=color, label=cat) for cat, color in colors.items()]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()
    return fig

def reload_meteo_final():
    import sys
    import os
    
    # Chemin explicite (adaptez selon votre cas)
    module_dir = r"d:\Ingénierie-privée\Météo-France"
    
    # Vérifiez que le module existe
    module_path = os.path.join(module_dir, 'meteo_api.py')
    if not os.path.exists(module_path):
        print(f"ERREUR: {module_path} n'existe pas")
        return False
    
    # Ajoutez au PATH si nécessaire
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    
    # Nettoyage et rechargement
    if 'meteo_api' in sys.modules:
        del sys.modules['meteo_api']
    
    import meteo_api
    globals()['meteo_api'] = meteo_api
    globals()['fournir_json'] = meteo_api.fournir_json
    
    print("Module rechargé avec succès")
    return True



def graphique_chronologie_paramètres_sélectionnés(station_id: str):
    data = fournir_json(station_id)

    selected_params = [
            "CUMUL DE PRECIPITATIONS EN 6 MN",
        "CUMUL DES HAUTEURS DE PRECIPITATIONS",
        "EPAISSEUR DE NEIGE TOTALE HORAIRE",
        "EPAISSEUR DE NEIGE TOTALE RELEVEE A 0600 FU",
        "EPAISSEUR MAXIMALE DE NEIGE",
        "ETP CALCULEE AU POINT DE GRILLE LE PLUS PROCHE",
        "HAUTEUR DE NEIGE FRAICHE TOMBEE EN 24H",
        "HAUTEUR DE PRECIPITATIONS HORAIRE",
        "HAUTEUR DE PRECIPITATIONS QUOTIDIENNE",
        "QUANTITE DE PRECIPITATIONS LORS DE L'EPISODE PLUVIEUX",
        "QUANTITE PRECIP BRUTE",
        "TEMPERATURE MAXIMALE SOUS ABRI HORAIRE",
        "TEMPERATURE MAXIMALE SOUS ABRI QUOTIDIENNE",
        "TEMPERATURE MINIMALE SOUS ABRI HORAIRE",
        "TEMPERATURE MINIMALE SOUS ABRI QUOTIDIENNE",
        "TEMPERATURE MOYENNE SOUS ABRI QUOTIDIENNE",
        'EPAISSEUR DE NEIGE TOTALE RELEVEE A 0600 FU'
    ]

    param_translation = {
        "CUMUL DE PRECIPITATIONS EN 6 MN": "Cumul de précipitations en 6 min",
        "CUMUL DES HAUTEURS DE PRECIPITATIONS": "Cumul des hauteurs de précipitations",
        "EPAISSEUR DE NEIGE TOTALE HORAIRE": "Épaisseur de neige totale horaire",
        "EPAISSEUR DE NEIGE TOTALE RELEVEE A 0600 FU": "Épaisseur de neige totale relevée à 0600 fu",
        "EPAISSEUR MAXIMALE DE NEIGE": "Épaisseur maximale de neige",
        "ETP CALCULEE AU POINT DE GRILLE LE PLUS PROCHE": "ETP calculée au point de grille le plus proche",
        "HAUTEUR DE NEIGE FRAICHE TOMBEE EN 24H": "Hauteur de neige fraîche tombée en 24h",
        "HAUTEUR DE PRECIPITATIONS HORAIRE": "Hauteur de précipitations horaire",
        "HAUTEUR DE PRECIPITATIONS QUOTIDIENNE": "Hauteur de précipitations quotidienne",
        "QUANTITE DE PRECIPITATIONS LORS DE L'EPISODE PLUVIEUX": "Quantité de précipitations lors de l'épisode pluvieux",
        "TEMPERATURE MAXIMALE SOUS ABRI HORAIRE": "Température maximale sous abri horaire",
        "TEMPERATURE MAXIMALE SOUS ABRI QUOTIDIENNE": "Température maximale sous abri quotidienne",
        "TEMPERATURE MINIMALE SOUS ABRI HORAIRE": "Température minimale sous abri horaire",
        "TEMPERATURE MINIMALE SOUS ABRI QUOTIDIENNE": "Température minimale sous abri quotidienne",
        "TEMPERATURE MOYENNE SOUS ABRI QUOTIDIENNE": "Température moyenne sous abri quotidienne"
    }

    # Organize parameters by category
    categories = {
            'Précipitation': ['PRECIPITATIONS', 'HAUTEUR', 'CUMUL', 'QUANTITE'],
            'Neige': ['NEIGE', 'EPAISSEUR','HAUTEUR'],
            'Température': ['TEMPERATURE', 'MAXIMALE', 'MINIMALE', 'MOYENNE'],
            'ETP': ['ETP']
        }

    # Color mapping for categories
    colors = {
    'Précipitation': '#1f77b4',  # Blue
    'Neige': '#17becf',           # Cyan
    'Température': '#d62728',    # Red
    'ETP': '#2ca02c',            # Green
    'Autre': '#7f7f7f'           # Gray
    }

    # =============================================
    # PROCESSING
    # =============================================
    filtered_data = defaultdict(list)
    all_dates = []

    for param in data['parametres']:
        original_name = param['nom']
        if original_name in param_translation:
            translated_name = param_translation[original_name]
            start = datetime.strptime(param['dateDebut'], '%Y-%m-%d %H:%M:%S')
            end = param['dateFin']
            end_date = datetime.strptime(end, '%Y-%m-%d %H:%M:%S') if end else datetime.now()
            
            all_dates.extend([start, end_date])
            
            category = 'Autre'
            for cat, keywords in categories.items():
                if any(keyword.lower() in original_name.lower() for keyword in keywords):
                    category = cat
                    break
                    
            filtered_data[translated_name].append({
                'start': start,
                'end': end_date,
                'category': category
            })

    # Calculate date range with padding
    min_date, max_date = min(all_dates), max(all_dates)
    date_padding = (max_date - min_date) * 0.05
    xlim_start, xlim_end = min_date - date_padding, max_date + date_padding

    # =============================================
    # VISUALIZATION
    # =============================================
    n_params = len(filtered_data)
    fig, ax = plt.subplots(figsize=(14, 0.6*n_params))

    # Plot bars
    for i, (param_name, periods) in enumerate(sorted(filtered_data.items())):
        for period in periods:
            ax.barh(
                y=i,
                width=period['end'] - period['start'],
                left=period['start'],
                height=0.5,
                color=colors[period['category']],
                edgecolor='white',
                alpha=0.8
            )

    # Y-axis formatting
    ax.set_yticks(range(len(filtered_data)))
    ax.set_yticklabels(sorted(filtered_data.keys()))
    plt.yticks(fontsize=10)

    # X-axis formatting with custom grid
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(which='major', axis='x', linestyle='-', linewidth=1.5, alpha=0.5)
    ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.7, alpha=0.3)
    plt.xticks(rotation=0)
    plt.xlim(xlim_start, xlim_end)

    # Title and legend
    plt.title('Chronologie des paramètres météorologiques sélectionnés', pad=20, fontsize=14)
    legend_elements = [Patch(facecolor=color, label=cat) for cat, color in colors.items() if cat != 'Autre']
    legend = ax.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.2),
        ncol=4,
        frameon=False
    )

    # Final layout adjustment
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()
    return fig




def get_parameter_date_range(data, parameter_name):
    """
    Renvoie la date de début la plus ancienne et la date de fin la plus récente pour un paramètre donné, au format ISO 8601.

    Arguments:
        data :   dictionnaire contenant les paramètres météorologiques
        parameter_name : Le nom exact du paramètre à rechercher

    Retourne:
        tuple: (date_de_début_la_plus_ancienne_str, date_de_fin_la_plus_récente_str) au format "AAAA-MM-JJThh:00:00Z"
               Retourne (None, None) si le paramètre n'est pas trouvé
    """
    start_dates = []
    end_dates = []

    local_tz = pytz.timezone('Europe/Zurich')  # e.g., 'Europe/Paris'




    for param in data['parametres']:
        if param['nom'] == parameter_name:
            # Conversion des dates
            start_date = datetime.strptime(param['dateDebut'], '%Y-%m-%d %H:%M:%S')
            local_dt = local_tz.localize(start_date.replace(hour=0, minute=0, second=0))
            start_date_utc = local_dt.astimezone(pytz.UTC)
            start_dates.append(start_date_utc)
            
            # Date de fin
            if param['dateFin']:
                end_date = datetime.strptime(param['dateFin'], '%Y-%m-%d %H:%M:%S')
                local_dt = local_tz.localize(end_date.replace(hour=0, minute=0, second=0))
                end_date_utc = local_dt.astimezone(pytz.UTC)
                #end_date_utc = end_date.replace(hour=23, minute=59, second=59).astimezone(timezone.utc)
            else:
                end_date_utc = datetime.now(local_tz)  #datetime.now(timezone.utc)  # Temps UTC
            end_dates.append(end_date_utc)
    
    if not start_dates:
        return (None, None)
    
    # Conversion au format requis
    def format_date(dt):
        #return dt.strftime('%Y-%m-%dT%H:00:00Z')
        return dt.strftime('%Y-%m-%d')
    
    return (format_date(min(start_dates)), format_date(max(end_dates)))

def ajouter_un_an(input_date_str: str):
    """    Génère deux dates au format ISO 8601 à partir d'une chaîne de date d'entrée.

    Entrée:
        input_date_str: Chaîne de date au format "AAAA-MM-JJ"

    Sortie:
        tuple: (date_iso, date_plus_un_an_moins_une_heure_iso)
               au format "AAAA-MM-JJThh:00:00Z"

    Exception:
        ValueError: Si le format de la date d'entrée est invalide
    """
    try:
        # Analyser la date d'entrée (UTC)
        input_date = datetime.strptime(input_date_str, "%Y-%m-%d").replace(
            hour=0, minute=0, second=0, microsecond=0,
            tzinfo=timezone.utc
        )
        
        # Calculer date + 1 an
        date_plus_one_year = input_date + relativedelta(years=1)
        
        # Soustraire une hezre
        date_plus_one_year_minus_hour = date_plus_one_year - timedelta(hours=1)
        
        # Format des dates
        date_iso = input_date.strftime("%Y-%m-%dT%H:00:00Z")
        date_plus_un_an_iso = date_plus_one_year_minus_hour.strftime("%Y-%m-%dT%H:00:00Z")
        
        return (date_iso, date_plus_un_an_iso)
        
    except ValueError as e:
        raise ValueError(f"Format de date invalide. Il faut une date sous la forme 'YYYY-MM-DD', et non '{input_date_str}'") from e

def get_station_commande(station_id: str,début: str,fin = None,période='quotidienne') -> Optional[Dict[str, Any]]:
    """
    Obtient le numéro de commande pour une station météo donnée.
    
    Entrée :
      - station_id: Identifiant de la station
      - début : date de début au format 'AAAA-MM-JJ'
      - fin : (optionnel) date de fin au format 'AAAA-MM-JJ'. Si 'fin' n'est pas précisé
              alors 'fin' = 'début' + 1 an - 1 h car on ne peut faire des requêtes que
              pour des durées inférieures à un an
      - période : par défaut 'quotidienne'. Argument disponible : 'horaire',
                  'infrahoraire', 'mensuelle' et 'décadaire'
        
    Sortie : données de la commande ou None en cas d'erreur
    """
    dico = {'horaire':'horaire','quotidienne':'quotidienne','infrahoraire':'infrahoraire-6m',
    'mensuelle':'mensuelle',"décadaire":'decadaire'}
    #fiche = fournir_json(station_id)
    #début, fin = get_parameter_date_range(fiche,variable)
    #fin = '1995-06-01T15:00:00Z'
    if fin == None:
        # si pas de date de fin (argument 'fin') définir date_fin comme date_déb + 1 ans
        date_déb, date_fin = ajouter_un_an(début)
    else:
        date_déb = datetime.strptime(début, '%Y-%m-%d').strftime("%Y-%m-%dT%H:00:00Z")
        date_fin = datetime.strptime(fin, '%Y-%m-%d').strftime("%Y-%m-%dT%H:00:00Z")
    try:
        token = get_token()
        url = f"{BASE_URL}/commande-station/"+dico[période]
        headers = {"Authorization": f"Bearer {token}"}
        params = {"id-station": station_id,
        "date-deb-periode":date_déb, "date-fin-periode":date_fin}
        
        response = requests.get(
            url, 
            headers=headers, 
            params=params,
            timeout=REQUEST_TIMEOUT,
            verify=True
        )
        response.raise_for_status()
        data = response.json()
        return data[0] if isinstance(data, list) and len(data) == 1 else data
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données de commande pour {station_id}: {e}")
        return None

def get_commande_complète(station_id: str, date_début: str, date_fin: str, période='quotidienne') -> List[Dict[str, Any]]:
    """
    Récupère les commandes nécessaires pour couvrir toute une période, même si elle dépasse un an.

    Entrée :
        - station_id : identifiant de la station
        - date_début : date de début au format 'AAAA-MM-JJ'
        - date_fin   : date de fin au format 'AAAA-MM-JJ'
        - période    : 'quotidienne', 'horaire', etc. (voir fonction get_station_commande)

    Sortie :
        - Liste de commandes (dictionnaires). Les erreurs sont ignorées.
    """
    commandes = []
    date_d = datetime.strptime(date_début, "%Y-%m-%d")
    date_f = datetime.strptime(date_fin, "%Y-%m-%d")

    delta_max = timedelta(days=365) - timedelta(hours=1)

    while date_d < date_f:
        next_date = min(date_d + delta_max, date_f)
        début_str = date_d.strftime("%Y-%m-%d")
        fin_str = next_date.strftime("%Y-%m-%d")

        commande = get_station_commande(station_id, début_str, fin_str, période=période)
        if commande:
            commandes.append(commande)
        else:
            logger.warning(f"Aucune commande pour la période {début_str} à {fin_str}")

        date_d = next_date + timedelta(hours=1)  # éviter chevauchement

    return commandes

###################################################################################
def get_station_commande_horaire(station_id: str,début: str,fin = None) -> Optional[Dict[str, Any]]:
    """
    Obtient le numéro de commande pour une station météo donnée.
    
    Args:
        station_id: Identifiant de la station
        
    Returns:
        Données de commande ou None en cas d'erreur
    """
    #fiche = fournir_json(station_id)
    #début, fin = get_parameter_date_range(fiche,variable)
    #fin = '1995-06-01T15:00:00Z'
    if fin == None:
        # si pas de date de fin (argument 'fin') définir date_fin comme date_déb + 1 ans
        date_déb, date_fin = ajouter_un_an(début)
    else:
        date_déb = datetime.strptime(début, '%Y-%m-%d').strftime("%Y-%m-%dT%H:00:00Z")
        date_fin = datetime.strptime(fin, '%Y-%m-%d').strftime("%Y-%m-%dT%H:00:00Z")
    try:
        token = get_token()
        url = f"{BASE_URL}/commande-station/horaire"
        headers = {"Authorization": f"Bearer {token}"}
        params = {"id-station": station_id,
        "date-deb-periode":date_déb, "date-fin-periode":date_fin}
        
        response = requests.get(
            url, 
            headers=headers, 
            params=params,
            timeout=REQUEST_TIMEOUT,
            verify=True
        )
        response.raise_for_status()
        data = response.json()
        return data[0] if isinstance(data, list) and len(data) == 1 else data
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données de commande pour {station_id}: {e}")
        return None

# Remplacez votre fonction get_commande_fichier par celle-ci :
def get_commande_fichier(numero: str) -> Optional[Dict[str, Any]]:
    """
    Obtient le fichier de données pour un numéro de commande donné.
   
    Args:
        numero: Numéro de commande
       
    Returns:
        Dictionnaire contenant les données CSV et métadonnées ou None en cas d'erreur
    """
    try:
        token = get_token()
        url = f"{BASE_URL}/commande/fichier"
        headers = {"Authorization": f"Bearer {token}"}
        params = {"id-cmde": numero}
       
        response = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT, verify=True)
        response.raise_for_status()
        
        # Extraire le nom du fichier
        content_disposition = response.headers.get('content-disposition', '')
        filename = 'fichier_meteo.csv'
        if 'filename=' in content_disposition:
            filename = content_disposition.split('filename=')[1].strip('"')
        
        # IMPORTANT : Utiliser response.text pour le contenu CSV
        csv_content = response.text
        
        return {
            'csv_content': csv_content,
            'filename': filename,
            'content_type': response.headers.get('content-type', ''),
            'size': len(csv_content),  # Taille du texte
            'status_code': response.status_code
        }
            
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du fichier pour la commande {numero}: {e}")
        return None

 

def obtenir_série_complète(station_id: str, date_début: str, date_fin: str, période='quotidienne',sélection=True):
    """ 
    Fournit la série de données du poste 'statiion_id' pour la période allant de date_début à date_fin
    * Entrée :
      - station_id : identificant à 8 chiffres de la station
      - date_début : date de début au format 'AAAA-MM-JJ'
      - date_fin   : date de fin au format 'AAAA-MM-JJ'
      - période    : 'quotidienne', 'horaire', etc. (voir fonction get_station_commande)
      - sélection  : booléen. Si sélection = True, alors on ne s'intéresse aux variables 'RR', 'TN', 'TX', 'HNEIGEF', 'NEIGETOTX', 'RR1'
                     sinon on retourne tout le tableau sauf les colonnes vides
    * Sortie : data frame
    """
    codes_brut = get_commande_complète(station_id, date_début, date_fin, période=période)
    #print(f"station {station_id}. Granulométrie : {période}. Début {date_début}")
    # Extraction des numéros de commande
    codes = [
        cmd.get('elaboreProduitAvecDemandeResponse', {}).get('return') 
        for cmd in codes_brut if cmd is not None
    ]

    df_list = []
    for numéro_commande in tqdm(codes):
        # pour les périodes horaires ou infra, le temps de chargement peut être long.
        # on répète la requête jusqu'à obtention du fichier
        fichier = None
        max_tentatives = 5
        délai = 2  # secondes
        
        for tentative in range(max_tentatives):
            fichier = get_commande_fichier(numéro_commande)
            
            if fichier and 'csv_content' in fichier and len(fichier['csv_content']) > 0:
                break  # Fichier prêt !
            
            if tentative < max_tentatives - 1:  # Pas la dernière tentative
                time.sleep(délai)
                délai *= 1.5  # Augmentation progressive du délai
    
        if not fichier or 'csv_content' not in fichier or len(fichier['csv_content']) == 0:
            print(f"Fichier non disponible après {max_tentatives} tentatives pour commande {numéro_commande}")
            continue
        #fichier = get_commande_fichier(numéro_commande)
        #print(f"taille de fichier = {len(fichier)}")
        # if not fichier or 'csv_content' not in fichier:
        #     print("Je ne trouve pas le contenu du fichier !")
        #     continue

        try:
                        # Lecture du contenu CSV
            df = pd.read_csv(
                StringIO(fichier['csv_content']),
                sep=';',
                decimal=',',
                engine='python',
                na_values=[''],
                skipinitialspace=True
            )
 
            # Vérification des colonnes nécessaires
            if sélection:
                colonnes_voulues = ['DATE', 'RR', 'TN', 'TX', 'HNEIGEF', 'NEIGETOTX','RR1']
                colonnes_existantes = [col for col in colonnes_voulues if col in df.columns]
                df = df[colonnes_existantes]
            else:
                # Suppression éventuelle des colonnes totalement vides
                df = df.dropna(how='all', axis=1)
            # Transformation de la date
            # if période != 'horaire':
            #     if 'DATE' in df.columns:
            #         df['DATE'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d').dt.strftime('%Y-%m-%d')
            #     else:
            #         df['DATE'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d%H').dt.strftime('%Y-%m-%d %:00:00')
            # Transformation de la date
            if période != 'horaire':
                if 'DATE' in df.columns:
                    df['DATE'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d').dt.strftime('%Y-%m-%d')
            else:  # période == 'horaire'
                if 'DATE' in df.columns:
                    df['DATE'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d%H').dt.strftime('%Y-%m-%d %H:00:00')

            df_list.append(df)

        except Exception as e:
            logger.warning(f"Erreur lors du traitement de la commande {numéro_commande}: {e}")
            continue

    # Concaténation et nettoyage final
    if not df_list:
        return pd.DataFrame()  # retour vide si aucun résultat

    df_final = pd.concat(df_list, ignore_index=True)

    # Tri par date si la colonne existe
    if 'DATE' in df_final.columns:
        df_final = df_final.sort_values(by='DATE').reset_index(drop=True)

    return df_final

def afficher_tableau_3_colonnes(liste, largeur_colonne = 60):
    """ 
    affiche une liste sous la forme d'un tableau de trois colonnes
    Entrées :
        * liste
        * largeur_colonne : en caractères, par défaut largeur_colonne = 60 
    """
    liste_triée = sorted(liste)
    print(f"{'  1':<{largeur_colonne}} | {'  2':<{largeur_colonne}} | {'  3':<{largeur_colonne}}")
    print("-" * (largeur_colonne * 3 + 6))
    
    for i in range(0, len(liste), 3):
        ligne = liste_triée[i:i+3]
        col1 = ligne[0] if len(ligne) > 0 else ''
        col2 = ligne[1] if len(ligne) > 1 else ''
        col3 = ligne[2] if len(ligne) > 2 else ''
        
        print(f"{col1:<{largeur_colonne}} | {col2:<{largeur_colonne}} | {col3:<{largeur_colonne}}")
# %%
