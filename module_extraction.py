# Module d'extraction
# clé API de thunderforest  https://www.thunderforest.com/
API_KEY_thunderforest = "XXXXXXXXXXXXXXXXXXXXXXXXXXXX" # à renseigner

import numpy as np
import os
import sys
from pathlib import Path
import pandas as pd
import datetime
import json
import requests
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go              # pour la carte de localisation
from matplotlib import colors as mcolors
colors = mcolors._colors_full_map
from termcolor import colored 
import importlib
import importlib.util
from pathlib import Path

# à supprimer
# urls = construire_dicts_urls(année_en_cours)
# url_M1_département = urls['M1']
# url_M2_département = urls['M2']
# url_M3_département = urls['M3']
# url_N1_département = urls['N1']
# url_N2_département = urls['N2']
# url_N3_département = urls['N3']

def reload_module_extraction(dossier=None):
    """
    Recharge le module module_extraction de façon robuste
    
    Entrée :
        * dossier : str, optional. Chemin vers le dossier contenant le module
        
    Sortie :
        * module ou None. Le module rechargé ou None en cas d'erreur
    """
    # Chemin explicite
    if dossier is None:
        module_dir = Path("d:")/ "Ingénierie-privée" /  "Météo-France"  #r"d:\Ingénierie-privée\Météo-France"
    else:
        module_dir = Path(dossier)
    
    module_path = module_dir / 'module_extraction.py' #os.path.join(module_dir, 'module_extraction.py')
    
    # Vérification de l'existence du fichier
    if not module_path.exists():
        print(f"ERREUR: {module_path} n'existe pas")
        return None
    
    try:
        # Ajouter le dossier au path si nécessaire
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        
        # Méthode 1: Si le module n'a jamais été importé
        if 'module_extraction' not in sys.modules:
            spec = importlib.util.spec_from_file_location("module_extraction", module_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["module_extraction"] = module
            spec.loader.exec_module(module)
            print("Module importé pour la première fois.")
        
        # Méthode 2: Si le module existe déjà, le recharger
        else:
            # Supprimer complètement le module du cache
            if 'module_extraction' in sys.modules:
                del sys.modules['module_extraction']
            
            # Nettoyer aussi les sous-modules si ils existent
            modules_to_remove = [name for name in sys.modules.keys() 
                               if name.startswith('module_extraction.')]
            for module_name in modules_to_remove:
                del sys.modules[module_name]
            
            # Reimporter complètement
            spec = importlib.util.spec_from_file_location("module_extraction", module_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["module_extraction"] = module
            spec.loader.exec_module(module)
            print("Module rechargé avec succès.")
        
        # Mettre à jour les globals() du notebook
        globals()['module_extraction'] = sys.modules['module_extraction']
        
        return sys.modules['module_extraction']
        
    except Exception as e:
        print(f"ERREUR lors du rechargement: {e}")
        return None



def tracer_carte_france(metadata_MF, hover_name, hover_data, params_dict):
    zoom        = params_dict['zoom']
    height      = params_dict['height']
    width       = params_dict['width']
    map_style   = params_dict['map_style']
    map_center  = params_dict['map_center']
    title       = params_dict['title']
    export      = params_dict['export']
    export_dic  = params_dict['export_dict']
    couche      = params_dict['map_layers']   # corrigé : params_dict, pas parameters_dict
    couleur     = params_dict['couleur']
    showlegend  = params_dict['showlegend']
    margin      = params_dict['margin']
    marker_size = params_dict['marker_size']
    latitude    = params_dict['latitude']
    longitude   = params_dict['longitude']
    couche_tuiles = {
        "OpenTopoMap": {
            "below": "traces",
            "sourcetype": "raster",
            "sourceattribution": "OpenTopoMap",
            "source": ["https://a.tile.opentopomap.org/{z}/{x}/{y}.png"],
        },
        "ThunderForest": {
            "below": "traces",
            "sourcetype": "raster",
            "sourceattribution": "ThunderForest",
            "source": [f"https://tile.thunderforest.com/outdoors/{{z}}/{{x}}/{{y}}.png?apikey={API_KEY_thunderforest}"],
        },
        "IGN_topo": {
            "below": "traces",
            "sourcetype": "raster",
            "source": [
                "https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile"
                "&VERSION=1.0.0&LAYER=GEOGRAPHICALGRIDSYSTEMS.PLANIGNV2"
                "&STYLE=normal&FORMAT=image/png&TILEMATRIXSET=PM"
                "&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}"
            ],
        },
        "IGN_satellite": {
            "below": "traces",
            "sourcetype": "raster",
            "source": [
                "https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile"
                "&VERSION=1.0.0&LAYER=ORTHOIMAGERY.ORTHOPHOTOS"
                "&STYLE=normal&FORMAT=image/jpeg&TILEMATRIXSET=PM"
                "&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}"
            ],
        },
        }

    # GeoJSON chargé côté Python et embarqué dans la figure -> pas de CORS dans le navigateur
    if not hasattr(tracer_carte_france, '_dept_geojson'):
        import requests as _req
        tracer_carte_france._dept_geojson = _req.get(
            "https://france-geojson.gregoiredavid.fr/repo/departements.geojson"
        ).json()
    couche_departements = {
        "below": "traces",
        "sourcetype": "geojson",
        "source": tracer_carte_france._dept_geojson,
        "type": "line",
        "color": "rgba(0, 0, 0, 0.9)",
        "line": {"width": 1},
    }

    all_layers = [couche_tuiles[couche], couche_departements]

    fig = px.scatter_map(metadata_MF, lat=latitude, lon=longitude,
                         hover_name=hover_name, hover_data=hover_data,
                         zoom=zoom, height=height, width=width,
                         color=couleur)

    if couleur is None:
        fig.update_traces(marker=dict(color='red', size=marker_size))
    else:
        fig.update_traces(marker=dict(size=marker_size))
    if marker_size is not None:
        fig.update_traces(
            marker=go.scattermap.Marker(
                size    = 10,
                opacity = 1.0
            )
        )
    fig.update_layout(
        title_text  = title,
        title_x     = 0.5,
        hoverlabel  = dict(bgcolor='rgb(255,255,255)'),
        map_style   = map_style,
        map_zoom    = zoom,
        map_center  = map_center,
        map_layers  = all_layers,   # tuiles + départements en un seul appel
        margin      = margin, 
        showlegend  = showlegend
    )

    if export:
        _w = fig.layout.width
        _h = fig.layout.height
        file_name = export_dic['nom_fichier'] + '.png'
        file_png  = export_dic['répertoire'] / file_name
        fig.write_image(file_png, scale = export_dic['scale'], width = _w, height = _h)
        print(f"Export de {file_name} vers {export_dic['répertoire']}.")

    return fig

def fn_latitude(x):
    y = x['positions']
    if len(y)>0:
        return float(y[-1]['latitude'])
    else:
        return 0
    
def fn_longitude(x):
    y = x['positions']
    if len(y)>0:
        return float(y[-1]['longitude'])
    else:
        return 0

def fn_commune(x):
    if 'nom' in x.keys():
        return x['nom'].strip()
    else:
        return None
    
def fn_numero(x):
    if 'id' in x.keys():
        return str(x['id'])
    else:
        return None
    
def fn_lieudit(x):
    if 'lieuDit' in x.keys():
        return x['lieuDit'].strip()
    else:
        return None
    
def fn_altitude(x):
    y = x['positions']
    if len(y)>0:
        return int(y[-1]['altitude'] )
    else:
        return 0
def fn_change(x):
    y = x['positions']
    return len(y)

#-------------------------------------------
# Fonctions 
#-------------------------------------------
def lecture_bool(x:bool): 
      """
      transforme un booléen en mot "oui" ou "non"
      """
      if x: return "oui"
      else:
            return "non"

# fonction pour télécharger les fichiers MF
def télécharger_fichier(url, nom_fichier, répertoire_principal, work_verbosity = True):
    """
    télécharge le fichier situé l'adresse url et le place dans le 
    répertoire principale avec le nom "nom_fichier"
    La variable verbosity permet de savoir s'il y a des erreurs.
    """
    fichier = os.path.join(répertoire_principal, nom_fichier) + '.gz'
    if os.path.isfile(fichier):
        if work_verbosity: print("Rien à faire. Le fichier est déjà présent.")
    else:
        requête_données = requests.get(url)
        if requête_données.status_code == 200:
            with open(fichier,mode='wb') as file:
             file.write(requête_données.content)
            if work_verbosity: print('Téléchargement : ', fichier)    
        else:
            print(colored("Le lien "+ url+ "ne marche pas !",'red'))
            sys.exit()

# fonction pour décompresser les fichiers MF
def décompresser_archive(archive, répertoire_principal, work_verbosity = True):
    """
    décompresse l'archive
    """
    import gzip
    from pathlib import Path
    fichier             =  répertoire_principal / (archive + '.gz')
    fichier_décompressé =  répertoire_principal / (archive + '.csv')
    if os.path.isfile(fichier_décompressé):
        if work_verbosity: print("Rien à faire. Le fichier est déjà présent.")
    else:
        if os.path.exists(fichier):
            with gzip.open(fichier, mode='rb') as f_in:
                with open(fichier_décompressé, mode='wb') as f_out:
                    f_out.write(f_in.read())
            if work_verbosity: print('Archive décompressée : ', fichier)
        else:
            print(colored("L'archive ", fichier, " n'est pas présente !",'red'))
            print("Téléchargez tout d'abord l'archive...")

def Lire_liste_postes(département,metadata_MF,liste_postes_alpes):
    """
    fournit la liste des postes et de leurs caractéristiques dans un département.
    liste_tous_postes correspond à la liste dans le fichier liste_postes_alpes tandis que
    selection correspond à la liste dans le fichier des postes climatos
    """
    selection        = metadata_MF[metadata_MF['num_poste'].str.startswith(département)]
    selection_nivo   = liste_postes_alpes[liste_postes_alpes['poste'].str.startswith(département)]
    df1 = pd.DataFrame()
    df1['nom'] = selection['nom_usuel'].apply(lambda x: x.title())
    df1['numéro'] = selection['num_poste']
    df1['commune'] = selection['commune'].apply(lambda x: x.title())
    df1['alt'] = selection['alti']
    df1['lat'] = selection['lat']
    df1['lon'] = selection['lon']
    df1['type'] = 'poste MF'
    df2 = pd.DataFrame()
    df2['nom'] = selection_nivo['nom']
    df2['commune'] = selection_nivo['nom']
    df2['numéro'] = selection_nivo['poste']
    df2['alt'] = selection_nivo['alt']
    df2['lat'] = selection_nivo['lat']
    df2['lon'] = selection_nivo['lon']
    df2['type'] = 'nivo'
    if not df2['numéro'].eq(df1['numéro']).any():
        liste_tous_postes = pd.concat([df1,df2], ignore_index=True)
    liste_tous_postes = liste_tous_postes.sort_values(by=['nom'])
    liste_tous_postes['commune'] = liste_tous_postes['commune'].apply(lambda x: x.replace('_Nivo',''))
    liste_tous_postes = liste_tous_postes.reset_index(drop = True)
    return liste_tous_postes, selection



def trouver_période_continue(liste):
    """ 
    fournit l'ensemble des sous-listes telles que chacun des éléments est séparé de l'élément par 1.
    Cette fonction permet de déterminer dans la série d'années avec des cumuls non nuls les séries continues
    d'années.
    """
    res, last = [[]], None
    for x in liste:
        if last is None or abs(last - x) ==1:
            res[-1].append(x)
        else:
            res.append([x])
        last = x
    return res


def convertir_numpy_types(obj):
    """
    Convertir récursivement les types NumPy en types Python natifs pour l'exportation en YAML.
    """
    if isinstance(obj, np.generic):
        return obj.item()  # Convertir scalaire numpy en type Python  
    elif isinstance(obj, dict):
        return {k: convertir_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convertir_numpy_types(v) for v in obj]
    return obj



def ajouter_information(text,lignes_sortie, afficher = True):
    if afficher: print(text)
    lignes_sortie.append(text)


# ---------------------------------------------------------------------------
# Chargement des données depuis le site Météo-France à partir d'un fichier
# de configuration YAML.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Constantes départementales
# ---------------------------------------------------------------------------

AN_DÉBUT_DÉPARTEMENT = {
    '01': '1852', '04': '1872', '05': '1877', '06': '1877',
    '09': '1872', '25': '1852', '26': '1877', '31': '1809',
    '38': '1872', '64': '1865', '65': '1863', '66': '1850',
    '73': '1871', '74': '1876',
}

LISTE_DÉPARTEMENTS = frozenset(AN_DÉBUT_DÉPARTEMENT.keys())

NOMS_DÉPARTEMENT = {
    '01': 'Ain',
    '04': 'Alpes-de-Haute-Provence',
    '05': 'Hautes-Alpes',
    '06': 'Alpes-Maritimes',
    '09': 'Ariège',
    '25': 'Doubs',
    '26': 'Drôme',
    '31': 'Haute-Garonne',
    '38': 'Isère',
    '64': 'Pyrénées-Atlantiques',
    '65': 'Hautes-Pyrénées',
    '66': 'Pyrénées-Orientales',
    '73': 'Savoie',
    '74': 'Haute-Savoie',
}


def construire_dicts_urls(année_en_cours=None):
    """
    Construit les six dictionnaires d'URL Météo-France (M1/M2/M3 et N1/N2/N3)
    pour tous les départements de LISTE_DÉPARTEMENTS.

    Paramètre
    ---------
    année_en_cours : int, optionnel
        Année de référence (défaut : année calendaire actuelle).

    Retourne
    --------
    dict avec les clés 'M1', 'M2', 'M3', 'N1', 'N2', 'N3',
    chacune étant un dict {code_département: url}.
    """
    if année_en_cours is None:
        année_en_cours = datetime.date.today().year
    base   = "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/QUOT/Q_"
    suffixe = f'_latest-{année_en_cours - 1}-{année_en_cours}'
    prev    = str(année_en_cours - 2)
    résultat = {k: {} for k in ('M1', 'M2', 'M3', 'N1', 'N2', 'N3')}
    for d in LISTE_DÉPARTEMENTS:
        a0 = AN_DÉBUT_DÉPARTEMENT[d]
        résultat['M1'][d] = f'{base}{d}_{a0}-1949_RR-T-Vent.csv.gz'
        résultat['M2'][d] = f'{base}{d}_previous-1950-{prev}_RR-T-Vent.csv.gz'
        résultat['M3'][d] = f'{base}{d}{suffixe}_RR-T-Vent.csv.gz'
        résultat['N1'][d] = f'{base}{d}_{a0}-1949_autres-parametres.csv.gz'
        résultat['N2'][d] = f'{base}{d}_previous-1950-{prev}_autres-parametres.csv.gz'
        résultat['N3'][d] = f'{base}{d}{suffixe}_autres-parametres.csv.gz'
    return résultat


def _construire_urls(département, année_en_cours):
    """Construit les URL des archives quotidiennes pour un département."""
    base = "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/QUOT/Q_"
    d    = département
    a0   = AN_DÉBUT_DÉPARTEMENT.get(d, '1950')
    prev = str(année_en_cours - 2)
    suf  = f'_latest-{année_en_cours - 1}-{année_en_cours}'
    return {
        'M1': f'{base}{d}_{a0}-1949_RR-T-Vent.csv.gz',
        'M2': f'{base}{d}_previous-1950-{prev}_RR-T-Vent.csv.gz',
        'M3': f'{base}{d}{suf}_RR-T-Vent.csv.gz',
        'N1': f'{base}{d}_{a0}-1949_autres-parametres.csv.gz',
        'N2': f'{base}{d}_previous-1950-{prev}_autres-parametres.csv.gz',
        'N3': f'{base}{d}{suf}_autres-parametres.csv.gz',
    }


def charger_données(config: dict):
    """
    Télécharge et charge les données quotidiennes MF pour un poste.

    Paramètre
    ---------
    config : dict
        Contenu d'un fichier YAML de configuration (voir config_74080001.yaml).

    Retourne
    --------
    données_pluie : DataFrame  (ou None si non demandé)
    données_neige : DataFrame  (ou None si non demandé)
    répertoire_résultats : str
    existe_HN : bool (peut être mis à False si toutes les valeurs sont nulles)
    """
    import datetime

    poste         = config['poste']
    cfg_données   = config['données']
    répertoires   = config['répertoires']

    poste_choisi       = str(poste['id'])
    département        = str(poste['département'])
    nom_poste          = str(poste['nom'])
    répertoire_principal = str(répertoires['principal'])
    répertoire_postes    = str(répertoires['postes'])

    date_début  = str(cfg_données['date_début'])
    existe_pluie = bool(cfg_données.get('existe_pluie', False))
    existe_neige = bool(cfg_données.get('existe_neige', False))
    existe_HN    = bool(cfg_données.get('existe_HN',    False))
    existe_TN    = bool(cfg_données.get('existe_TN',    False))
    existe_TM    = bool(cfg_données.get('existe_TM',    False))
    existe_TX    = bool(cfg_données.get('existe_TX',    False))

    année_en_cours = datetime.date.today().year
    an_début = int(datetime.datetime.strptime(date_début, '%Y-%m-%d').strftime('%Y'))

    # Choix des archives selon la période
    if 1950 <= an_début <= année_en_cours - 2:
        clés_pluie = ['M2', 'M3']
        clés_neige = ['N2', 'N3']
        préfixe_pluie = lambda i: f'archives_pluie_{i + 1}_{département}'
        préfixe_neige = lambda i: f'archives_neige_{i + 1}_{département}'
    else:  # an_début <= 1949
        clés_pluie = ['M1', 'M2', 'M3']
        clés_neige = ['N1', 'N2', 'N3']
        préfixe_pluie = lambda i: f'archives_pluie_{i}_{département}'
        préfixe_neige = lambda i: f'archives_neige_{i}_{département}'

    from pathlib import Path

    urls = _construire_urls(département, année_en_cours)
    rép  = Path(répertoire_principal)

    def _charger(clés, préfixe_fn, colonnes):
        from tqdm import tqdm as _tqdm
        df = pd.DataFrame()
        for i, clé in _tqdm(list(enumerate(clés))):
            archive = préfixe_fn(i)
            télécharger_fichier(urls[clé], archive, str(rép))
            décompresser_archive(archive, rép)
            fichier = rép / (archive + '.csv')
            données = pd.read_csv(
                fichier, sep=';', encoding='utf-8',
                parse_dates=['AAAAMMJJ'], dtype={'NUM_POSTE': str},
            )
            données['NUM_POSTE'] = données['NUM_POSTE'].str.strip()
            sél = données[données['NUM_POSTE'] == poste_choisi]
            cols = [c for c in colonnes if c in données.columns]
            df = pd.concat([df, sél[cols].copy()])
        return df

    # ---- Précipitations / températures ----
    données_pluie = None
    if existe_pluie:
        df = _charger(clés_pluie, préfixe_pluie, ['AAAAMMJJ', 'RR', 'TM', 'TN', 'TX'])
        données_pluie = df.rename(columns={'AAAAMMJJ': 'date', 'RR': 'pluie'})
    elif existe_TM or existe_TN or existe_TX:
        df = _charger(clés_pluie, préfixe_pluie, ['AAAAMMJJ', 'TM', 'TN', 'TX'])
        données_pluie = df.rename(columns={'AAAAMMJJ': 'date'})

    if données_pluie is not None and existe_TX and existe_TN and not existe_TM:
        tn = pd.to_numeric(données_pluie['TN'], errors='coerce')
        tx = pd.to_numeric(données_pluie['TX'], errors='coerce')
        données_pluie['TM'] = (tn + tx) / 2

    # ---- Neige ----
    données_neige = None
    if existe_neige or existe_HN:
        if existe_neige and existe_HN:
            cols = ['AAAAMMJJ', 'HNEIGEF', 'NEIGETOTX']
            renames = {'AAAAMMJJ': 'date', 'HNEIGEF': 'neige', 'NEIGETOTX': 'HN'}
        elif existe_neige:
            cols = ['AAAAMMJJ', 'HNEIGEF']
            renames = {'AAAAMMJJ': 'date', 'HNEIGEF': 'neige'}
        else:  # only existe_HN
            cols = ['AAAAMMJJ', 'NEIGETOTX']
            renames = {'AAAAMMJJ': 'date', 'NEIGETOTX': 'HN'}

        df = _charger(clés_neige, préfixe_neige, cols)
        données_neige = df.rename(columns=renames)

        if existe_HN and 'HN' in données_neige.columns:
            if np.nanmax(np.array(données_neige['HN'])) == 0:
                print(colored(
                    "Les données relatives à l'épaisseur du manteau neigeux "
                    "sont renseignées, mais toutes nulles...", 'red'
                ))
                existe_HN = False

    # ---- Répertoire des résultats ----
    nom_poste_plein      = nom_poste.replace(' ', '_')
    répertoire_résultats = str(Path(répertoire_postes) / f'{nom_poste_plein}_{poste_choisi}')
    os.makedirs(répertoire_résultats, exist_ok=True)
    print(f'Répertoire des résultats : {répertoire_résultats}')
    os.chdir(répertoire_résultats)

    return données_pluie, données_neige, répertoire_résultats, existe_HN