import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
from datetime import datetime
import sys
import os
import logging
# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from scipy import stats                           # lois de probabilité (valeurs extrêmes)
from scipy.stats import gumbel_r, genextreme      # valeurs extrêmes
 
import importlib
import importlib.util
import yaml
from pathlib import Path
 
from IPython import get_ipython

def reload_valeurs_extremes(dossier=None):
    """
    Recharge le module module_valeurs_extremes de façon robuste
    
    Entrée :
        * dossier : str, optional. Chemin vers le dossier contenant le module
        
    Sortie :
        * module ou None. Le module rechargé ou None en cas d'erreur
    """
    # Chemin explicite
    if dossier is None:
        module_dir = r"d:\Ingénierie-privée\Météo-France"
    else:
        module_dir = dossier
    
    module_path = os.path.join(module_dir, 'module_valeurs_extremes.py')
    
    # Vérification de l'existence du fichier
    if not os.path.exists(module_path):
        print(f"ERREUR: {module_path} n'existe pas")
        return None
    
    try:
        # Ajouter le dossier au path si nécessaire
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        
        # Méthode 1: Si le module n'a jamais été importé
        if 'module_valeurs_extremes' not in sys.modules:
            spec = importlib.util.spec_from_file_location("module_valeurs_extremes", module_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["module_valeurs_extremes"] = module
            spec.loader.exec_module(module)
            print("Module importé pour la première fois.")
        
        # Méthode 2: Si le module existe déjà, le recharger
        else:
            # Supprimer complètement le module du cache
            if 'module_valeurs_extremes' in sys.modules:
                del sys.modules['module_valeurs_extremes']
            
            # Nettoyer aussi les sous-modules si ils existent
            modules_to_remove = [name for name in sys.modules.keys() 
                               if name.startswith('module_valeurs_extremes.')]
            for module_name in modules_to_remove:
                del sys.modules[module_name]
            
            # Reimporter complètement
            spec = importlib.util.spec_from_file_location("module_valeurs_extremes", module_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["module_valeurs_extremes"] = module
            spec.loader.exec_module(module)
            print("Module rechargé avec succès.")
        
        # Mettre à jour les globals() du notebook
        globals()['module_valeurs_extremes'] = sys.modules['module_valeurs_extremes']
        
        return sys.modules['module_valeurs_extremes']
        
    except Exception as e:
        print(f"ERREUR lors du rechargement: {e}")
        return None

def vérifier_configuration(verbosity=False):
    """ permet de tester si BLAS est chargé (utile pour l'accélération des calculs dans PyMC)"""
    import pytensor
    if verbosity:
        print("blas.ldflags =", pytensor.config.blas__ldflags)
        print("mode =", pytensor.config.mode)
        print("optimizer =", pytensor.config.optimizer)
    return f"blas.ldflags = {pytensor.config.blas__ldflags}",  f"mode = {pytensor.config.mode}",   f"optimizer = {pytensor.config.optimizer}" 


def lister_fonctions_module(module_name):
    """
    Enumérer les fonctions de façon alphatique avec indication des arguments et de la ligne du module 
    où la fonction est définie.
    """
    import ast
    from tabulate import tabulate
    import importlib.util
    try:
        # Tester si le module existe
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            return f"Importation impossible de {module_name} !"
        
        if not spec.origin.endswith('.py'):
            return f"Le module {module_name} n'est pas un module python de la forme *.py"
        
        # Analyser le fichier source
        with open(spec.origin, 'r', encoding='utf-8') as file:
            content = file.read()
            tree = ast.parse(content)
        
        fonctions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # détermine le numéro de ligne
                line_no = node.lineno
                
                # détermine les arguments
                args = [arg.arg for arg in node.args.args]
                if node.args.vararg:
                    args.append(f"*{node.args.vararg.arg}")
                if node.args.kwarg:
                    args.append(f"**{node.args.kwarg.arg}")
                
                sig = f"({', '.join(args)})"
                fonctions.append([node.name, sig, line_no])
        # tri alphabétique
        fonctions.sort(key=lambda x: x[0].lower())
        if fonctions:
            return tabulate(fonctions, headers=['Function', 'Signature', 'Line'])#, tablefmt='grid')
        else:
            return f"Il n'y a pas de fonction dans '{module_name}'"
    except FileNotFoundError:
        return f"Je n'ai pas trouvé '{module_name}'"
    except Exception as e:
        return f"Je ne parviens à analyser : {str(e)}"

# Fonction pour vérifier l'état du module
def check_module_status(paramètres,tester_importation=True, tester_BLAS=True, énumérer_fonctions=True, 
                       information_système=True, sauver_fichier=True, fichier=None,
                       afficher_écran=False,sortie=False):
    """
    Affiche des informations sur l'état du module et optionnellement les sauvegarde dans un fichier.
    
    Entrées:
        * paramètres du fichier yaml
        * afficher_écran (bool): Si True, affiche les messages à l'écran. Si False, silence complet.
    """
    import sys
    import os
    from datetime import datetime
    #from module_valeurs_extremes import vérifier_configuration
    # initialise information
    lignes_sortie = []
    
    def ajouter_information(text):
        if afficher_écran:
            print(text)
        lignes_sortie.append(text)
    timestamp = datetime.now().strftime("%Y/%m/%d à %H:%M:%S")
    ajouter_information('###################################')
    ajouter_information('# Résumé des paramètres du calcul #')
    ajouter_information('###################################')
    ajouter_information(' ')
    ajouter_information('date de création : '+timestamp)
    ajouter_information("")
    flat_paramètres = {**{k: v for k, v in paramètres.items() if k != 'variables'}, **paramètres['variables']}
    liste_paramètres = ['* '+k+':'+str(v) for k, v in flat_paramètres.items() ]
    ajouter_information('--- information sur les données traitées ---')
    for info in liste_paramètres:
        ajouter_information(info)
    ajouter_information("")

    if information_système:
        try:
            from watermark import watermark
            ajouter_information('--- kernel ---')
            ajouter_information(str(watermark(conda=True)))
            ajouter_information('')
            ajouter_information("--- information système ---")
            ajouter_information(str(watermark()))
            ajouter_information(str(watermark(packages="numpy,scipy,pymc,pymc_extras,arviz,plotly,pandas,matplotlib,pandas,pyextremes")))
        except ImportError:
            ajouter_information("watermark module not available")
    
    if tester_importation:
        ajouter_information("")
        ajouter_information("--- statut du module valeurs_extremes ---")
        
        if 'module_valeurs_extremes' in sys.modules:
            module = sys.modules['module_valeurs_extremes']
            if hasattr(module, '__file__'):
                ajouter_information(f"Fichier: {module.__file__}")
                ajouter_information(f"Modifié: {os.path.getmtime(module.__file__)}")
            else:
                ajouter_information("Module sans fichier associé")
        else:
            ajouter_information("Module non chargé")
        ajouter_information(f"Modules chargés contenant 'module_valeurs_extremes': {[name for name in sys.modules.keys() if 'module_valeurs_extremes' in name]}")
    
    if tester_BLAS:
        ajouter_information("")
        ajouter_information("--- paramètres BLAS pour PyMC ---")
        # Assuming vérifier_configuration() returns a string or can be converted
        try:
            blas_info = vérifier_configuration()
            for info in blas_info: ajouter_information(info)
        except:
            ajouter_information("Erreur lors de la vérification BLAS")
    
    if énumérer_fonctions:
        ajouter_information("")
        ajouter_information("--- fonctions du module ---")
        try:
            functions_list = lister_fonctions_module('module_valeurs_extremes')
            ajouter_information(str(functions_list))
        except:
            ajouter_information("Erreur lors de l'énumération des fonctions")
    
    # Sauvegarde
    if sauver_fichier:
        if fichier is None:
            fichier = f"information_calcul_et_python.txt"
        
        try:
            with open(fichier, 'w', encoding='utf-8') as f:
                f.write("\n".join(lignes_sortie))
            print(f"\nInformations sauvegardées dans: {fichier}")
        except Exception as e:
            if afficher_écran:
                print(f"Erreur lors de la sauvegarde: {e}")
    
    if sortie:
        return lignes_sortie



# Utilisation recommandée dans Jupyter
def setup_auto_reload():
    """
    Configure le rechargement automatique dans Jupyter
    """
    try:
        # Vérifier si on est dans un environnement IPython/Jupyter
        ipython = get_ipython()
        if ipython is None:
            print("Pas dans un environnement IPython/Jupyter")
            return False
            
        # Magic commands pour Jupyter
        ipython.run_line_magic('load_ext', 'autoreload')
        ipython.run_line_magic('autoreload', '2')
        print("Rechargement automatique activé dans Jupyter")
        print("Tous les modules seront rechargés automatiquement")
        return True
        
    except ImportError:
        print("IPython n'est pas disponible")
        print("Installez IPython/Jupyter ou utilisez le rechargement manuel")
        return False
    except Exception as e:
        print(f"Erreur lors de l'activation du rechargement automatique: {e}")
        print("Utilisez manuellement: %load_ext autoreload puis %autoreload 2")
        return False




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

def Tracer_maxi(maxi,ax,légende=True,color='deepskyblue',label = 'quantiles mesurés'):
    """ reporte les couples (période de retour, quantile) sur un graphique x"""
    na = len(maxi)
    maxi.sort()
    période = [(na+0.28)/(na-i-1+0.56) for i in range(na)]
    if légende:
        ax.scatter(période,maxi,  marker = "o",edgecolors='white', s=45, color=color, alpha = 1,label = label)
    else:
        ax.scatter(période,maxi,  marker = "o",edgecolors='white', s=45, color=color, alpha = 1)

def Trouver_maxi_annuels(série):
    """ calcule les maxima annuels d'une série.
    Entrée : 
        * série
    Sortie :
        * série temporelle des maxima annuels
        * liste des maxima annuels
    """
    maxi_valeur =  série.groupby(by=[série.index.year]).max().values
    maxi_date   =  série.groupby(by=[série.index.year]).max().index
    bloc_date   = []
    bloc_maxi   = []
    for i in range(len(maxi_date)): 
        selection =  série[(série==maxi_valeur[i]) & (série.index.year==maxi_date[i])] 
        date_occurrence = selection.index[0]
        valeur = selection.iloc[0]
        bloc_date.append(date_occurrence)
        bloc_maxi.append(valeur)
    série_max = pd.Series(bloc_maxi,index = bloc_date)
    return série_max, bloc_maxi  



def flatten(A):
    """ 
    fournit une liste 'applatie' (flattened) à partir d'une liste de listes
    """
    rt = []
    for i in A:
        if isinstance(i,list): rt.extend(flatten(i))
        else: rt.append(i)
    return rt

##################################################
# fonctions lambda utilisées pour l'export latex #
##################################################

def quantile_to_période(x, m, a): return round(1/(1-np.exp(((-np.exp(((m-x)/a)))))))

def quantile_to_période_lve(x, m, a, xi): return round(1/(1-np.exp(((-(1.+(((x-m)*xi)/a))**(-1./xi))))))

##################################################
# fonctions pour l'export latex                  #
##################################################

def convertir_latex(tableau_résultat, fichier_sortie=None):
    """ 
    Convertir le fichier de résultats en tableau latex, avec un bon alignement pour la lisibilité
    
    Entrée :
        * tableau_résultat : list
        * fichier_sortie : str, optional. Nom du fichier de sortie (avec extension .txt ou .tex).
          Si None, affiche seulement dans le notebook

    Sortie : la conversin est soit affichée à l'écran, soit exporté sur le disque dur
    """
    # nombre de colonnes
    max_len = max(len(row) for row in tableau_résultat)
    # normalisation pour que le tableau est le même nombre de colonnes
    tableau_résultat_normalisé = [row + [''] * (max_len - len(row)) for row in tableau_résultat]
    
    def format_ligne(row):
        """
        formate les valeurs numériques :
        * entier pour les quantiles,
        * 1 chiffre après la virgule pour les paramètres mu, sigma, etc.
        * 2 chiffres après la virgule pour xi
        """
        label  = row[0]
        valeur = row[1:]
        if label in ['$T=10$ ans', '$T=30$ ans', '$T=100$ ans', '$T=300$ ans']:
            fmt = '{:.0f}'
        elif label in ['$\\xi$']:
            fmt = '{:.2f}'
        else:  # e.g., 'AIC', '$\\ell$'
            fmt = '{:.1f}'
        formaté = [label] + [fmt.format(x) if isinstance(x, (float, int)) else x for x in valeur]
        return formaté
    
    def aligner_tableau(latex_code):
        """ 
        aligne le tableau 
        """
        lignes = latex_code.splitlines()
        lignes_formatées = []
        for line in lignes:
            # s'applique que pour les lignes avec des &
            if '&' in line and not line.strip().startswith('%'):
                # séparation des colonnes
                cols = [col.strip() for col in line.split('&')]
                # Suppression de '\\' (elles seront ajoutées plus loin)
                if cols[-1].endswith('\\\\'):
                    cols[-1] = cols[-1][:-2].strip()
                    end_slash = True
                else:
                    end_slash = False
                lignes_formatées.append(cols + ['\\\\' if end_slash else ''])
            else:
                # Copier lignes telles que \toprule, \midrule, \begin{tabular}, etc.
                lignes_formatées.append(line)
        
        # largeur max de chaque colonne
        if not lignes_formatées:
            return latex_code
        
        # On considère les lignes aves des données
        rows       = [row for row in lignes_formatées if isinstance(row, list)]
        num_cols   = max(len(row) for row in rows)
        col_widths = [0] * num_cols
        for row in rows:
            for i, col in enumerate(row[:-1]):  # skip last item if it's '\\'
                col_widths[i] = max(col_widths[i], len(col))
        
        # Tableau aligné
        aligned_output = []
        for row in lignes_formatées:
            if isinstance(row, list):
                aligned_row = ' & '.join(
                    col.ljust(col_widths[i]) for i, col in enumerate(row[:-1])
                )
                if row[-1] == '\\\\':
                    aligned_row += r' \\'
                aligned_output.append(aligned_row)
            else:
                aligned_output.append(row)
        return '\n'.join(aligned_output)
    
    formatted_data = [format_ligne(row) for row in tableau_résultat_normalisé]
    df_formatted = pd.DataFrame(formatted_data)
    latex_table = df_formatted.to_latex(index=False, header=False, escape=False)
   
    # Replace decimal separator and minus sign
    latex_table = latex_table.replace('.', ',').replace('-', '--')
    
    # Supprimer les lignes de structure LaTeX générées par pandas
    lignes_a_supprimer = [
        r'\begin{tabular}',
        r'\toprule',
        r'\midrule',
        r'\bottomrule',
        r'\end{tabular}'
    ]
    
    lignes = latex_table.split('\n')
    lignes_filtrées = []
    for ligne in lignes:
        ligne_nettoyée = ligne.strip()
        # Vérifier si la ligne contient une des structures à supprimer
        if not any(structure in ligne_nettoyée for structure in lignes_a_supprimer):
            lignes_filtrées.append(ligne)
    
    latex_table_nettoyé = '\n'.join(lignes_filtrées)
    
    # Insérer les lignes d'en-tête au début
    lignes_entete = [
        r'\emph{Gumbel}   & $d=1$ j & $d=2$ j & $d=3$ j & $d=n$ j \\',
        r'\hline'
    ]
    
    # Insérer les lignes personnalisées après la 8ème ligne
    lignes_finales = latex_table_nettoyé.split('\n')
    lignes_a_inserer = [
        r'\hline',
        r'\emph{Loi de valeurs extrêmes}   &  &   &    &    \\'
    ]
    
    # Insérer l'en-tête au début
    lignes_finales = lignes_entete + lignes_finales
    
    # Insérer après la 8ème ligne (maintenant à l'index 9 à cause de l'en-tête)
    if len(lignes_finales) >= 10:  # 8 + 2 lignes d'en-tête
        for i, ligne in enumerate(lignes_a_inserer):
            lignes_finales.insert(10 + i, ligne)
    
    latex_table_avec_insertion = '\n'.join(lignes_finales)
    aligned = aligner_tableau(latex_table_avec_insertion)
    
    # Affichage dans le notebook
    print(aligned)
    
    # Sauvegarde dans un fichier si spécifié
    if fichier_sortie:
        try:
            with open(fichier_sortie, 'w', encoding='utf-8') as f:
                f.write(aligned)
            print(f"\nTableau exporté vers : {fichier_sortie}")
        except Exception as e:
            print(f"Erreur lors de l'écriture du fichier : {e}")
    
    #return aligned  # Retourne aussi la chaîne pour usage ultérieur





####################################################################################
# fonctions pour le calage
####################################################################################
def log_vraisemblance_LVE(mu,sig,xi,data):
    """
    Calcule la log vraisemblance pour une fonction de valeurs extrêmes
    Entrée : 
    * mu
    * sigma
    * xi
    * data : liste
    """
    n      = len(data)
    output = np.zeros(n)
    for i in range(n):
        z      = data[i]
        output[i]=(((-(1.+(xi**-1.))*(np.log((1.+((xi*(z-mu))/sig))))))-
                    (np.log(sig)))-((1.+((xi*(z-mu))/sig))**(-1./xi))
    return output.sum()

def log_vraisemblance_gumbel(mu,sig,data):
    """
    Calcule la log vraisemblance pour une fonction de Gumbel
    Entrée : 
    * mu
    * sigma
    * data : liste
    """
    n      = len(data)
    output = np.zeros(n)
    for i in range(n):
        z      = data[i]
        output[i]=(((-np.log(sig)))-((z-mu)/sig))-(np.exp(((mu-z)/sig)))
    return output.sum()

def dérivéeLVE_mu(mu,sig,xi,data):
    """
    Calcule la dérivée de la log vraisemblance par rapport à mu pour une fonction LVE
    Entrée : 
    * mu
    * sigma
    * data : liste
    """
    n      = len(data)
    output = np.zeros(n)
    for i in range(n):
        z      = data[i]
        aux0   = (sig**-2.)*((-1.-(xi**-1.))*(xi*((1.+((xi*(z-mu))/sig))**(-2.-(xi**-1.))))) 
        output[i] =((sig**-2.)*((1.+(xi**-1.))*((xi**2)*((1.+((xi*(z-mu))/sig))**-2.))))+aux0 
    return output.sum()   

def dérivéeGumbel_mu(mu,sig,data):
    """
    Calcule la dérivée de la log vraisemblance par rapport à sigma pour une fonction Gumbeé
    Entrée : 
    * mu
    * sigma
    * data : liste
    """
    n      = len(data)
    output = np.zeros(n)
    for i in range(n):
        z      = data[i]
        output[i]=(sig**-1.)-((np.exp(((mu-z)/sig)))/sig)
    return output.sum()

def dérivéeLVE_sigma(mu,sig,xi,data):
    """
    Calcule la dérivée de la log vraisemblance par rapport à sigma pour une fonction LVE
    Entrée : 
    * mu
    * sigma
    * data : liste
    """
    n      = len(data)
    output = np.zeros(n)
    for i in range(n):
        z   = data[i]
        aux0=(-1.-(xi**-1.))*(xi*((((z-mu)**2))*((1.+((xi*(z-mu))/sig))**(-2.-(xi**-1.)))))
        aux1=((sig**-4.)*aux0)+(2.*((sig**-3.)*((z-mu)*((1.+((xi*(z-mu))/sig))**(-1.-(xi**-1.))))))
        aux2=((2.*((sig**-3.)*(xi*(z-mu))))/(1.+((xi*(z-mu))/sig)))-((sig**-4.)*((xi**2)*((((z-mu)**2))*((1.+((xi*(z-mu))/sig))**-2.))))
        output[i]=((sig**-2.)+aux1)-((1.+(xi**-1.))*aux2)
    return output.sum()  

def dérivéeLVE_xi(mu,sig,xi,data):
    """
    Calcule la dérivée de la log vraisemblance par rapport à xi pour une fonction LVE
    Entrée : 
    * mu
    * sigma
    * data : liste
    """
    n      = len(data)
    output = np.zeros(n)
    for i in range(n):
        z   = data[i]
        aux0=(sig**-2.)*((1.+(xi**-1.))*((((z-mu)**2))*((1.+((xi*(z-mu))/sig))**-2.)))
        aux1=(((2.*((xi**-2.)*(z-mu)))/(1.+((xi*(z-mu))/sig)))/sig)+(-2.*((xi**-3.)*(np.log((1.+((xi*(z-mu))/sig))))))
        aux2=((xi**-2.)*(np.log((1.+((xi*(z-mu))/sig)))))-((((z-mu)/(1.+((xi*(z-mu))/sig)))/xi)/sig)
        aux3=(((2.*((xi**-2.)*(z-mu)))/(1.+((xi*(z-mu))/sig)))/sig)+(-2.*((xi**-3.)*(np.log((1.+((xi*(z-mu))/sig))))))
        aux4=((1.+((xi*(z-mu))/sig))**(-1./xi))*((((sig**-2.)*((((z-mu)**2))*((1.+((xi*(z-mu))/sig))**-2.)))/xi)+aux3)
        output[i]=((aux0+aux1)-(((1.+((xi*(z-mu))/sig))**(-1./xi))*(aux2**2)))-aux4
    return output.sum() 

def dérivéeLVE_xisig(mu,sig,xi,data):
    """
    Calcule la dérivée d'ordre 2 de la log vraisemblance par rapport à sigma et xi pour une fonction LVE
    Entrée : 
    * mu
    * sigma
    * data : liste
    """
    n      = len(data)
    output = np.zeros(n)
    for i in range(n):
        z   = data[i]
        aux0=(((sig**-2.)*((1.+(xi**-1.))*(z-mu)))/(1.+((xi*(z-mu))/sig)))+((sig**-3.)*((((z-mu)**2))*((1.+((xi*(z-mu))/sig))**(-2.-(xi**-1.)))))
        aux1=((xi**-2.)*(np.log((1.+((xi*(z-mu))/sig)))))-((((z-mu)/(1.+((xi*(z-mu))/sig)))/xi)/sig)
        aux2=aux0-((sig**-2.)*((z-mu)*(((1.+((xi*(z-mu))/sig))**(-1.-(xi**-1.)))*aux1)))
        aux3=(sig**-3.)*((1.+(xi**-1.))*(xi*((((z-mu)**2))*((1.+((xi*(z-mu))/sig))**-2.))))
        output[i]=(aux2-((((sig**-2.)*(z-mu))/(1.+((xi*(z-mu))/sig)))/xi))-aux3
    return output.sum()

def dérivéeLVE_musig(mu,sig,xi,data):
    """
    Calcule la dérivée d'ordre 2 de la log vraisemblance par rapport à sigma et mu pour une fonction LVE
    Entrée : 
    * mu
    * sigma
    * data : liste
    """
    n      = len(data)
    output = np.zeros(n)
    for i in range(n):
        z   = data[i]
        aux0=(sig**-3.)*((1.+(xi**-1.))*((xi**2)*((z-mu)*((1.+((xi*(z-mu))/sig))**-2.))))
        aux1=(-1.-(xi**-1.))*(xi*((z-mu)*((1.+((xi*(z-mu))/sig))**(-2.-(xi**-1.)))))
        aux2=((sig**-3.)*aux1)+((sig**-2.)*((1.+((xi*(z-mu))/sig))**(-1.-(xi**-1.))))
        output[i]=(aux0+aux2)-(((sig**-2.)*((1.+(xi**-1.))*xi))/(1.+((xi*(z-mu))/sig)))
    return output.sum()

def dérivéeLVE_muxi(mu,sig,xi,data):
    """
    Calcule la dérivée d'ordre 2 de la log vraisemblance par rapport à xi et mu pour une fonction LVE
    Entrée : 
    * mu
    * sigma
    * data : liste
    """
    n      = len(data)
    output = np.zeros(n)
    for i in range(n):
        z   = data[i]
        aux0=((((-1.-(xi**-1.))*(z-mu))/(1.+((xi*(z-mu))/sig)))/sig)+((xi**-2.)*(np.log((1.+((xi*(z-mu))/sig)))));
        aux1=(((1.+(xi**-1.))/(1.+((xi*(z-mu))/sig)))/sig)-((((1.+((xi*(z-mu))/sig))**(-1.-(xi**-1.)))*aux0)/sig);
        aux2=(sig**-2.)*((1.+(xi**-1.))*(xi*((z-mu)*((1.+((xi*(z-mu))/sig))**-2.))));
        output[i]=(aux1-((((1.+((xi*(z-mu))/sig))**-1.)/xi)/sig))-aux2;
    return output.sum()

def matrice_information(mu,sig,xi,data):
    """
    Calcule la matrice d'information observée
    Entrée : 
    * mu
    * sigma
    * data : liste
    """
    matrice = -np.array([ 
        [dérivéeLVE_mu(mu,sig,xi,data),dérivéeLVE_musig(mu,sig,xi,data),dérivéeLVE_muxi(mu,sig,xi,data)],
        [dérivéeLVE_musig(mu,sig,xi,data),dérivéeLVE_sigma(mu,sig,xi,data),dérivéeLVE_xisig(mu,sig,xi,data)],
        [dérivéeLVE_muxi(mu,sig,xi,data),dérivéeLVE_xisig(mu,sig,xi,data),dérivéeLVE_xi(mu,sig,xi,data)]
        ])
    return matrice

def varT(t,fit,mat):
    """
    Calcule la variance
    Entrée : 
    * t : période de retour
    * fit : calage des coefficients LVE fit = (mu, sigma, xi)
    * mat : matrice d'informaion
    """
    mu  = fit[0]
    sig = fit[1]
    xi  = fit[2]
    varMu = mat[0,0]
    varSi = mat[1,1]
    varXi = mat[2,2]
    covMuSi = mat[0,1]
    covMuXi = mat[0,2]
    covSiXi = mat[1,2]
    dF1    = 1
    dF2    = (-1.+(((-np.log((1.-(t**-1.)))))**(-xi)))/xi 
    aux0   = (-1.+(((-np.log(((-1.+t)/t))))**xi))-(xi*(np.log(((-np.log(((-1.+t)/t))))))) 
    dF3    = sig*((xi**-2.)*((((-np.log(((-1.+t)/t))))**(-xi))*aux0)) 
    aux1   = (2.*(covSiXi*(dF2*dF3)))+(((dF1**2)*varMu)+(((dF2**2)*varSi)+((dF3**2)*varXi))) 
    output = (2.*(covMuSi*(dF1*dF2)))+((2.*(covMuXi*(dF1*dF3)))+aux1) 
    return output    

def dérivéeGumbel_mu(mu,sig,data):
    """
    Calcule la dérivée de la loi de Gumbel par rapport à mu
    Entrée : 
    * mu
    * sigma
    * data : liste
    """
    n      = len(data)
    output = np.zeros(n)
    for i in range(n):
        z         = data[i]
        output[i] = (-(np.exp(((mu-z)/sig)))*(sig**-2.))
    return output.sum()

def dérivéeGumbel_sig(mu,sig,data):
    """
    Calcule la dérivée de la loi de Gumbel par rapport à mu
    Entrée : 
    * mu
    * sigma
    * data : liste
    """
    n      = len(data)
    output = np.zeros(n)
    for i in range(n):
        z         = data[i]
        aux0      = (-2.*((np.exp(((mu-z)/sig)))*((sig**-3.)*(mu-z))))+(-2.*((sig**-3.)*(z-mu)));
        output[i] = ((sig**-2.)+aux0)-((np.exp(((mu-z)/sig)))*((sig**-4.)*(((mu-z)**2))));
    return output.sum()

def dérivéeGumbel_musig(mu,sig,data):
    """
    Calcule la dérivée d'ordre 2 de la loi de Gumbel par rapport à mu et sigma
    Entrée : 
    * mu
    * sigma
    * data : liste
    """
    n      = len(data)
    output = np.zeros(n)
    for i in range(n):
        z         = data[i]
        aux0      = ((np.exp(((mu-z)/sig)))*(sig**-2.))+((np.exp(((mu-z)/sig)))*((sig**-3.)*(mu-z))) 
        output[i] = aux0-(sig**-2.) 
    return output.sum()

def matrice_information_gumbel(mu, sig, data):
    """
    Calcule la matrice d'information pour la loi de Gumbel
    Entrée : 
    * mu
    * sigma
    * data : liste
    """
    matrice = -np.array([ 
        [dérivéeGumbel_mu(mu, sig, data),dérivéeGumbel_musig(mu, sig, data)],
        [dérivéeGumbel_musig(mu, sig, data),dérivéeGumbel_sig(mu, sig, data)],
        ])
    return matrice

def varT_gumbel(t,fit,mat):
    """
    Calcule la variance
    Entrée : 
    * t : période de retour
    * fit : calage des coefficients LVE fit = (mu, sigma)
    * mat : matrice d'informaion
    """
    mu  = fit[0]
    sig = fit[1]
    varMu = mat[0,0]
    varSi = mat[1,1]
    covMuSi = mat[0,1]
    dF1    = 1
    dF2    = (-np.log(((-np.log((1.-(t**-1.))))))) 
    output = 2.*covMuSi*dF1*dF2+(dF1**2)*varMu + (dF2**2)*varSi 
    return output   

def convertir_date(date,format='complet'):
    if format == 'simple':
        return datetime.datetime.strptime(date.to_string(index = False)  ,"%Y-%m-%d").strftime("%d/%m/%Y")
    else:
        return date.iloc[0].strftime('%Y-%m-%d')
    
def quantile_lve(T,mu,sigma,xi):
    if xi != 0:
        C = mu-sigma/xi*(1-(-np.log(1-1/T))**(-xi))
    else:
        C = mu-sigma *np.log(-np.log(1-1/T) )
    return C

def type_lve(xi):
    if xi>0:
        nom_loi = 'Fréchet'
    else:
        nom_loi = 'Weibull'
    return nom_loi

def tracer_résultats(i,ax,data, z_alpha,période, résultat_lve, résultat_gumbel,unité='mm'):
    """ Tracer les résultats pour l'indice avec i dans l'intervalle 1--4
    Entrée :
    * i : integer
    * ax : axe de figure
    """
    symboles_variables = ['$P_{{1}}$','$P_{{2}}$','$P_{{3}}$','$P_{{n}}$']
    _, maxi_annuels_val = Trouver_maxi_annuels(data[i])
    nom_figure = 'abcd'[i]
    mu, sigma, xi = résultat_lve[i]
    mu_g, sigma_g = résultat_gumbel[i]
    varIO         = np.linalg.inv(matrice_information(mu,sigma,xi,maxi_annuels_val)) 
    varIO_gumbel  = np.linalg.inv(matrice_information_gumbel(mu_g,sigma_g,maxi_annuels_val))
    ax.set_xlabel(r"$T$ [an]")
    ax.set_ylabel(rf"{symboles_variables[i]} [{unité}]")
    ax.grid()
    ax.grid(which='minor', color='grey', linestyle='-', alpha=0.5,linewidth=0.5)
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(which='major', length=7)
    ax.tick_params(which='minor', length=4)
    q_gumbel = [quantile_lve(T,mu_g,sigma_g,0)  for T in période]
    q_lve    = [quantile_lve(T,mu,sigma,xi)  for T in période]
    ax.semilogx(période,q_gumbel,'-',color='red' , label = "loi de Gumbel")
    ax.semilogx(période,q_lve,'--',color='black' , label = "loi de "+type_lve(xi))
    borne_sup = [quantile_lve(T,mu_g,sigma_g,0)+z_alpha*varT_gumbel(T,[mu_g, sigma_g ],varIO_gumbel)**0.5 for T in période]
    borne_inf = [quantile_lve(T,mu_g,sigma_g,0)-z_alpha*varT_gumbel(T,[mu_g, sigma_g ],varIO_gumbel)**0.5 for T in période]
    ax.fill_between(période,borne_sup,borne_inf,alpha = 0.25, color = 'orange')
    Tracer_maxi(maxi_annuels_val,ax)
    ax.legend(loc="upper left",ncol=2)
    
    ax.text(0.1, 0.8,  '('+nom_figure+')',
    horizontalalignment='center',
    verticalalignment='center',
    transform = ax.transAxes,fontsize=12)
      
def tracer_résultats_pymc(i,ax,data,période_retour, réca_lve, réca_gumbel, bornes_inf, bornes_sup,unité='mm'):
    """ 
    Tracer les résultats pour l'indice avec i dans l'intervalle 1--4 pour les résultats
    obtenus avec pymc
    Entrée :
    * i : integer
    * ax : axe de figure
    """
    #global période_retour, réca_lve, réca_gumbel, borne_inf, borne_sup
    _, maxi_annuels_val = Trouver_maxi_annuels(data[i])
    nom_figure = 'abcd'[i]
    mu, sigma, xi = réca_lve[i]
    mu_g, sigma_g = réca_gumbel[i]
    symboles_variables = ['$P_{{1}}$','$P_{{2}}$','$P_{{3}}$','$P_{{n}}$']
    ax.set_xlabel(r"$T$ [an]")
    ax.set_ylabel(rf"{symboles_variables[i]} [{unité}]")
    ax.grid()
    ax.grid(which='minor', color='grey', linestyle='-', alpha=0.5,linewidth=0.5)
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(which='major', length=7)
    ax.tick_params(which='minor', length=4)
    q_gumbel = [quantile_lve(T,mu_g,sigma_g,0)  for T in période_retour]
    q_lve    = [quantile_lve(T,mu,sigma,xi)  for T in période_retour]
    ax.semilogx(période_retour,q_gumbel,'-',color='red' , label = "loi de Gumbel")
    ax.semilogx(période_retour,q_lve,'--',color='black' , label = "loi de "+type_lve(xi))
    borne_sup = np.array(bornes_sup[i])
    borne_inf = np.array(bornes_inf[i])
    ax.fill_between(période_retour,borne_sup,borne_inf,alpha = 0.25, color = 'orange')
    Tracer_maxi(maxi_annuels_val,ax)
    ax.legend(loc="upper left",ncol=2)
    
    ax.text(0.1, 0.8,  '('+nom_figure+')',
    horizontalalignment='center',
    verticalalignment='center',
    transform = ax.transAxes,fontsize=12)

 

 ###################################################################
def filtrer_années_complètes(df, colonne_date='date'):
    """
    Filtre un DataFrame pour ne garder que les années civiles complètes
    en considérant la première et dernière date disponible.

    Entrées :
        * df : pandas.DataFrame. Attention : si DataFrame ne contient pas des données continues, la fonction
            ne scince pas la série
        * colonne_date : str, optional. Le nom de la colonne contenant les dates (par défaut: 'date')
    Sortie :
        * pandas.DataFrame. DataFrame filtré avec seulement les années complètes
    """
    # S'assurer que la colonne date est en format datetime
    df_copy = df.copy()
    df_copy[colonne_date] = pd.to_datetime(df_copy[colonne_date])

    # Obtenir la première et dernière date
    premiere_date = df_copy[colonne_date].min()
    derniere_date = df_copy[colonne_date].max()

    # Calculer la première année complète
    if premiere_date.month == 1 and premiere_date.day == 1:
        premiere_annee_complete = premiere_date.year
    else:
        premiere_annee_complete = premiere_date.year + 1

    # Calculer la dernière année complète
    if derniere_date.month == 12 and derniere_date.day == 31:
        derniere_annee_complete = derniere_date.year
    else:
        derniere_annee_complete = derniere_date.year - 1

    # Filtrer le DataFrame
    if premiere_annee_complete <= derniere_annee_complete:
        mask = ((df_copy[colonne_date].dt.year >= premiere_annee_complete) & 
                (df_copy[colonne_date].dt.year <= derniere_annee_complete))
        return df_copy[mask].reset_index(drop=True)
    else:
        return pd.DataFrame(columns=df.columns)

 

def vérifier_existence_fichier_répertoire(fichier, dossier):
    """
    vérifie si un fichier 'fichier' est dans le répertoire 'dossier'.
    
    Entrées :
        * répertoire : adresse du répertoire
       *  fichier : nom du fichier
        
    Sortie :
        bool : True si le fichier existe, False sinon.
    """
    file_path = os.path.join(dossier, fichier)
    bool = os.path.isfile(file_path)
    if not bool: print(f"Erreur ! le fichier {fichier} n'est pas le répertoire {dossier}.")
    return bool



def importer_config_yaml(utilisation_safran=False, lire_pickle = True, variable_pluie = True, adresse = None, fichier_yaml  = 'station_safran.yaml'):
    """ 
    Retourne le fichier de paramètres (dictionnaire) du fichier de configuration yaml
    Ce fichier a été créé par le script extraction_poste
    Ce fichier est censé se trouver dans le répertoire principal.
    La fonction affiche quelques informations sur le poste
    Entrée :
        * utilisation_safran : boolean. Par défaut False. Si True alors on considère que les données
            traitées sont issues du modèle safran
        * lire_pickle : si True, lire les fichiers de données au format de données. Sinon les lires
          au format csv.
    Sortie : un dictionnaire avec 
        * 'paramètres': paramètres,
        * 'pluie': données_pluie,
        * 'format_date': format_date_fichier,
        * 'nom': nom_choisi,
        * 'id_station': id_station,
        * 'nom_poste': nom_poste,
        * 'répertoire': répertoire_résultats
    """

    if adresse is None:
        répertoire_base      = '/Data/'
        répertoire_principal = répertoire_base + 'meteo.data/base'
    #fichier_yaml        = répertoire_base+'station_safran.yaml'
    
    else:
        répertoire_base = adresse

    try:
        if vérifier_existence_fichier_répertoire('station_safran.yaml',répertoire_base):
            with open(Path(répertoire_base) / fichier_yaml, 'r',encoding='utf-8') as file:
                    paramètres = yaml.safe_load(file)
            logger.info(f"Données lues depuis {fichier_yaml}")
    except IOError as e:
        logger.error(f"Erreur lors dans la lecture du fichier yaml : {e}")
        raise


    nom_poste    = paramètres['poste']
    nom_commune  = paramètres['commune']  
    id_station   = paramètres['id_station']     
    existe_pluie = paramètres['variables']['pluie']       
    existe_neige = paramètres['variables']['neige'] 
    existe_HN    = paramètres['variables']['manteau'] 
    existe_TN    = paramètres['variables']['TN'] 
    existe_TX    = paramètres['variables']['TX'] 
    existe_TM    = paramètres['variables']['TM'] 

    # paramètres
    longitude_poste      = paramètres['longitude']
    latitude_poste       = paramètres['latitude']  
    altitude_poste       = paramètres['altitude'] 
    répertoire_résultats = paramètres['dossier']
    répertoire_travail   = paramètres['dossier']+'/'

    # conversion booléen en chaîne de caractères
    fnc_oui = lambda boolean: 'oui' if boolean == True   else 'non'

    print(f"* Nom du poste : {nom_poste}.")
    print(f"* Localisation : sur la commune de {nom_commune}, avec le numéro d'identification : {id_station}.")
    print(f"* Coordonnées du poste : longitude = {longitude_poste:.2f}° et latitude {latitude_poste:.2f}° Altitude = {altitude_poste} m.")
    print(f"* Répertoire de travail : {répertoire_résultats}.")
    print(f"* Données de pluie : {fnc_oui(existe_pluie)}.")
    print(f"* Données de neige : {fnc_oui(existe_neige)}.")
    print(f"* Données safran ? {fnc_oui(utilisation_safran)}.")

    if existe_pluie:
        if utilisation_safran:
            if lire_pickle:
                if not vérifier_existence_fichier_répertoire('données_safran_précipitations_totales.pkl',répertoire_travail):
                    print("Il y a un problème avec les données de pluie : le fichier 'données_safran_précipitations_totales.pkl' est manquant.")
            else:
                if not vérifier_existence_fichier_répertoire('données_safran_précipitations_totales.csv',répertoire_travail):
                    print("Il y a un problème avec les données de pluie : le fichier 'données_safran_précipitations_totales.csv' est manquant.")
        else:
            if lire_pickle:
                if not vérifier_existence_fichier_répertoire('données_pluie.pkl',répertoire_travail):
                    print("Il y a un problème avec les données de pluie : le fichier 'données_safran_précipitations_totales.pkl' est manquant.")
            else:
                if not vérifier_existence_fichier_répertoire('données_pluie.csv',répertoire_travail):
                    print("Il y a un problème avec les données de pluie : le fichier 'données_safran_précipitations_totales.csv' est manquant.")
    if existe_neige:
        if utilisation_safran:
            if lire_pickle:
                if not vérifier_existence_fichier_répertoire('données_safran_neige.pkl',répertoire_travail):
                    print("Il y a un problème avec les données de pluie : le fichier 'données_safran_précipitations_totales.pkl' est manquant.")
            else:
                if not vérifier_existence_fichier_répertoire('données_safran_neige.csv',répertoire_travail):
                    print("Il y a un problème avec les données de pluie : le fichier 'données_safran_précipitations_totales.csv' est manquant.")
        else:
            if lire_pickle:
                if not vérifier_existence_fichier_répertoire('données_neige.pkl',répertoire_travail):
                    print("Il y a un problème avec les données de pluie : le fichier 'données_safran_précipitations_totales.pkl' est manquant.")
            else:
                if not vérifier_existence_fichier_répertoire('données_neige.csv',répertoire_travail):
                    print("Il y a un problème avec les données de pluie : le fichier 'données_safran_précipitations_totales.csv' est manquant.")

    if utilisation_safran:
        ### données Safran
        if lire_pickle:
            données_pluie        = pd.read_pickle(répertoire_travail+'données_safran_précipitations_totales.pkl' ) ### données Safran pluie+neige
            données_neige        = pd.read_pickle(répertoire_travail+'données_safran_neige.pkl' )                  ### données Safran neige
            données_pluie = données_pluie.rename(columns={'précipitation': 'pluie'})
        else:
            données_pluie        = pd.read_csv(répertoire_travail+'données_safran_précipitations_totales.csv', sep="\t") ### données Safran pluie+neige
            données_neige        = pd.read_csv(répertoire_travail+'données_safran_neige.csv', sep="\t")                  ### données Safran neige
        format_date_fichier  = 'complet'
        nom_choisi           = 'safran'
        #données_pluie        = pd.read_csv(répertoire_travail+'données_safran_pluie.csv', sep="\t") ### données Safran
        format_date_fichier  = 'complet'
        nom_choisi           = 'safran'

    else:
        ### données poste MF
        if existe_pluie:
            try:
                données_pluie       = pd.read_csv(répertoire_travail+'données_pluie.csv', sep="\t")
                logger.info(f"Données de pluie {répertoire_travail+'données_pluie.csv'} importées.")
            except IOError as e:
                logger.error(f"Erreur lors dans la lecture du fichier de pluies : {e}")
                raise
        else:
            données_pluie = []
        if existe_neige:
            try:
                données_neige      = pd.read_csv(répertoire_travail+'données_neige.csv', sep="\t")
                logger.info(f"Données de neige {répertoire_travail+'données_neige.csv'} importées.")
            except IOError as e:
                logger.error(f"Erreur lors dans la lecture du fichier de neige : {e}")
                raise
        else:
            données_neige = []
        
        format_date_fichier = 'simple'
        nom_choisi          = nom_poste

    print(données_neige.head())
    if existe_neige:
        données_neige.date=pd.to_datetime(données_neige.date)
    if existe_pluie:
        données_pluie.date=pd.to_datetime(données_pluie.date)
    

    os.chdir(répertoire_travail)
    print(f"Le répertoire de travail est {répertoire_travail}.")

    #return paramètres, données_pluie, format_date_fichier, nom_choisi
    if variable_pluie:
        return {
            'paramètres' : paramètres,
            'pluie'      : données_pluie,
            'format_date': format_date_fichier,
            'nom_choisi' : nom_choisi,
            'id_station' : id_station,
            'nom_poste'  : nom_poste,
            'répertoire' : répertoire_résultats,
            'format_date': format_date_fichier,
            'nom_choisi' : nom_choisi,
            'unité'      : 'mm'
        }
    else:
        return {
            'paramètres' : paramètres,
            'format_date': format_date_fichier,
            'nom_choisi' : nom_choisi,
            'id_station' : id_station,
            'nom_poste'  : nom_poste,
            'répertoire' : répertoire_résultats,
            'neige'      : données_neige,
            'format_date': format_date_fichier,
            'nom_choisi' : nom_choisi,
            'unité'      : 'mm' if utilisation_safran else 'cm'  
        }


# def importer_config_yaml(utilisation_safran=False):
#     """ 
#     Retourne le fichier de paramètres (dictionnaire) du fichier de configuration yaml
#     Ce fichier a été créé par le script extraction_poste
#     Ce fichier est censé se trouver dans le répertoire principal.
#     La fonction affiche quelques informations sur le poste
#     Entrée :
#         * utilisation_safran : boolean. Par défaut False. Si True alors on considère que les données
#             traitées sont issues du modèle safran
#     Sortie : un dictionnaire avec 
#         * 'paramètres': paramètres,
#         * 'pluie': données_pluie,
#         * 'format_date': format_date_fichier,
#         * 'nom': nom_choisi,
#         * 'id_station': id_station,
#         * 'nom_poste': nom_poste,
#         * 'répertoire': répertoire_résultats
#     """


#     import yaml
     
#     répertoire_base      = '/Data/'
#     répertoire_principal = répertoire_base + 'meteo.data/base'
#     #fichier_yaml        = répertoire_base+'station_safran.yaml'
#     fichier_yaml         = '/Data/station_safran.yaml'
#     try:
#         with open(fichier_yaml, 'r',encoding='utf-8') as file:
#                 paramètres = yaml.safe_load(file)
#         logger.info(f"Données lues depuis {fichier_yaml}")
#     except IOError as e:
#         logger.error(f"Erreur lors dans la lecture du fichier yaml : {e}")
#         raise


#     nom_poste    = paramètres['poste']
#     nom_commune  = paramètres['commune']  
#     id_station   = paramètres['id_station']     
#     existe_pluie = paramètres['variables']['pluie']       
#     existe_neige = paramètres['variables']['neige'] 
#     existe_HN    = paramètres['variables']['manteau'] 
#     existe_TN    = paramètres['variables']['TN'] 
#     existe_TX    = paramètres['variables']['TX'] 
#     existe_TM    = paramètres['variables']['TM'] 

#     répertoire_résultats = paramètres['dossier']
#     longitude_poste      = paramètres['longitude']
#     latitude_poste       = paramètres['latitude']  
#     altitude_poste       = paramètres['altitude'] 
#     répertoire_travail   = paramètres['dossier']+'/'
#     fnc_oui = lambda boolean: 'oui' if boolean == True   else 'non'
#     print(f"* Nom du poste : {nom_poste}.")
#     print(f"* Localisation : sur la commune de {nom_commune}, avec le numéro d'identification : {id_station}.")
#     print(f"* Coordonnées du poste : longitude = {longitude_poste:.2f}° et latitude {latitude_poste:.2f}° Altitude = {altitude_poste} m.")
#     print(f"* Répertoire de travail : {répertoire_résultats}.")
#     print(f"* Données de pluie : {fnc_oui(existe_pluie)}.")
#     print(f"* Données de neige : {fnc_oui(existe_neige)}.")
#     print(f"* Données safran ? {fnc_oui(utilisation_safran)}.")

#     if not utilisation_safran:
#         if existe_pluie:
#             try:
#                 données_pluie       = pd.read_csv(répertoire_travail+'données_pluie.csv', sep="\t")
#                 logger.info(f"Données de pluie {répertoire_travail+'données_pluie.csv'} importées.")
#             except IOError as e:
#                 logger.error(f"Erreur lors dans la lecture du fichier de pluies : {e}")
#                 raise
#         else:
#             données_pluie = []
#         if existe_neige:
#             try:
#                 données_neige      = pd.read_csv(répertoire_travail+'données_neige.csv', sep="\t")
#                 logger.info(f"Données de neige {répertoire_travail+'données_neige.csv'} importées.")
#             except IOError as e:
#                 logger.error(f"Erreur lors dans la lecture du fichier de neige : {e}")
#                 raise
#         else:
#             données_neige = []
        
#         format_date_fichier = 'simple'
#         nom_choisi          = nom_poste
#     else:
#         try:
#             données_pluie        = pd.read_csv(répertoire_travail+'données_safran_pluie_sans_neige.csv', sep="\t") ### données Safran
#             données_neige        = pd.read_csv(répertoire_travail+'données_safran_neige.csv', sep="\t") ### données Safran
#             logger.info(f"Données de pluie {répertoire_travail+'données_safran_pluie_sans_neige.csv'} importées.")
#             logger.info(f"Données de neige {répertoire_travail+'données_safran_neige.csv'} importées.")
#         except IOError as e:
#             logger.error(f"Erreur lors dans la lecture du fichier : {e}")
#             raise
#         #données_pluie        = pd.read_csv(répertoire_travail+'données_safran_pluie.csv', sep="\t") ### données Safran
#         format_date_fichier  = 'complet'
#         nom_choisi           = 'safran'

#     données_pluie.date=pd.to_datetime(données_pluie.date)

#     os.chdir(répertoire_travail)
#     print(f"Le répertoire de travail est {répertoire_travail}.")

#     #return paramètres, données_pluie, format_date_fichier, nom_choisi
#     return {
#         'paramètres' : paramètres,
#         'pluie'      : données_pluie,
#         'format_date': format_date_fichier,
#         'nom_choisi' : nom_choisi,
#         'id_station' : id_station,
#         'nom_poste'  : nom_poste,
#         'répertoire' : répertoire_résultats,
#         'neige'      : données_neige#,
#         # 'longitude'  : longitude_poste,
#         # 'latitude'   : latitude_poste,
#         # 'altitude'   : altitude_poste
        
#     }



def tester_années_civiles(df, colonne_date='date',type_variable='pluie'):
    """ Teste si le dataframe couvre des années civiles ou non"""
    df_copy = df.copy()
    df_copy[colonne_date] = pd.to_datetime(df_copy[colonne_date])
    première_date = df_copy[colonne_date].min()
    dernière_date = df_copy[colonne_date].max()
    print(f"* Date de début : {première_date.strftime('%d/%m/%Y')  }")
    print(f"* Date de fin   : {dernière_date.strftime('%d/%m/%Y')  }")
    if première_date.month == 1 and première_date.day == 1:
        première_année_civile = True
    else:
        première_année_civile = False
    if dernière_date.month == 12 and dernière_date.day == 31:
        dernière_année_civile = True
    else:
        dernière_année_civile = False
    if not première_année_civile and not dernière_année_civile:
        print("La série chronologique ne couvre pas des années civiles.")
        print(f"Il est recommandé d'employer la fonction 'filtrer_années_complètes(données_{type_variable})'")


def tracer_données_loi_extrêmes(maxi_annuels_val,mu,sigma,xi,mu_g,sigma_g,varIO, varIO_gumbel,z_alpha, confiance= True,unité='mm',période_max = None,incertitude='Gumbel'):
    """
    reporte sur un diagramme semi log les maxima annuels, puis trace la loi de valeurs extrêmes
    et la loi de Gumbel. Pour la loi de Gumbel, l'intervalle de confiance à 70 % est tracé en orange.
    Entrée : 
        * maxi_annuels_val : liste de maxi annuels
        * mu      : coefficient mu de la loi de valeurs extrêmes
        * sigma   : coefficient sigma de la loi de valeurs extrêmes
        * xi      : coefficient xi de la loi de valeurs extrêmes
        * mu_g    : coefficient mu de la loi de Gumbel
        * sigma_g : coefficient sigma de la loi de Gumbel
        * varIO_gumbel : variance de l'information observée pour la loi de Gumbeé
        * z_alpha      : coefficient pour l'intervalle de confiance
        * confiance    : boolean. Si True trace l'intervalle de confiance
    """
    # loi GEV
    if xi>0: 
        nom_loi = 'Fréchet'
    else:
        nom_loi = 'Weibull'

    if période_max is None:
        if max(maxi_annuels_val)<100:
            max_période = 100
        else:
            max_période = np.ceil(max(maxi_annuels_val))*10
    else:
        max_période = période_max
    période = np.linspace(1.01,max_période,1000)

    quantile_gev    = lambda T:  mu-sigma/xi*(1-(-np.log(1-1/T))**(-xi))
    quantile_gumbel = lambda T: mu_g-sigma_g *np.log(-np.log(1-1/T) )
    
    fig, axes = plt.subplots(figsize=(8,4))
    axes.set_xlabel(r"$T$ [an]")
    axes.set_ylabel(rf"$P$ [{unité}]")
    axes.grid()
    axes.grid(which='minor', color='grey', linestyle='-', alpha=0.5,linewidth=0.5)
    axes.yaxis.set_minor_locator(AutoMinorLocator(5))
    axes.tick_params(which='major', length=7)
    axes.tick_params(which='minor', length=4)
    plt.xscale('symlog', linthresh=1 )
    if incertitude == 'Gumbel':
        axes.semilogx(période,quantile_gumbel(période),'-',color='red' , label = "loi de Gumbel")
        axes.plot(période,quantile_gev(période),'--',color='black' , label = "loi de "+nom_loi)
    else:
        axes.semilogx(période,quantile_gumbel(période),'--',color='black', label = "loi de Gumbel")
        axes.plot(période,quantile_gev(période),'-',color='red' , label = "loi de "+nom_loi)
    if confiance:
        if incertitude == 'Gumbel':
            borne_sup = [quantile_gumbel(T)+z_alpha*varT_gumbel(T,[mu, sigma,xi ],varIO_gumbel)**0.5 for T in période]
            borne_inf = [quantile_gumbel(T)-z_alpha*varT_gumbel(T,[mu, sigma,xi ],varIO_gumbel)**0.5 for T in période]
            axes.fill_between(période,borne_sup,borne_inf,alpha = 0.25, color = 'orange')
        else:
            borne_sup = [quantile_gev(T)+z_alpha*varT(T,[mu, sigma,xi ],varIO)**0.5 for T in période]
            borne_inf = [quantile_gev(T)-z_alpha*varT(T,[mu, sigma,xi ],varIO)**0.5 for T in période]
            axes.fill_between(période,borne_sup,borne_inf,alpha = 0.25, color = 'orange')

    Tracer_maxi(maxi_annuels_val,axes)
    fig.legend(loc="upper center",ncol=3,frameon=False)
    return fig, axes, nom_loi




def tracer_pymc(période,maxi,mu,sigma,xi,μ_g,σ_g,y_L,y_U,unité='mm'):
    """" Tracer le résultat PyMC après une calibration"""
    # loi GEV
    if xi>0:
        nom_loi = 'Fréchet'
    else:
        nom_loi = 'Weibull'

    fnc_quantile_gev  = lambda T:  mu-sigma/xi*(1-(-np.log(1-1/T))**(-xi))
    fnc_quantile_gumbel = lambda T: μ_g-σ_g *np.log(-np.log(1-1/T) )
    #résultat_lve.append([mu, sigma,xi ]) 
    #résultat_gumbel.append([mu_g, sigma_g ]) 
    fig, axes = plt.subplots(figsize=(8,4))
    axes.set_xlabel(r"$T$ [an]")
    axes.set_ylabel(rf"$P$ [{unité}]")
    axes.grid()
    axes.grid(which='minor', color='grey', linestyle='-', alpha=0.5,linewidth=0.5)
    axes.yaxis.set_minor_locator(AutoMinorLocator(5))
    axes.tick_params(which='major', length=7)
    axes.tick_params(which='minor', length=4)
    plt.xscale('symlog', linthresh=1 )
    axes.semilogx(période,fnc_quantile_gev(période),'-',color='red' , label = "loi de "+nom_loi)
    axes.semilogx(période,fnc_quantile_gumbel(période),'--',color='black' , label = "loi de Gumbel")
    #axes.plot(période,quantile_gev(période),'-',color='red' , label = "loi de "+nom_loi)

    borne_sup = np.array(y_L)
    borne_inf = np.array(y_U)
    axes.fill_between(période,borne_sup,borne_inf,alpha = 0.25, color = 'orange')

    Tracer_maxi(maxi,axes)
    fig.legend(loc="upper center",ncol=3,frameon=False)
    return fig, nom_loi

##########################################################################
# Fonctions d'export
##########################################################################
def exporter_image(fig,nom,analyse,variable,extension='png',dpi=300):
    print(f"Dossier d'export {os.getcwd()}")
    nom_complet = nom+'_'+analyse+'_'+variable
    nom_complet = nom_complet.replace(' ','-')
    if extension == 'png':
        print(f"export de {nom_complet+'.png'} réussi")
        fig.savefig(nom_complet+'.png',bbox_inches='tight',dpi=dpi)
    else:
        print(f"export de {nom_complet+'.pdf'} réussi")
        fig.savefig(nom_complet+'.pdf',bbox_inches='tight' )

def exporter_données(série,nom,type_variable,compatibilité_ancien_fichier = True):
    série_export = série.copy()
    if compatibilité_ancien_fichier:
        série_export.index = série_export.index.strftime('%Y %m %d')
    nom_fichier = 'Chute'+'_'+nom+'_'+type_variable+'.csv'
    (série_export.fillna(value=0)).to_csv(nom_fichier , sep=' ', encoding='utf-8',index = True)
    print(f"Export du fichier {nom_fichier} dans {os.getcwd()}.")
    if compatibilité_ancien_fichier: print("Compatibilité avec les anciens fichiers Mathematica")

 

def exporter_precipitations(df, fichier_sortie="precipitations.tex",unité='mm'):
    """
    Exporte un DataFrame de précipitations vers un fichier LaTeX propre
    
    Entrée :
        * df : pandas.DataFrame, Le DataFrame contenant les données de précipitations
        * fichier_sortie : str, optional. Le nom du fichier de sortie (par défaut "precipitations.tex")
        
    """
    
    # Générer le tableau LaTeX avec pandas
    latex_table = df.to_latex(index=False)
    
    # Supprimer les lignes de structure LaTeX générées par pandas
    lignes_a_supprimer = [
        r'\begin{tabular}',
        r'\toprule',
        r'\midrule',
        r'\bottomrule',
        r'\end{tabular}'
    ]
    
    lignes = latex_table.split('\n')
    lignes_filtrées = []
    for ligne in lignes:
        ligne_nettoyée = ligne.strip()
        # Vérifier si la ligne contient une des structures à supprimer
        if not any(structure in ligne_nettoyée for structure in lignes_a_supprimer):
            if ligne_nettoyée:  # Éviter les lignes vides
                lignes_filtrées.append(ligne)
    
    # Supprimer aussi la ligne d'en-tête générée automatiquement (0 & 1 & 2 & 3 \\)
    # Cette ligne correspond généralement à la première ligne des données filtrées
    if lignes_filtrées and any(char.isdigit() for char in lignes_filtrées[0]) and '&' in lignes_filtrées[0]:
        lignes_filtrées.pop(0)
    
    # Insérer l'en-tête personnalisé au début
    lignes_entete = [
        rf'date          & $P$ ({unité}) & durée (j)& $T$ (ans)\\',
        r'\hline'
    ]
    
    # Combiner en-tête et données
    lignes_finales = lignes_entete + lignes_filtrées
    
    # Joindre toutes les lignes
    latex_final = '\n'.join(lignes_finales)
    
    # Affichage dans le notebook
    print(latex_final)
    
    # Sauvegarde dans le fichier
    try:
        with open(fichier_sortie, 'w', encoding='utf-8') as f:
            f.write(latex_final)
        print(f"\nTableau des précipitations exporté vers : {fichier_sortie}")
    except Exception as e:
        print(f"Erreur lors de l'écriture du fichier : {e}")
    
    #return latex_final