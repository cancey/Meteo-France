"""
Module : module_rapport
Date de création : 18 avril 2026
Rapport climatologique Météo-France -- génération PDF via Jinja2 + LaTeX.

Principe : alimenter un objet RapportMF au fil du notebook, puis appeler .generer().

Exemple d'usage dans le notebook
---------------------------------
    from module_rapport import RapportMF

    rapport = RapportMF()

    # ---- section précipitations ----
    rapport.section("Précipitations liquides")
    rapport.item(f"Maximum journalier : {val_max:.1f} mm le {date_max}")
    rapport.item(f"Cumul annuel moyen : {cumul_moy:.0f} mm")
    rapport.figure(nom_figure, "Chronique des précipitations journalières")
    rapport.figures_cote_a_cote(
        fig_cumul_mensuel, "Cumuls mensuels interannuels",
        fig_cumul_annuel,  "Cumuls annuels",
    )

    # ---- section températures ----
    rapport.section("Températures")
    rapport.item(f"Maximum : {tx_max:.1f} °C le {date_tx_max}")
    rapport.figure(nom_fig_temp, "Évolution des températures min/max")

    # ---- génération finale ----
    rapport.generer(
        répertoire   = répertoire_résultats,
        nom_poste    = nom_poste,
        poste_choisi = poste_choisi,
        département  = dico_param['département'],
        commune      = nom_commune,
        altitude     = dico_param['altitude'],
        lat          = dico_param['lat'],
        lon          = dico_param['lon'],
        date_debut   = dico_param['date_debut'],
        date_fin     = dico_param['date_fin'],
        n_valeurs    = dico_param['n_valeurs'],
        existe_pluie = existe_pluie,
        existe_neige = existe_neige,
        existe_HN    = existe_HN,
        existe_TN    = existe_TN,
        existe_TM    = existe_TM,
        existe_TX    = existe_TX,
    )
"""

import os
import subprocess
import jinja2
import importlib
import importlib.util
from pathlib import Path


chemin_modèle = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'synthese_meteo.tex.j2')

########################
# recharche du module  #
########################
def reload_module_rapport(dossier=None):
    """
    Recharge le module module_rapport de façon robuste
    
    Entrée :
        * dossier : str, optional. Chemin vers le dossier contenant le module
        
    Sortie :
        * module ou None. Le module rechargé ou None en cas d'erreur
    """
    # Chemin explicite
    if dossier is None:
        module_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    else:
        module_dir = Path(dossier)
    
    module_path = module_dir / 'module_rapport.py' #os.path.join(module_dir, 'module_rapport.py')
    
    # Vérification de l'existence du fichier
    if not module_path.exists():
        print(f"ERREUR: {module_path} n'existe pas")
        return None
    
    try:
        # Ajouter le dossier au path si nécessaire
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        
        # Méthode 1: Si le module n'a jamais été importé
        if 'module_rapport' not in sys.modules:
            spec = importlib.util.spec_from_file_location("module_rapport", module_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["module_rapport"] = module
            spec.loader.exec_module(module)
            print("Module importé pour la première fois.")
        
        # Méthode 2: Si le module existe déjà, le recharger
        else:
            # Supprimer complètement le module du cache
            if 'module_rapport' in sys.modules:
                del sys.modules['module_rapport']
            
            # Nettoyer aussi les sous-modules si ils existent
            modules_to_remove = [name for name in sys.modules.keys() 
                               if name.startswith('module_rapport.')]
            for module_name in modules_to_remove:
                del sys.modules[module_name]
            
            # Reimporter complètement
            spec = importlib.util.spec_from_file_location("module_rapport", module_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["module_rapport"] = module
            spec.loader.exec_module(module)
            print("Module rechargé avec succès.")
        
        # Mettre à jour les globals() du notebook
        globals()['module_rapport'] = sys.modules['module_rapport']
        
        return sys.modules['module_rapport']
        
    except Exception as e:
        print(f"ERREUR lors du rechargement: {e}")
        return None

######################
# fonctions diverses #
######################
# ---------------------------------------------------------------------------
# Chemin de l'image et construction du nom   
# ---------------------------------------------------------------------------

def chemin_image(analyse, variable, nom_poste, répertoire_résultats):
    """
    fournit l'adresse du fichier
    Le nom du fichier est composé de
    - analyse (str) : le type d'analyse
    - variable (str) : le nom de la variable sur laquelle porte l'analyse
    - nom du poste (str)
    - répertoire où placer le fichier (str)
    """
    nom_complet = f'{nom_poste}_{analyse}_{variable}'.replace(' ', '-')
    return os.path.join(répertoire_résultats, nom_complet + '.png')


# ---------------------------------------------------------------------------
# Normalisation du nom de poste   
# ---------------------------------------------------------------------------



def normaliser_nom(nom):
    """
    Réordonne les noms de postes Météo-France du style "Nom (Article)"
    en "Article Nom".

    Exemples
    --------
    "Clusaz (La)"        → "La Clusaz"
    "Grand-Bornand (Le)" → "Le Grand-Bornand"
    "Deux-Alpes (Les)"   → "Les Deux-Alpes"
    "Alpe-d'Huez (L')"   → "L'Alpe-d'Huez"
    "Bourg-Saint-Maurice" → "Bourg-Saint-Maurice"  (inchangé)
    """
    import re as _re
    #m = _re.match(r'^(.+?)\s+\((L[ae]s?|L\')\)\s*$', nom.strip(), _re.IGNORECASE)
    m = _re.match(r'^(.+?)_\((L[ae]s?|L\')\)\s*$', nom.strip(), _re.IGNORECASE)
    if m:
        corps, article = m.group(1).strip(), m.group(2)
        # Cas L' : pas d'espace entre l'article et le nom
        sep = '' if article.endswith("'") else ' '
        return article + sep + corps
    return nom


# ---------------------------------------------------------------------------
# Échappement LaTeX — passage caractère par caractère pour éviter les
# doubles échappements (ex. \ -> \textbackslash{} puis { -> \{)
# ---------------------------------------------------------------------------

def escape_latex(valeur):
    """Échappe les caractères spéciaux LaTeX dans une chaîne quelconque."""
    texte = str(valeur) if not isinstance(valeur, str) else valeur
    TABLE = {
        '\\': r'\textbackslash{}',
        '&':  r'\&',
        '%':  r'\%',
        '$':  r'\$',
        '#':  r'\#',
        '_':  r'\_',
        '{':  r'\{',
        '}':  r'\}',
        '~':  r'\textasciitilde{}',
        '^':  r'\^{}',
    }
    return ''.join(TABLE.get(c, c) for c in texte)


#####################
# classe rapport MF #
#####################

class RapportMF:
    """
    Accumule le contenu d'un rapport météo (sections, items, figures)
    et génère un PDF via un template Jinja2/LaTeX.
    """

    def __init__(self):
        self._sections = []           # [{'titre': str, 'blocs': list}, ...]
        self._section_courante = None

    # ---- construction du contenu ----

    def section(self, titre):
        """
        Démarre une nouvelle section (ou reprend une section existante).
        Tous les appels suivants à .item(), .texte(), .figure() alimentent cette section.
        """
        for s in self._sections:
            if s['titre'] == titre:
                self._section_courante = s
                return self
        s = {'titre': titre, 'blocs': []}
        self._sections.append(s)
        self._section_courante = s
        return self

    def item(self, texte):
        """
        Ajoute un élément de liste puce à la section courante.
        Les items consécutifs sont regroupés dans un même environnement itemize.
        """
        texte = texte.replace('* ','')
        self._verif_section()
        texte_esc = escape_latex(texte)
        blocs = self._section_courante['blocs']
        if blocs and blocs[-1]['type'] == 'items':
            blocs[-1]['items'].append(texte_esc)
        else:
            blocs.append({'type': 'items', 'items': [texte_esc]})
        return self

    def texte(self, texte):
        """Ajoute un paragraphe de texte libre à la section courante."""
        self._verif_section()
        self._section_courante['blocs'].append({
            'type': 'texte',
            'contenu': escape_latex(texte),
        })
        return self

    def figure(self, chemin, legende='', largeur=0.9):
        """
        Ajoute une figure à la section courante.
        chemin : chemin absolu ou relatif vers le fichier PNG/PDF.
        legende : légende affichée sous la figure (optionnelle).
        largeur : fraction de \\textwidth (défaut 0.9).
        """
        self._verif_section()
        if chemin:
            self._section_courante['blocs'].append({
                'type': 'figure',
                'chemin': str(chemin).replace('\\', '/'),
                'legende': escape_latex(legende),
                'largeur': largeur,
            })
        return self

    def figures_cote_a_cote(self, chemin1, legende1, chemin2, legende2, largeur=0.47):
        """
        Ajoute deux figures côte à côte (minipage) à la section courante.
        largeur : fraction de \\textwidth pour chaque figure (défaut 0.47).
        """
        self._verif_section()
        self._section_courante['blocs'].append({
            'type': 'figures2',
            'chemin1': str(chemin1).replace('\\', '/') if chemin1 else None,
            'legende1': escape_latex(legende1),
            'chemin2': str(chemin2).replace('\\', '/') if chemin2 else None,
            'legende2': escape_latex(legende2),
            'largeur': largeur,
        })
        return self

    # ---- génération du PDF ----

    def generer(
        self,
        répertoire,
        nom_poste,
        poste_choisi,
        département,
        commune,
        altitude,
        lat,
        lon,
        date_debut,
        date_fin,
        n_valeurs,
        existe_pluie=False,
        existe_neige=False,
        existe_HN=False,
        existe_TN=False,
        existe_TM=False,
        existe_TX=False,
        chemin_modèle=chemin_modèle,
    ):
        """
        Génère le fichier .tex puis compile en PDF.
        Retourne le chemin du PDF, ou None en cas d'erreur.
        """
        répertoire = os.path.abspath(répertoire)

        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(
                os.path.dirname(os.path.abspath(chemin_modèle))
            ),
            block_start_string='<%',
            block_end_string='%>',
            variable_start_string='<+',
            variable_end_string='+>',
            comment_start_string='<#',
            comment_end_string='#>',
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )
        template = env.get_template(os.path.basename(chemin_modèle))

        fiche = {
            'nom_poste':    escape_latex(normaliser_nom(nom_poste)),
            'poste_choisi': poste_choisi,
            'département':  escape_latex(str(département)),
            'commune':      escape_latex(commune),
            'altitude':     altitude,
            'lat':          f'{lat:.4f}',
            'lon':          f'{lon:.4f}',
            'date_debut':   date_debut,
            'date_fin':     date_fin,
            'n_valeurs':    n_valeurs,
            'existe_pluie': bool(existe_pluie),
            'existe_neige': bool(existe_neige),
            'existe_HN':    bool(existe_HN),
            'existe_TN':    bool(existe_TN),
            'existe_TM':    bool(existe_TM),
            'existe_TX':    bool(existe_TX),
        }

        contenu_tex = template.render(fiche=fiche, sections=self._sections)

        nom_base = f'synthese_{nom_poste.replace(" ", "_")}_{poste_choisi}'
        fichier_tex = os.path.join(répertoire, nom_base + '.tex')
        with open(fichier_tex, 'w', encoding='utf-8') as f:
            f.write(contenu_tex)
        print(f'Fichier .tex écrit : {fichier_tex}')

        cmd = [
            'pdflatex', '-interaction=nonstopmode',
            f'-output-directory={répertoire}',
            fichier_tex,
        ]
        for _ in range(2):
            résultat = subprocess.run(cmd, capture_output=True, text=True, cwd=répertoire)

        fichier_pdf = os.path.join(répertoire, nom_base + '.pdf')
        log = résultat.stdout

        if "can't write on file" in log or "cannot write" in log.lower():
            print("ERREUR : pdflatex ne peut pas écrire le PDF.")
            print("→ Fermez le PDF dans votre lecteur (Acrobat, Edge…) et relancez.")
            print("  Conseil : utilisez SumatraPDF qui ne verrouille pas les fichiers.")
            return None

        if os.path.exists(fichier_pdf):
            print(f'PDF généré : {fichier_pdf}')
            return fichier_pdf

        print('Erreur lors de la compilation pdflatex.')
        lignes = log.splitlines()
        erreurs = [l for l in lignes if l.startswith('!')]
        print('\n'.join(erreurs) if erreurs else '\n'.join(lignes[-40:]))
        return None

    # ---- interne ----

    def _verif_section(self):
        if self._section_courante is None:
            raise RuntimeError(
                "Appelez rapport.section('Titre') avant d'ajouter du contenu."
            )
