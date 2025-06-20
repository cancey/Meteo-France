# Meteo-France
scripts python pour analyser les données météo en France

1. creation_liste_postes : sert à générer la liste des postes de Météo-France dans un département ou un ensemble de départements. Une carte est également générée.
2. extraction-poste : sert à extraire les séries de données depuis le site de Météo-France. La procédure change un tout peu selon que le poste est climatologique ou autre (nivo-météo, Nivôse). Le script extrait les données, trace les courbes et fait une analyse des valeurs extrêmes.
3. safran_extraction : analyse les données safran pour la région autour du poste de mesures (l'information est fournie via un fichier de configuration yaml)

Note : ces cahiers font appel à des paquets à installer (si ce n'est déjà fait) et des fichiers supplémentaires de données topographiques :
* [https://france-geojson.gregoiredavid.fr/repo/departements.geojson](https://www.data.gouv.fr/fr/datasets/r/90b9341a-e1f7-4d75-a73c-bbc010c7feeb) : limites départementales
* les liens vers la fiche d'information des postes (de la forme https://publitheque.meteo.fr/okapi/_composantsHTML/refGeo/pageInfosStation.jsp?insee=) sont cassés
* cartographie de l'Europe (format SHP) : https://www.naturalearthdata.com/downloads/10m-cultural-vectors/
* liste des communes françaises : https://planete-excel.fr/liste-des-communes-de-france-au-format-excel/
* MNT Alpes tiré de https://ec.europa.eu/eurostat/web/gisco/geodata/digital-elevation-model/copernicus#Elevation
