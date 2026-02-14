import numpy as np
from typing import Tuple, List, Dict
import random

class Vertex:

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.label_list=[[],[]] # liste des listes forward et backward des etiquettes (Dijkstra MO bi-directionnel)

    def __eq__(self, vertexPrime):
        '''
        Retourne True si le vecteur self est le vecteur vertexPrime (comparaison des noms),
        False sinon.
        
        :param vertexPrime:
        '''
        if not isinstance(vertexPrime, Vertex):
            return False
        return self.name == vertexPrime.name  
    
    def __hash__(self):
        return hash(self.name)
    
    def coordonnees(self): 
        '''
        Retourne le tuple de coordonnees associe au sommet. 
        (MOGPL : positions representees par des sommets)
        '''
        l = self.label
        j = 0
        x = -1
        while True:
            if l[j] == "-":
                if x == -1:
                    x = int(l[:j])
                    i = j+1
                else:
                    return (x, int(l[i:j]))
            j+=1

    def addLabel(self, label, direction: int) -> None: 
        '''
        Ajoute une etiquette a la liste des etiquettes du noeud.
        (Dijkstra MO bi-directionnel)
        
        :param label: etiquette (noeud courant, vecteur de cout, etiquette precedente)
        :param direction: forward (0) ou backward (1)
        '''
        vector = label.vector
        new_list = []

        # Construction de la nouvelle liste d'etiquettes pour le noeud
        for old_label in self.label_list[direction]:
            vectorTemp = old_label.vector

            # Si la nouvelle etiquette est dominee, on garde notre liste initiale
            if dominates(vectorTemp, vector):
                return
            
            if not dominates(vector, vectorTemp):
                new_list.append(old_label)

        # Si la nouvelle etiquette n'est pas dominee, on met a jour notre liste (sans les etiquettes dominees)
        new_list.append(label)
        self.label_list[direction] = new_list

    
class Edge:

    def __init__(self, v1: Vertex, v2: Vertex, dist: float, classe: str) -> None:
        self.vertices: Tuple[Vertex, Vertex] = (v1, v2)
        self.weight = (dist, classe)  # exemple : (5,"B") -> la classe doit etre une lettre majuscule autorisee (en fonction de nbClasses)


class Label:

    def __init__(self, vertex: Vertex, cost_vector: List[float], previous_label, code: int):
        self.vertex = vertex
        self.vector = cost_vector
        self.prev_label = previous_label
        self.code = code

    def labelToString(self):
        '''
        Transforme une etiquette en chaines de caracteres.
        
        :param label: etiquette (noeud courant, vecteur de cout, etiquette precedente)
        '''
        res : str = "(" + self.vertex.name + "," + str(self.vector) + ", "
        if self.prev_label == None:
            return res + "None)"
        return res + self.labelToString(self.prev_label) + ")" 

    def dominated_by_list(self, labelListe: List) -> bool:
        """
        Retourne True si (le vecteur de) label est domine par au moins une etiquette de labelListe,
        False sinon.

        :param label: etiquette (noeud courant, vecteur de cout, etiquette precedente)
        :param labelListe: liste d'etiquettes
        """
        for i in range(len(labelListe)):
            if dominates(labelListe[i].vector, self.vector):
                return True
        return False
    
    def succ_label(self, new_vertex: Vertex, edge: Edge, nbClasses: int, code: int):
        '''
        Cree une nouvelle etiquette qui succede a self (vecteur de cout MAJ avec edge).
        
        :param label: etiquette (noeud courant, vecteur de cout, etiquette precedente)
        :param edge: arc
        :param nbClasses: nombre de dimensions du vecteur de cout
        '''
        vector = self.vector
        classe = ord(edge.weight[1]) - 65 # ord('A')
        dist = edge.weight[0]
        
        return Label(new_vertex, [dist + vector[i] if i <= classe else vector[i] for i in range(nbClasses)], self, code)
    
    def combine(self, labelListe, direction: int) -> List:
        '''
        Retourne une liste des chemins combines entre une etiquette et une liste d'etiquettes.
        Un chemin : (etiquette depuis origine, etiquette depuis destination, vecteur de cout total)
        ou etiquette = label ou les deux procedures se rejoignent

        :param label: etiquette 
        :param labelListe: liste d'etiquettes dans la direction opposee
        :param direction: direction de l'etiquette label
        '''
        vecteurs_cout_finaux = []
        vec = self.vector
        nb_dim = len(vec)

        # Determiner les vecteurs de cout totaux
        for i in range(len(labelListe)):
            vec_suivant = labelListe[i].vector
            vecteurs_cout_finaux.append([vec[j] + vec_suivant[j] for j in range(nb_dim)])

        if direction == 0: # forward
            return [(self, labelListe[i], vecteurs_cout_finaux[i]) for i in range(len(vecteurs_cout_finaux))]
        # backward
        return [(labelListe[i], self, vecteurs_cout_finaux[i]) for i in range(len(vecteurs_cout_finaux))]


class Graph:

    def __init__(self, name: str, nbClasses: int) -> None:
        self.name: str = name
        self.vertices: set[Vertex] = set()
        self.edges: set[Edge] = set()
        self.adj: List[Dict[Vertex, set[Edge]], Dict[Vertex, set[Edge]]] = [dict(), dict()] # liste de successeurs, liste de predecesseurs (donnes par les arcs)
        self.nbClasses = nbClasses # niveaux de securite d'un troncon (lettres majuscules)

    def copy(self): 
        """
        Renvoie une copie du graphe.
        """
        g = Graph(self.name, self.nbClasses) 

        # Copie des sommets
        for v in self.vertices: 
            g.add_vertex(v.name) 

        # Copie des arcs
        E = self.edges.copy()
        for e in E: 
            g.add_edge(e.vertices[0].name, e.vertices[1].name, e.weight[0], e.weight[1]) 

        # Copie des dictionnaires d'adjacence
        dico_succ = {}
        for (sommet, ens) in self.ajd[0].items():
            dico_succ[sommet] = ens.copy()

        dico_pred = {}
        for (sommet, ens) in self.ajd[1].items():
            dico_pred[sommet] = ens.copy()

        g.adj = [dico_succ, dico_pred]

        return g

    def nbVertices(self) -> int:
        """
        Renvoie le nombre de sommets du graphe.
        """
        return len(self.vertices)

    def nbEdges(self) -> int:
        """
        Renvoie le nombre d'aretes du graphe.
        """
        return len(self.edges)

    def add_vertex(self, name: str) -> Vertex:
        """
        Ajoute un sommet a un graphe s'il n'y est pas deja
        Et le retourne.

        :param name: nom du sommet a ajouter
        """
        v = Vertex(name)
        if v not in self.vertices:
            self.vertices.add(v)
            self.adj[0][v] = set()
            self.adj[1][v] = set()
        return v 

    def add_edge(self, namev1: str, namev2: str, dist: float, classe: str) -> None:
        '''
        Ajoute une arete.
        
        :param namev1: nom d'un vecteur qui *existe* dans le graphe
        :param namev2: nom d'un vecteur qui *existe* dans le graphe
        :param dist: poids de l'arete
        :param classe: classe de l'arete
        '''
        # Recuperation des vecteurs dans le graphe
        vertex1 = next(v for v in self.vertices if v.name == namev1)
        vertex2 = next(v for v in self.vertices if v.name == namev2)

        # Pas de boucle autorisee
        if vertex1 == vertex2:
            return

        # Creation de l'arete
        e = Edge(vertex1, vertex2, dist, classe)

        # Ajout dans les listes d'adjacence
        self.edges.add(e)
        self.adj[0][vertex1].add(e)
        self.adj[1][vertex2].add(e)

    def delete_vertex(self, name: str) -> None: 
        """
        Supprime un sommet du graphe s'il y est present.

        :param name: nom du sommet a supprimer
        """
        # Retrouver le sommet a supprimer
        vertex = Vertex(name)
        if vertex not in self.vertices:
            return #le sommet n'est deja pas dans le graphe

        # Mettre a jour l'ensemble des aretes
        self.edges = {e for e in self.edges if vertex not in e.vertices} 

        # Construction des nouveaux dictionnaires d'adjacence
        new_adj = [{sommet:set() for sommet in self.adj[0]}, {sommet:set() for sommet in self.adj[1]}]

        for sommet, ens in self.adj[0]:
            for e in ens:
                if e not in self.edges:
                    new_adj[0][sommet].remove(e)

        for sommet, ens in self.adj[1]:
            for e in ens:
                if e not in self.edges:
                    new_adj[1][sommet].remove(e)

        # Retirer le sommet de l'ensemble des sommets
        self.vertices.remove(vertex)

    def delete_vertices(self, name_list: List[str]) -> None: 
        """
        Supprime l'ensemble des sommets du graphe dont les noms sont présents dans name_list.

        :param name_list: liste de noms de sommets (qui existent ou non dans le graphe)
        """
        for nom in name_list:
            self.delete_vertex(nom)

    def degres(self, sens: int) -> List:
        """
        Renvoie un tableau contenant les tuples (sommet s, degre(s)) pour les sommets du graphe.

        :param sens: 1 si degres sortants, 0 si degres entrants
        """
        return [(v, len(neighbors)) for (v, neighbors) in self.adj[sens]]


    def max_degre(self, sens: int) -> str:
        """
        Renvoie le nom du (premier) sommet de degre maximum du graphe.

        :param sens: 1 si degres sortants, 0 si degres entrants
        """
        deg = self.degres(sens)
        return [x for x in deg if x[1] == max([lenNeighbors[1] for lenNeighbors in deg])][0][0].name


    def affiche_dico_adj(self) -> None:
        """
        Affiche le tableau des dictionnaires d'adjacence du graphe.
        """
        for v in self.adj[0]:
            affiche = str(v.name) + " -> ["
            for e in self.adj[0][v]:
                affiche += e.vertices[1].name + "(" + str(e.weight[0]) + "," + e.weight[1] + "), "
            if self.adj[0][v]:
                print(affiche[:-2] + "]")
            else:
                print(affiche + "]")


    def affiche_etats_avec_labels(self) -> None:
        """
        Affiche les differents sommets et leurs etiquettes associees (forward et backward).
        """
        for v in self.vertices:
            res = str(v.name) + " -> FORWARD ["
            for l in v.label_list[0]:
                res += l.labelToString() + ", "
            res += "] BACKWARD ["
            for l in v.label_list[1]:
                res += l.labelToString() + ", "
            print(res + "]")


    def getNeighbors(self, vertex : Vertex, dir : int) -> List[Edge]:
        '''
        Renvoie la liste des arcs de vertex.
        
        :param vertex: sommet courant
        :param dir: direction de parcours (0: successeurs, 1: predecesseurs)
        '''
        return [e for e in self.adj[dir][vertex]]
    

    def DijkstraMultiObjBidirectionnel(self, origin: Vertex, dest: Vertex) -> List:
        '''
        Applique l'algorithme de Dijkstra multi-objectif bi-directionnel
        pour recuperer l'ensemble des chemins Pareto-optimaux 
        allant du sommet origin au sommet dest.
        
        :param origin: sommet de depart
        :param dest: sommet d'arrivee
        '''
        T = [[],[]] # liste des etiquettes temporaires (pour les deux directions)
        Lres = [] # liste des chemins Pareto-optimaux

        # Ajout de l'etiquette d'origine a T[0]
        code = 0 # compteur du nombre de labels cres
        originLabel = Label(origin, [0 for _ in range(self.nbClasses)], None, code) # Un label = (noeud de depart, vecteur de cout, etiquette precedente)
        code +=1 
        origin.addLabel(originLabel, 0)
        T[0].append(originLabel)

        # Ajout de l'etiquette de destination a T[1]
        destLabel = Label(dest, [0 for _ in range(self.nbClasses)], None, code)
        code += 1
        dest.addLabel(destLabel, 1)
        T[1].append(destLabel)

        d: int = 1 # direction
        while not (stop(T, Lres)):
            d = 1-d # changement de direction

            # Recuperation d'une etiquette dans T[d]
            label = T[d][0] # car pour l'instant, label = removeMin(T[d]) trop lent -> a vectoriser
            T[d] = T[d][1:]

            # Recuperation du noeud courant et de ses arcs (entrants ou sortants en fonction de la direction)
            owner: Vertex = label.vertex
            neighbors: List[Edge] = self.getNeighbors(owner, d)

            # Parcours des voisins
            e: Edge
            for e in neighbors:
                voisin = e.vertices[1-d] # recuperation du voisin
                newLabel = label.succ_label(voisin, e, self.nbClasses, code)
                code += 1

                # Si la nouvelle etiquette n'est pas dominee par celles dans la liste de voisin, on l'ajoute
                if not newLabel.dominated_by_list(voisin.label_list[d]): # dominated by list ca fait deja ca dans add label ?
                    voisin.addLabel(newLabel, d)
                    T[d].append(newLabel)

                    # Si la liste des etiquettes dans l'autre direction n'est pas vide, combiner les chemins
                    if voisin.label_list[1-d] != []:
                        for c in newLabel.combine(voisin.label_list[1-d], d):
                            addResults(c, Lres)

        return Lres
    

### PARETO DOMINANCE ###

def dominates(v1: List[float], v2: List[float]) -> bool:
    '''
    Retourne True si le vecteur v1 domine v2,
    False sinon.

    :param v1: vecteur de cout 
    :param v2: vecteur de cout
    '''
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.all(v1 <= v2) and np.any(v1 < v2)


### ORDRE LEXICOGRAPHIQUE ###

def ordre_lexicographique(v1, v2):
    '''
    Retourne True si v1 <= v2 pour l'ordre lexicographique,
    False sinon.
    
    :param v1: vecteur de taille n
    :param v2: vecteur de taille n
    '''
    for e1, e2 in zip(v1, v2):
        if e1 < e2:
            return True 
        if e2 < e1:
            return False
    return True

def removeMin(T):
    '''
    Renvoie la plus petite etiquette dans T pour l'ordre lexicographique. 
    
    :param T: liste d'etiquettes
    '''
    label_min = T[0]
    for i in range(len(T)):
        if ordre_lexicographique(T[i].vector, label_min.vector):
            label_min = T[i]
        
    T.pop(i)
    return label_min


### CONDITION D'ARRET DANS DIJKSTRA MO DB ###

def stop(T, Lres):
    '''
    Retourne True si Tmin est domine par au moins une etiquette de Lres,
    False sinon.
    (Dijkstra MO BD : boucle a arreter si True)
    
    :param T: liste des etiquettes temporaires
    :param Lres: liste des chemins Pareto-optimaux
    '''
    TF = T[0] # liste des etiquettes temporaires (forward)
    TB = T[1] # liste des etiquettes temporaires (backward)

    # Il n'y a plus d'etiquettes dans l'une des deux listes
    if not TF or not TB:
        return True

    # Forward : construire le vecteur de cout minimum pour chaque objectif a partir des vecteurs de T[0]
    
    TminF = TF[0].vector.copy() # vecteur de cout du premier element
    nb_dim: int = len(TminF) # nombre de classes / dimensions du vecteur de cout
    for i in range(1, len(TF)):
        for j in range(nb_dim):
            if TminF[j] > TF[i].vector[j]:
                TminF[j] = TF[i].vector[j]

    # Backward : construire le vecteur de cout minimum pour chaque objectif a partir des vecteurs de T[1]
    TminB = TB[0].vector.copy()
    for i in range(1, len(TB)): 
        for j in range(nb_dim):
            if TminB[j] > TB[i].vector[j]:
                TminB[j] = TB[i].vector[j]

    Tmin = [TminB[i] + TminF[i] for i in range(nb_dim)]
    labTmin = Label(None, Tmin, None, -1)
    Lres_labels = [Label(None, vect, None, -1) for (_, vect) in Lres]
    return labTmin.dominated_by_list(Lres_labels) 


### FONCTIONS LIEES AUX CHEMINS DANS DIJKSTRA MO DB ###

def reconstruireChemin(chemin):
    '''
    Retourn un chemin reconstruit, i.e.
    ([liste des sommets du chemin d'origine a destination], vecteur de cout total)
    
    :param chemin: (etiquette depuis l'origine, etiquette depuis la destination, vecteur de cout total)
    '''
    depuis_ori, depuis_dest, vect = chemin
    
    # Chemin jusqu'a origine
    chemin_ori = []
    sommet = depuis_ori.vertex
    label_prec = depuis_ori.prev_label
    sommet_union = sommet.name
    while label_prec != None:
        sommet = label_prec.vertex
        label_prec = label_prec.prev_label
        chemin_ori = [sommet.name] + chemin_ori

    # Chemin jusqu'a destination
    chemin_dest = []
    label_prec = depuis_dest.prev_label
    while label_prec != None:
        sommet = label_prec.vertex
        label_prec = label_prec.prev_label
        chemin_dest = chemin_dest + [sommet.name]
        
    return (chemin_ori + [sommet_union] + chemin_dest, vect)


def addResults(path, liste_res) -> None:
    """
    Reconstruit le chemin path et l'ajoute a liste_res s'il n'est pas domine par un chemin de liste_res.

    :param path: chemin a ajouter (depuis_ori, depuis_dest, vecteur cout)
    :param liste_res: liste des chemins (liste_sommets, vecteur cout) deja decouverts
    """
    liste_sommets, vec = reconstruireChemin(path)

    # Pour tout chemin r dans Lres
    for r in liste_res:
        liste_sommetsTemp, vecTemp = r

        # Chemin deja dans liste_res
        if liste_sommetsTemp == liste_sommets:
            return

        # Si le chemin est domine par le nouveau chemin path, on retire r
        if dominates(vec, vecTemp):
            liste_res.remove(r)

        # Si le nouveau chemin est domine, on ne change rien a Lres
        if dominates(vecTemp, vec):
            return
    
    liste_res.append((liste_sommets, vec))    


### GENERATION DE GRAPHES ALEATOIRES ###
    
def generate_random_graph(name: str, nbVertex: int, probaEdge: float, nbClasses: int):
    '''
    Genere un graphe aleatoire.
    
    :param name: nom du graphe
    :param nbVertex: nombre de sommets
    :param probaEdge: probabilite d'ajouter un arc pour chaque paire de sommets (distance entre 1 et 50)
    :param nbClasses: nombre de niveaux de securite / dimensions du vecteur de cout
    '''
    G = Graph(name, nbClasses)

    # Creation de nbVertex sommets
    for i in range(nbVertex):
        G.add_vertex(f"V{i}")

    # Creation d'arcs
    ascii_A = ord('A')
    for i in range(nbVertex):
        for j in range(nbVertex):
            if i != j and np.random.random() < probaEdge:
                G.add_edge(f"V{i}", f"V{j}", np.random.randint(1, 50), chr(ascii_A + np.random.randint(nbClasses)))

    return G


### TEST SUR DES GRAPHES GENERES ALEATOIREMENT ###

G = generate_random_graph("test", 5, 0.8, 3)
G.affiche_dico_adj()

vertices = list(G.vertices)
origin, dest = random.sample(vertices, 2) 
res = G.DijkstraMultiObjBidirectionnel(origin, dest)
print(f"ORIGINE = {origin.name}, ARRIVEE = {dest.name}")
print(res)
