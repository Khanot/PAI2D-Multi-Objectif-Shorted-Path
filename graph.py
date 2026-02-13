import numpy as np
from typing import Tuple, List, Dict
from collections import deque

class Vertex:

    def __init__(self, label: str) -> None:
        self.label: str = label
        self.liste=[[],[]]

    def __eq__(self, vertexPrime):
        if not isinstance(vertexPrime, Vertex):
            return False
        return self.label == vertexPrime.label  
    def __hash__(self):
        return hash(self.label)
    
    def coordonnees(self):
        l=self.label
        j=0
        x=-1
        while True:
            if l[j]=="-":
                if x==-1:
                    x=int(l[:j])
                    i=j+1
                else:
                    return x,int(l[i:j])
            j+=1
    def addLabel(self, label, direction):
        _, vector, _=label
        isDominated=False
        for i in range(len(self.liste[direction])):
            _, vectorTemp, _=self.liste[direction][i]
            if vectorComp(vector,vectorTemp):
                self.liste[direction]=self.liste[direction][:i]+self.liste[direction][i+1:]
                i-=1
            isDominated= isDominated or vectorComp(vectorTemp,vector)
        if not isDominated:
            self.liste[direction].append(label)



    
class Edge:

    def __init__(self, v1: Vertex, v2: Vertex,dist: int, classe: str) -> None:
        self.vertices: Tuple[Vertex, Vertex] = (v1, v2)
        self.label=(dist,classe)  #(5,"B")

class Graph:

    def __init__(self, name: str, nbClasses : int) -> None:
        self.name: str = name
        self.vertices: set[Vertex] = set()
        self.edges: set[Edge] = set()
        self.adj: [Dict[Vertex, set[Edge]],Dict[Vertex, set[Edge]]] = [dict(),dict()]
        self.nbClasses=nbClasses

    def copy(self): 
        """
        Renvoie une copie du graphe.
        """
        g=Graph(self.name) 
        E=self.edges 
        for v in self.vertices: 
            g.add_vertex(v.label) 
        for e in E: 
            g.add_edge(e.vertices[0].label, e.vertices[1].label) 

        g.adj=self.adj
        return g

    def add_vertex(self, label: str) -> Vertex:
        """
        Ajoute un sommet à un graphe s'il n'y est pas déjà.
        """
        v = Vertex(label)
        if v not in self.vertices:
            self.vertices.add(v)
            self.adj[0][v]=set()
            self.adj[1][v]=set()
        return v 

    def add_edge(self, labelv1: str, labelv2: str,dist: int, classe: str) -> None:
        """
        Ajoute une arête à un graphe si labelv1 et labelv2 sont des labels de sommets du graphe.
        """
        vertex1 = Vertex(labelv1)
        vertex2 = Vertex(labelv2)
        if vertex1 == vertex2:
            return #pas d'arêtes d'un sommet dans lui-même
  
        if vertex1 not in self.vertices or vertex2 not in self.vertices:
            raise ValueError("vertex1 ou vertex2 ne sont pas des sommets du graphe.")

        if vertex2 in self.adj[vertex1]:
            return #pas d'arêtes en double 
        e=Edge(vertex1,vertex2,dist,classe)
        self.edges.add(e)
        self.adj[0][vertex1].add(e)
        self.adj[1][vertex2].add(e)
        return 

    def delete_vertex(self, l: str) -> None:
        """
        Supprime le sommet de label l du graphe.
        """
        vertex = Vertex(l)
        if vertex not in self.vertices:
            return #le sommet n'est déjà pas dans le graphe

        self.adj = [(v, neighbors) for (v, neighbors) in self.adj if v != vertex]
        self.edges = [e for e in self.edges if vertex not in e.vertices]

        for (_, neighbors) in self.adj:
            if vertex in neighbors:
                neighbors.remove(vertex) #suppression du sommet dans les voisins des autres sommets
        self.vertices.remove(vertex)
        return None

    def delete_vertices(self, label_set) -> None:
        """
        Supprime l'ensemble des sommets du graphe dont les labels sont présents dans label_set
        """
        for l in label_set:
            self.delete_vertex(l)

        return None

    def degrees(self) -> List:
        """
        Renvoie un tableau contenant les tuples (sommet s, degré(s)) pour les sommets du graphe
        """
        return [(v, len(neighbors)) for (v, neighbors) in self.adj]

    def max_degree(self) -> str:
        """
        Renvoie le label d'un sommet de degré maximum du graphe
        """
        deg=self.degrees()
        return [x for x in deg if x[1]==max([lenNeighbors[1] for lenNeighbors in deg])][0][0].label

    def print_al(self) -> None:
        """
        Affiche le tableau des listes d'adjacence du graphe
        """
        res=""
        for v in self.adj.keys():
            res=str(v.label)+" -> ["
            for e in self.adj[v]:
                res+=e.vertices[1].label+"("+str(e.label[0])+","+e.label[1]+"), "
            print(res[:-2]+"]")


    def nbVertices(self) -> int:
        """
        Renvoie le nombre de sommets du graphe
        """
        return len(self.vertices)

    def nbEdges(self) -> int:
        """
        Renvoie le nombre d'arêtes du graphe
        """
        return len(self.edges)
    def is_cover(self,verticesSet):
        for e in self.edges:
            if e.vertices[0].label not in verticesSet and e.vertices[1].label not in verticesSet:
                
                return False
        return True
    
    
    def BFS(self, i0, j0, ori, i, j):

        v1 = None
        for v in self.vertices:
            if v.label == f'{i0}-{j0}-{ori}':
                v1 = v
                break

        if v1 is None:
            return []

        file = deque([v1])
        vu = {v1}

        parent = {v1: None}
        action = {v1: "Start"}

        while file:
            courant = file.popleft()

            for elem, act in self.adj[courant]:
                if elem not in vu:
                    vu.add(elem)
                    parent[elem] = courant
                    action[elem] = act

                    i1, j1 = elem.coordonnees()
                    if i1 == i and j1 == j:
                        chemin = []
                        while elem is not None:
                            chemin.append((elem, action[elem]))
                            elem = parent[elem]
                        return list(reversed(chemin))

                    file.append(elem)

        return []
    def getNeighbours(self,vertex,dir):
        return [e for e in self.adj[dir][vertex]]
    
    def DijkstraMultiObjBidirectionnel(self,origin: Vertex, dest:Vertex):
        T=[[],[]]
        Lres=[]

        originLabel=(origin,[0 for _ in range(self.nbClasses)],None)
        origin.addLabel(originLabel,0)
        T[0].append(originLabel)

        destLabel=(dest,[0 for _ in range(self.nbClasses)],None)
        dest.addLabel(destLabel,1)
        T[1].append(destLabel)

        d=1
        while not (stop(T,Lres)):
            d=1-d
            label=T[d][0]
            T[d]=T[d][1:]
            owner=label[0]
            neighbours=self.getNeighbours(owner,d)
            for e in neighbours:
                n=e.vertices[1]
                newLabel=(e.vertices[1], cost(label,e,self.nbClasses) ,label)
                if not compListe(newLabel, n.liste[d]):
                    n.addLabel(newLabel,d)
                    T[d].append(newLabel)
                    if n.liste[1-d]!=[]:
                        comb=combine(newLabel,n.liste[1-d])
                        for c in comb:
                            addResults(c,Lres)

        return Lres
    def labelToString(self,label):
        res="("+label[0].label+","+str(label[1])
        if label[2]==None:
            return res+"None)"
        else:
            return res+self.labelToString(label[2])+")"
    def reconstruireChemin(self, L):
        res="("
        for l in L:
            res+=self.labelToString(l)+"------"
        return res+")"
    


def addResults(label, Lres):
        """
        Add the label to Lres if not dominated and remove the dominated labels

        """
        _, vector, _=label
        isDominated=False
        for r in Lres:
            _, vectorTemp, _=r
            if vectorComp(vector,vectorTemp):
                Lres.remove(r)
            isDominated= isDominated or vectorComp(vectorTemp,vector)
        if not isDominated:
            Lres.append(label)                    
def combine(label,labelListe):
    """
    Returns a list of combined paths costs between a path and a list of paths
    
    """
    res=[]
    n,l,nprime=label
    for i in range(len(labelListe)):
        _,ll,_=labelListe[i]
        res.append([l[j]+ll[j] for j in range(len(l))])
    return [(n,res[i],nprime) for i in range(len(res))]
    
def cost(label,edge,nbClasses):
    _,vector,_=label
    classe=ord(edge.label[1])-65
    dist=edge.label[0]
    return [dist + vector[i]  if i<=classe else vector[i] for i in range(nbClasses)]

"""
def removeMin(T):
    if len(T[0])==0:
        return T[0]
    min=T[0][0]
    minIndex=0
    egalite=False
    for i in range(1,len(T)):
        if T[i][0]<min:
            min=T[i][0]
            minIndex=i
        if 


            [T[i][1:] for ]
"""
def compListe(label, labelListe):
    """Returns True if label is dominated by at least one label in labelListe"""
    for i in range(len(labelListe)):
        if vectorComp(labelListe[i][1],label):
            return True
    return False

def stop(T,Lres):
    TminF=T[0][0][1]
    n=len(TminF)
    for i in range(1,len(T[0])):
        for j in range(n):
            if TminF[j]>T[0][i][1][j]:
                TminF[j]=T[0][i][1][j]
    TminB=T[1][0][1]
    for i in range(1,len(T[1])):
        for j in range(n):
            if TminB[j]>T[1][i][1][j]:
                TminB[j]=T[1][i][1][j]
    Tmin=[TminB[i]+TminF[i] for i in range(n)]
    return compListe(Tmin,Lres)
    


def vectorComp(v1,v2):

    "Returns True if v1 domainates v2"
    return np.all(np.array(v1)-np.array(v2)>=0)
       
    


    
def generate_random_graph(name,nbVertex,probaEdge,nbClasses):
    G=Graph(name,nbClasses)
    for i in range(nbVertex):
        G.add_vertex(f"V{i}")
    cpt=0
    for i in range(nbVertex):
        for j in range(nbVertex):
            if i!=j and np.random.random()<probaEdge:
                G.add_edge(f"V{i}", f"V{j}",np.random.randint(1,50),chr(65+np.random.randint(nbClasses)))
                cpt+=1
    return G






G=generate_random_graph("test",10,0.5,5)
G.print_al()
res=G.DijkstraMultiObjBidirectionnel(next(iter(G.vertices)),next(iter(G.vertices)))
print(G.reconstruireChemin(res))






