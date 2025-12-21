# Third-party libraries
import numpy as np

# Author's libraries
from SimuFrame.core.model import Structure

"""
Casos de estudo:    1: Treliça espacial 1;
                    2: Treliça plana;
                    3: Treliça espacial 2;
                    4: Treliça espacial 3;

As condições de contorno são definidas conforme os termos:

    XSYMM: Simetria em x (U = θy = θz = 0)
    YSYMM: Simetria em y (V = θx = θz = 0)
    ZSYMM: Simetria em z (W = θx = θy = 0)
    XASYMM: Antissimetria em x (V = W = θx = 0)
    YASYMM: Antissimetria em y (U = W = θy = 0)
    ZASYMM: Antissimetria em z (U = V = θz = 0)
    ARTICULADO: Apoio articulado: (U = V = W = 0)
    FIXOXY: Apoio fixo em x e y: (U = V = W = θx = θy = 0)
    FIXOXZ: Apoio fixo em x e z: (U = Z = W = θx = θz = 0)
    FIXOYZ: Apoio fixo em y e z: (V = Z = W = θy = θz = 0)
    ENGASTE: Apoio engastado (U = V = W = θx = θy = θz = 0)
"""


# Adicionar cargas e definir parâmetros constitutivos e geométricos
def exemplo(analise, caso, n):
    if caso == 1:
        # Matriz de coordenadas dos pontos (x, y, z)
        coord = np.array([[0, 0, 0],
                          [2, 0, 0],
                          [4, 0, 0],

                          [0, 2, 0],
                          [2, 2, 0],
                          [4, 2, 0],

                          [0, 0, 2],
                          [2, 0, 2],
                          [4, 0, 2],

                          [0, 2, 2],
                          [2, 2, 2],
                          [4, 2, 2]])

        # Matriz de conectividade
        conec = np.array([[0, 1],
                          [1, 2],

                          [3, 4],
                          [4, 5],

                          [6, 7],
                          [7, 8],

                          [9, 10],
                          [10, 11],

                          [0, 6],
                          [1, 7],
                          [2, 8],

                          [3, 9],
                          [4, 10],
                          [5, 11],

                          [0, 3],
                          [1, 4],
                          [2, 5],

                          [6, 9],
                          [8, 11],

                          [6, 1],
                          [8, 1],

                          [9, 4],
                          [11, 4]])

        # Definir os índices dos apoios
        condicoes_contorno = {'XSYMM': [],
                              'YSYMM': [1, 4, 6, 7, 8, 9, 10, 11],
                              'ZSYMM': [],
                              'XASYMM':[],
                              'YASYMM': [],
                              'ZASYMM': [],
                              'ARTICULADO': [],
                              'FIXOXY': [],
                              'FIXOXZ': [],
                              'FIXOYZ': [],
                              'ENGASTE': [0, 2, 3, 5]
                            }

        # Definir o modelo estrutural (viga ou treliça)
        modelo = "treliça"

        # Definir parâmetros constitutivos e geométricos
        secoes = {
            range(len(conec)): {"geometry_type": "tubular", "E": 2.1e8, "nu": 0.3, "raio_ext": 0.05, "raio_int": 0.04},
        }

        # Criar os dados do elemento estrutural
        estrutura = Structure(analise, modelo, coord, conec, secoes, n)
        estrutura.define_supports(condicoes_contorno)

        # Adicionar cargas
        estrutura.add_nodal_loads({
            range(12): [0, 0, -30, 0, 0, 0],
        })

    elif caso == 2:
        # Matriz de coordenadas dos pontos (x, y, z)
        coord = np.array([[0, 0, 0],
                          [4, 0, 0],
                          [8, 0, 0],
                          [12, 0, 0],
                          [2, 0, 4],
                          [6, 0, 4],
                          [10, 0, 4]])

        # Matriz de conectividade
        conec = np.array([[0, 1],
                          [1, 2],
                          [2, 3],

                          [0, 4],
                          [4, 1],
                          [1, 5],
                          [5, 2],
                          [2, 6],
                          [6, 3],

                          [4, 5],
                          [5, 6]])

        # Definir os índices dos apoios
        condicoes_contorno = {'XSYMM': [],
                              'YSYMM': [1, 2, 4, 5, 6],
                              'ZSYMM': [],
                              'XASYMM': [],
                              'YASYMM': [],
                              'ZASYMM': [],
                              'ARTICULADO': [],
                              'FIXOXY': [],
                              'FIXOXZ': [],
                              'FIXOYZ': [],
                              'ENGASTE': [0, 3]
                            }

        # Definir o modelo estrutural (viga ou treliça)
        modelo = "treliça"

        # Definir parâmetros constitutivos e geométricos
        secoes = {
            range(len(conec)): {"geometry_type": "circular", "E": 2.1e8, "nu": 0.3, "raio": 0.05},
        }

        # Criar os dados do elemento estrutural
        estrutura = Structure(analise, modelo, coord, conec, secoes, n)
        estrutura.define_supports(condicoes_contorno)

        # Adicionar cargas
        estrutura.add_nodal_loads({
            range(len(coord)): [0, 0, -100, 0, 0, 0],
        })

    elif caso == 3:
        # Matriz de coordenadas dos pontos (x, y, z)
        coord = np.array([[0, 0, 0],
                          [2, 0, 0],
                          [4, 0, 0],
                          [6, 0, 0],
                          [8, 0, 0],
                          [10, 0, 0],
                          [12, 0, 0],
                          [14, 0, 0],
                          [16, 0, 0],
                          [18, 0, 0],
                          [20, 0, 0],

                          [0, 2, 0],
                          [2, 2, 0],
                          [4, 2, 0],
                          [6, 2, 0],
                          [8, 2, 0],
                          [10, 2, 0],
                          [12, 2, 0],
                          [14, 2, 0],
                          [16, 2, 0],
                          [18, 2, 0],
                          [20, 2, 0],

                          [2, 0, 2],
                          [4, 0, 2],
                          [6, 0, 2],
                          [8, 0, 2],
                          [10, 0, 2],
                          [12, 0, 2],
                          [14, 0, 2],
                          [16, 0, 2],
                          [18, 0, 2],

                          [2, 2, 2],
                          [4, 2, 2],
                          [6, 2, 2],
                          [8, 2, 2],
                          [10, 2, 2],
                          [12, 2, 2],
                          [14, 2, 2],
                          [16, 2, 2],
                          [18, 2, 2]])

        # Matriz de conectividade
        conec = np.array([
                        # Banzo inferior (frontal)
                        [0, 1],
                        [1, 2],
                        [2, 3],
                        [3, 4],
                        [4, 5],
                        [5, 6],
                        [6, 7],
                        [7, 8],
                        [8, 9],
                        [9, 10],

                        # Banzo inferior (posterior)
                        [11, 12],
                        [12, 13],
                        [13, 14],
                        [14, 15],
                        [15, 16],
                        [16, 17],
                        [17, 18],
                        [18, 19],
                        [19, 20],
                        [20, 21],

                        # Banzo superior (frontal)
                        [22, 23],
                        [23, 24],
                        [24, 25],
                        [25, 26],
                        [26, 27],
                        [27, 28],
                        [28, 29],
                        [29, 30],

                        # Banzo superior (posterior)
                        [31, 32],
                        [32, 33],
                        [33, 34],
                        [34, 35],
                        [35, 36],
                        [36, 37],
                        [37, 38],
                        [38, 39],

                        # Transversinas inferiores
                        [0, 11],
                        [1, 12],
                        [2, 13],
                        [3, 14],
                        [4, 15],
                        [5, 16],
                        [6, 17],
                        [7, 18],
                        [8, 19],
                        [9, 20],
                        [10, 21],

                        # Transversinas superiores
                        [22, 31],
                        [23, 32],
                        [24, 33],
                        [25, 34],
                        [26, 35],
                        [27, 36],
                        [28, 37],
                        [29, 38],
                        [30, 39],

                        # Diagonais inferiores
                        [0, 12],
                        [12, 2],
                        [2, 14],
                        [14, 4],
                        [4, 16],
                        [16, 6],
                        [6, 18],
                        [18, 8],
                        [8, 20],
                        [20, 10],

                        [22, 32],
                        [32, 24],
                        [24, 34],
                        [34, 26],
                        [26, 36],
                        [36, 28],
                        [28, 38],
                        [38, 30],

                        # Montantes frontais
                        [1, 22],
                        [2, 23],
                        [3, 24],
                        [4, 25],
                        [5, 26],
                        [6, 27],
                        [7, 28],
                        [8, 29],
                        [9, 30],

                        # Montantes posteriores
                        [12, 31],
                        [13, 32],
                        [14, 33],
                        [15, 34],
                        [16, 35],
                        [17, 36],
                        [18, 37],
                        [19, 38],
                        [20, 39],

                        # Diagonais frontais
                        [0, 22],
                        [22, 2],
                        [23, 3],
                        [24, 4],
                        [25, 5],
                        [27, 5],
                        [28, 6],
                        [29, 7],
                        [30, 8],
                        [30, 10],

                        # Diagonais posteriores
                        [11, 31],
                        [31, 13],
                        [32, 14],
                        [33, 15],
                        [34, 16],
                        [36, 16],
                        [37, 17],
                        [38, 18],
                        [39, 19],
                        [39, 21]])

        # Definir os índices dos apoios
        condicoes_contorno = {'XSYMM': [],
                              'YSYMM': [],
                              'ZSYMM': [],
                              'XASYMM': [],
                              'YASYMM': [],
                              'ZASYMM': [],
                              'ARTICULADO': [],
                              'FIXOXY': [],
                              'FIXOXZ': [],
                              'FIXOYZ': [],
                              'ENGASTE': [0, 10, 11, 21]
                            }

        # Definir o modelo estrutural (viga ou treliça)
        modelo = "viga"

        # Add member_releases
        releases = {
            'HINGE-HINGE': range(len(conec))
        }

        # Definir parâmetros constitutivos e geométricos
        secoes = {
            range(len(conec)): {"geometry_type": "tubular", "E": 2.1e8, "nu": 0.3, "raio_ext": 0.05, "raio_int": 0.04},
        }

        # Criar os dados do elemento estrutural
        estrutura = Structure(analise, modelo, coord, conec, secoes, n, releases)
        estrutura.define_supports(condicoes_contorno)

        # Adicionar cargas (banzo inferior e superior, respectivamente)
        estrutura.add_nodal_loads({
            range(0, 22):  [0, 0, -30, 0, 0, 0],
            range(22, 40): [0, 0, -15, 0, 0, 0]
        })

    elif caso == 4:
        # Matriz de coordenadas dos pontos (x, y, z)
        coord = np.array([[0, 0, 0],
                          [2, 0, 0],
                          [4, 0, 0],
                          [6, 0, 0],
                          [8, 0, 0],
                          [10, 0, 0],
                          [12, 0, 0],
                          [14, 0, 0],
                          [16, 0, 0],
                          [18, 0, 0],
                          [20, 0, 0],

                          [0, 2, 0],
                          [2, 2, 0],
                          [4, 2, 0],
                          [6, 2, 0],
                          [8, 2, 0],
                          [10, 2, 0],
                          [12, 2, 0],
                          [14, 2, 0],
                          [16, 2, 0],
                          [18, 2, 0],
                          [20, 2, 0],

                          [0, 0, 2],
                          [2, 0, 2],
                          [4, 0, 2],
                          [6, 0, 2],
                          [8, 0, 2],
                          [10, 0, 2],
                          [12, 0, 2],
                          [14, 0, 2],
                          [16, 0, 2],
                          [18, 0, 2],
                          [20, 0, 2],

                          [0, 2, 2],
                          [2, 2, 2],
                          [4, 2, 2],
                          [6, 2, 2],
                          [8, 2, 2],
                          [10, 2, 2],
                          [12, 2, 2],
                          [14, 2, 2],
                          [16, 2, 2],
                          [18, 2, 2],
                          [20, 2, 2]])

        # Matriz de conectividade
        conec = np.array([
                        # Banzo inferior (frontal)
                        [0, 1],
                        [1, 2],
                        [2, 3],
                        [3, 4],
                        [4, 5],
                        [5, 6],
                        [6, 7],
                        [7, 8],
                        [8, 9],
                        [9, 10],

                        # Banzo inferior (posterior)
                        [11, 12],
                        [12, 13],
                        [13, 14],
                        [14, 15],
                        [15, 16],
                        [16, 17],
                        [17, 18],
                        [18, 19],
                        [19, 20],
                        [20, 21],

                        # Banzo superior (frontal)
                        [22, 23],
                        [23, 24],
                        [24, 25],
                        [25, 26],
                        [26, 27],
                        [27, 28],
                        [28, 29],
                        [29, 30],
                        [30, 31],
                        [31, 32],

                        # Banzo superior (posterior)
                        [33, 34],
                        [34, 35],
                        [35, 36],
                        [36, 37],
                        [37, 38],
                        [38, 39],
                        [39, 40],
                        [40, 41],
                        [41, 42],
                        [42, 43],

                        # Transversinas inferiores
                        [0, 11],
                        [1, 12],
                        [2, 13],
                        [3, 14],
                        [4, 15],
                        [5, 16],
                        [6, 17],
                        [7, 18],
                        [8, 19],
                        [9, 20],
                        [10, 21],

                        # Transversinas superiores
                        [22, 33],
                        [23, 34],
                        [24, 35],
                        [25, 36],
                        [26, 37],
                        [27, 38],
                        [28, 39],
                        [29, 40],
                        [30, 41],
                        [31, 42],
                        [32, 43],

                        # Diagonais inferiores
                        [0, 12],
                        [12, 2],
                        [2, 14],
                        [14, 4],
                        [4, 16],
                        [16, 6],
                        [6, 18],
                        [18, 8],
                        [8, 20],
                        [20, 10],

                        # Diagonais superiores
                        [33, 23],
                        [23, 35],
                        [35, 25],
                        [25, 37],
                        [37, 27],
                        [27, 39],
                        [39, 29],
                        [29, 41],
                        [41, 31],
                        [31, 43],

                        # Montantes frontais
                        [0, 22],
                        [1, 23],
                        [2, 24],
                        [3, 25],
                        [4, 26],
                        [5, 27],
                        [6, 28],
                        [7, 29],
                        [8, 30],
                        [9, 31],
                        [10, 32],

                        # Montantes posteriores
                        [11, 33],
                        [12, 34],
                        [13, 35],
                        [14, 36],
                        [15, 37],
                        [16, 38],
                        [17, 39],
                        [18, 40],
                        [19, 41],
                        [20, 42],
                        [21, 43],

                        # Diagonais frontais
                        [22, 1],
                        [23, 2],
                        [24, 3],
                        [25, 4],
                        [26, 5],
                        [5, 28],
                        [6, 29],
                        [7, 30],
                        [8, 31],
                        [9, 32],

                        # Diagonais posteriores
                        [33, 12],
                        [34, 13],
                        [35, 14],
                        [36, 15],
                        [37, 16],
                        [16, 39],
                        [17, 40],
                        [18, 41],
                        [19, 42],
                        [20, 43]])

        # Definir os índices dos apoios
        condicoes_contorno = {'XSYMM': [],
                              'YSYMM': [22, 32, 33, 43],
                              'ZSYMM': [],
                              'XASYMM': [],
                              'YASYMM': [],
                              'ZASYMM': [],
                              'ARTICULADO': [],
                              'FIXOXY': [],
                              'FIXOXZ': [],
                              'FIXOYZ': [],
                              'ENGASTE': [0, 10, 11, 21]
                              }

        # Definir o modelo estrutural (viga ou treliça)
        modelo = "treliça"

        # Definir parâmetros constitutivos e geométricos
        secoes = {
            range(len(conec)): {"geometry_type": "tubular", "E": 2.1e8, "nu": 0.3, "raio_ext": 0.05, "raio_int": 0.04},
        }

        # Criar os dados do elemento estrutural
        estrutura = Structure(analise, modelo, coord, conec, secoes, n)
        estrutura.define_supports(condicoes_contorno)

        # Adicionar cargas (banzo inferior  superior, respectivamente)
        estrutura.add_nodal_loads({
            range(0, 22):  [0, 0, -30, 0, 0, 0],
            range(22, 44): [0, 0, -15, 0, 0, 0]
        })
    
    elif caso == 5:
        # Matriz de coordenadas dos pontos (x, y, z)
        coord = np.array([[0, 0, 0],
                          [4, 0, 0],
                          [0, 0, 2],
                          [0, 0, 4],
                          [2, 0, 2]])

        # Matriz de conectividade
        conec = np.array([[0, 4],
                          [1, 4],
                          [2, 4],
                          [3, 4]])

        # Definir os índices dos apoios
        condicoes_contorno = {'XSYMM': [],
                              'YSYMM': range(len(coord)),
                              'ZSYMM': [],
                              'XASYMM': [],
                              'YASYMM': [],
                              'ZASYMM': [],
                              'ARTICULADO': [],
                              'FIXOXY': [],
                              'FIXOXZ': [0, 1, 2, 3],
                              'FIXOYZ': [],
                              'ENGASTE': []
                            }

        # Definir o modelo estrutural (viga ou treliça)
        modelo = "treliça"

        # Definir parâmetros constitutivos e geométricos
        secoes = {
            range(len(conec)): {"geometry_type": "circular", "E": 2.1e5, "nu": 0.3, "raio": 0.01},
        }

        # Criar os dados do elemento estrutural
        estrutura = Structure(analise, modelo, coord, conec, n)
        estrutura.define_supports(condicoes_contorno)

        # Adicionar cargas
        estrutura.add_nodal_loads({
            4: [10, 0, 0, 0, 0, 0]
        })

    elif caso == 6:
        # Matriz de coordenadas dos pontos (x, y, z)
        coord = np.array([[0, 0, 0],
                          [3, 0, 0],
                          [6, 0, 0],
                          [9, 0, 0],
                          [12, 0, 0],
                          [15, 0, 0],
                          [18, 0, 0],

                          [0, 0, 1 + (0.8 / 9) * 0],
                          [3, 0, 1 + (0.8 / 9) * 3],
                          [6, 0, 1 + (0.8 / 9) * 6],
                          [9, 0, 1 + (0.8 / 9) * 9],
                          [12, 0, 1.8 - (0.8 / 9) * 3],
                          [15, 0, 1.8 - (0.8 / 9) * 6],
                          [18, 0, 1.8 - (0.8 / 9) * 9]])

        # Matriz de conectividade
        conec = np.array([[0, 1],
                          [1, 2],
                          [2, 3],
                          [3, 4],
                          [4, 5],
                          [5, 6],

                          [0, 7],
                          [1, 8],
                          [2, 9],
                          [3, 10],
                          [4, 11],
                          [5, 12],
                          [6, 13],

                          [7, 8],
                          [8, 9],
                          [9, 10],
                          [10, 11],
                          [11, 12],
                          [12, 13],

                          [7, 1],
                          [8, 2],
                          [9, 3],
                          [11, 3],
                          [12, 4],
                          [13, 5]])

        # Definir os índices dos apoios
        condicoes_contorno = {
            'YSYMM': range(len(coord)),
            'FIXOXZ': [0, 6],
            }

        # Definir member_releases
        releases = {
            'HINGE-HINGE': list(range(6, 13)) + list(range(19, 25)),
            }

        # Definir o modelo estrutural (viga ou treliça)
        modelo = "viga"

        # Definir parâmetros constitutivos e geométricos
        secoes = {
            range(len(conec)): {"geometry_type": "circular", "E": 2.1e7, "nu": 0.3, "raio": 0.1},
            # range(len(connectivity)): {"geometry_type": 'retangular', "E": 2.1e7, "nu": 0.3,  "base": 0.1, "altura": 0.1}
        }

        # Criar os dados do elemento estrutural
        estrutura = Structure(analise, modelo, coord, conec, secoes, n, releases)
        estrutura.define_supports(condicoes_contorno)

        # Adicionar cargas
        estrutura.add_nodal_loads({
            range(7, 14): [0, 0, -8.2, 0, 0, 0],
        })

    elif caso == 8:
        # Matriz de coordenadas dos pontos (x, y, z)
        coord = np.array([[0, 0, 0],
                          [3, 0, 0],
                          [6, 0, 0],
                          [12, 0, 0],
                          [15, 0, 0],
                          [18, 0, 0],

                          [3, 0, (4.5 / 9) * 3],
                          [6, 0, (4.5 / 9) * 6],
                          [9, 0, (4.5 / 9) * 9],
                          [12, 0, 4.5 - (4.5 / 9) * 3],
                          [15, 0, 4.5 - (4.5 / 9) * 6]])

        # Matriz de conectividade
        conec = np.array([[0, 1],
                          [1, 2],
                          [2, 3],
                          [3, 4],
                          [4, 5],

                          [0, 6],
                          [6, 7],
                          [7, 8],
                          [8, 9],
                          [9, 10],
                          [10, 5],

                          [1, 6],
                          [2, 7],
                          [3, 9],
                          [4, 10],

                          [1, 7],
                          [2, 8],
                          [8, 3],
                          [9, 4]])

        # Definir os índices dos apoios
        condicoes_contorno = {
            'YSYMM': range(len(coord)),
            'ARTICULADO': [0],
            }

        # Definir member_releases
        releases = {
            'HINGE-HINGE': range(len(conec))
        }

        # Definir o modelo estrutural (viga ou treliça)
        modelo = "viga"

        # Definir parâmetros constitutivos e geométricos
        secoes = {
            range(len(conec)): {"geometry_type": "circular", "E": 2.1e7, "nu": 0.3, "raio": 0.1},
        }

        # Criar os dados do elemento estrutural
        estrutura = Structure(analise, modelo, coord, conec, secoes, n, releases)
        estrutura.define_supports(condicoes_contorno)

        # Adicionar cargas
        estrutura.add_nodal_loads({
            0: [0, 0, -7.65/2, 0, 0, 0],
            (6, 7, 8, 9, 10, 5): [0, 0, -7.65, 0, 0, 0],
        })

    elif caso == 7:
        # Matriz de coordenadas dos pontos (x, y, z)
        coord = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        # Matriz de conectividade
        conec = np.array([
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [2, 3],
            [3, 1]
        ])

        # Definir os índices dos apoios
        condicoes_contorno = {
            'ARTICULADO': [0],
            'ROLETE_X': [1],
            'ZSYMM': [2]
            }

        # Definir member_releases
        releases = {
            'HINGE-HINGE': range(len(conec))
        }

        # Definir o modelo estrutural (viga ou treliça)
        modelo = "viga"

        # Definir parâmetros constitutivos e geométricos
        secoes = {
            range(len(conec)): {"geometry_type": "circular", "E": 1000000, "nu": 0.0, "raio": 0.005},
        }

        # Criar os dados do elemento estrutural
        estrutura = Structure(analise, modelo, coord, conec, secoes, n, releases)
        estrutura.define_supports(condicoes_contorno)

        # Adicionar cargas
        estrutura.add_nodal_loads({
            3: [1, 1, -1, 0, 0, 0],
        })

    return estrutura

