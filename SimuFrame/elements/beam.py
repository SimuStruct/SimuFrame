# Third-party libraries
import numpy as np

# Author's libraries
from SimuFrame.core.model import Structure

"""
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
        coord = np.array([
            [0, 0, 0],
            [3, 0, 0],
            [6, 0, 0]
        ])

        # Matriz de conectividade
        conec = np.array([
            [0, 1],
            [1, 2]
        ])

        # Definir os índices dos apoios
        condicoes_contorno = {
            'FIXOXZ': [0, 2],
        }

        # Definir o modelo estrutural (viga ou treliça)
        modelo = "viga"

        # Definir parâmetros constitutivos e geométricos
        secoes = {
            range(len(conec)): {"geometry_type": "retangular", "E": 2500, "nu": 0.3, "base": 1.0, "altura": 0.12},
        }

        # Criar os dados do elemento estrutural
        estrutura = Structure(analise, modelo, coord, conec, secoes, n)
        estrutura.define_supports(condicoes_contorno)

        # Adicionar cargas
        estrutura.add_distributed_loads({
            range(len(conec)): [[0, 0, -0.2168], [0, 0, -0.2168]]
        })


    elif caso == 2:
        # Matriz de coordenadas dos pontos (x, y, z)
        coord = np.array([
            [0, 0, 0],
            [4, 0, 0],
            [8, 0, 0]
        ])

        # Matriz de conectividade
        conec = np.array([
            [0, 1],
            [1, 2]
        ])

        # Definir os member_releases nodes elements (se houver)
        releases = {
            'FREE-HINGE': [0],
        }

        # Definir os índices dos apoios
        condicoes_contorno = {'FIXOXZ': [0, 1, 2]}

        # Definir o modelo estrutural (viga ou treliça)
        modelo = "viga"

        # Definir parâmetros constitutivos e geométricos
        secoes = {
            range(len(conec)): {"geometry_type": "retangular", "E": 2.1e7, "nu": 0.3, "base": 0.1, "altura": 0.1},
        }

        # Criar os dados do elemento estrutural
        estrutura = Structure(analise, modelo, coord, conec, secoes, n, releases)
        estrutura.define_supports(condicoes_contorno)

        # Adicionar cargas
        # estrutura.NLOAD({
        #     1: [0, 0, -0.5, 0, 0, 0]
        # })

        estrutura.add_distributed_loads({
            range(len(conec)): [[0, 0, -10], [0, 0, -10]]
        })

    elif caso == 3:
        # Matriz de coordenadas dos pontos (x, y, z)
        coord = np.array([
            [0, 0, 0],
            [6, 0, 0]
        ])

        # Matriz de conectividade
        conec = np.array([
            [0, 1],
        ])

        # Definir os índices dos apoios
        condicoes_contorno = {'ENGASTE': [0]}

        # Definir o modelo estrutural (viga ou treliça)
        modelo = "viga"

        # Definir parâmetros constitutivos e geométricos
        secoes = {
            range(len(conec)): {"geometry_type": "retangular", "E": 2.7e3, "nu": 0.2, "base": 0.20, "altura": 0.40},
        }

        # Criar os dados do elemento estrutural
        estrutura = Structure(analise, modelo, coord, conec, secoes, n)
        estrutura.define_supports(condicoes_contorno)

        # Adicionar cargas
        estrutura.add_nodal_loads({
            1: [0, 0, -10, 0, 0, 0]
        })


    elif caso == 4:
        # Matriz de coordenadas dos pontos (x, y, z)
        coord = np.array([
            [0, 0, 0],
            [3, 0, 0],
            [6, 0, 0]
        ])

        # Matriz de conectividade
        conec = np.array([
            [0, 1],
            [1, 2]
        ])

        # Definir os índices dos apoios
        condicoes_contorno = {
            'ENGASTE': [0, 2]
        }

        # Definir o modelo estrutural (viga ou treliça)
        modelo = "viga"

        # Definir parâmetros constitutivos e geométricos
        secoes = {
            range(len(conec)): {"geometry_type": "retangular", "E": 2.7e7, "nu": 0.3, "base": 0.2, "altura": 0.4},
        }

        # Criar os dados do elemento estrutural
        estrutura = Structure(analise, modelo, coord, conec, secoes, n)
        estrutura.define_supports(condicoes_contorno)

        # Adicionar cargas
        estrutura.add_distributed_loads({
            range(len(conec)): [[0, -100, 0], [0, -100, 0]]
        })

    return estrutura
