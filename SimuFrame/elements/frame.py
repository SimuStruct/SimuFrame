# Third-party libraries
import numpy as np

# Author's libraries
# from SimuFrame.core.model_type import Structure

"""
Casos de estudo:    Pórtico 1: Viga biapoiada com uma carga distribuída sobre toda a viga, utilizando parâmetros constitutivos de FIGUEIRAS;
                    Pórtico 2: Viga biapoiada com uma carga concentrada no centro, utilizando parâmetros constitutivos de FIGUEIRAS;
                    Pórtico 3: Viga em balanço com uma carga concentrada na extremidade livre, utilizando parâmetros do concreto C30;
                    Pórtico 4: Viga biapoiada com uma carga concentrada no centro, utilizando parâmetros do concreto C30;

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
                          [0, 0, 4],
                          [4, 0, 4]])

        # Matriz de conectividade
        conec = np.array([[0, 1],
                          [1, 2]])

        # Definir os índices dos apoios
        condicoes_contorno = {
            'ENGASTE': [0, 2],
            }

        # Definir o modelo estrutural (viga ou treliça)
        modelo = "viga"

        # Definir parâmetros constitutivos e geométricos
        secoes = {
            range(len(conec)): {"geometry_type": "retangular", "E": 2e6, "nu": 0.2, "base": 0.1, "altura": 0.1},
            # range(len(connectivity)): {"geometry_type": "circular", "E": 2e6, "v": 0.0, "raio": 0.05},
        }

        # Criar os dados do elemento estrutural
        estrutura = Structure(analise, modelo, coord, conec, secoes, n)
        estrutura.define_supports(condicoes_contorno)

        # Adicionar cargas
        estrutura.add_nodal_loads({
            1: [0, 10, 0, 0, 0, 0]
            # 2: [0, 0, -17, 0, 0, 0]
        })
    
    if caso == 2:
        # Matriz de coordenadas dos pontos (x, y, z)
        coord = np.array([[0, 0, 0],
                          [4, 0, 0],
                          [0, 4, 0],
                          [4, 4, 0],
                          [0, 0, 4],
                          [4, 0, 4],
                          [0, 4, 4],
                          [4, 4, 4]])

        # Matriz de conectividade
        conec = np.array([[0, 4],
                          [1, 5],
                          [2, 6],
                          [3, 7],
                          [4, 5],
                          [6, 7],
                          [4, 6],
                          [5, 7]])

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
                              'ENGASTE': [0, 1, 2, 3]
                              }

        # Definir o modelo estrutural (viga ou treliça)
        modelo = "viga"

        # Definir parâmetros constitutivos e geométricos
        secoes = {
            range(len(conec)): {"geometry_type": "retangular", "E": 2.7e7, "nu": 0.2, "base": 0.2, "altura": 0.2},
        }

        # Criar os dados do elemento estrutural
        estrutura = Structure(analise, modelo, coord, conec, secoes, n)
        estrutura.define_supports(condicoes_contorno)

        # Adicionar cargas
        estrutura.add_distributed_loads({
            # (4, 5): [[0, 100, 0], [0, 100, 0]],
            (6, 7): [[100, 0, 0], [100, 0, 0]]
        })
        # estrutura.NLOAD({
        #     range(4, 8): [0, 0, -100, 0, 0, 0]
        # })

    if caso == 3:
        # Parâmetros da Estrutura
        num_pavimentos = 4
        num_niveis = num_pavimentos + 1

        # Posições dos pilares nodes eixos X e Y
        coords_x = [0, 5, 10]
        coords_y = [0, 5, 10]
        altura_piso = 4.0

        # Geração da Matriz de Coordenadas
        lista_coords = []
        nos_por_pavimento = len(coords_x) * len(coords_y)

        # Loop para cada nível (incluindo o térreo)
        for i in range(num_niveis):
            z = i * altura_piso
            # Loop para cada posição em Y e X para manter a ordem original
            for y in coords_y:
                for x in coords_x:
                    lista_coords.append([x, y, z])

        # Converte a lista para um array NumPy
        coord = np.array(lista_coords)

        # Geração da Matriz de Conectividade
        lista_conec = []

        # Geração dos Pilares
        # Para cada pavimento, conectar os nós ao pavimento superior
        for i in range(num_pavimentos):
            no_base_pavimento = i * nos_por_pavimento
            for j in range(nos_por_pavimento):
                no_inicial = no_base_pavimento + j
                no_final = no_inicial + nos_por_pavimento
                lista_conec.append([no_inicial, no_final])

        # Geração das Vigas
        # Para cada pavimento acima do térreo
        for i in range(1, num_niveis):
            no_base_pavimento = i * nos_por_pavimento
            
            # Vigas na direção X
            for row in range(len(coords_y)):
                for col in range(len(coords_x) - 1):
                    no_inicial = no_base_pavimento + row * len(coords_x) + col
                    no_final = no_inicial + 1
                    lista_conec.append([no_inicial, no_final])
                    
            # Vigas na direção Y
            for col in range(len(coords_x)):
                for row in range(len(coords_y) - 1):
                    no_inicial = no_base_pavimento + row * len(coords_x) + col
                    no_final = no_inicial + len(coords_x)
                    lista_conec.append([no_inicial, no_final])

        # Converte a lista para um array NumPy
        conec = np.array(lista_conec)

        # Definir os índices dos apoios
        condicoes_contorno = {'ENGASTE': [0, 1, 2, 3, 4, 5, 6, 7, 8]}

        # Definir o modelo estrutural (viga ou treliça)
        modelo = "viga"

        # Definir parâmetros constitutivos e geométricos
        secoes = {
            range(len(conec)): {"geometry_type": "retangular", "E": 2.7e7, "nu": 0.2, "base": 0.2, "altura": 0.4},
        }

        # Criar os dados do elemento estrutural
        # estrutura = Structure(analise, modelo, coordinates, connectivity, secoes, n)
        # estrutura.define_supports(condicoes_contorno)

        # Dicionário para agrupar elements com o mesmo carregamento
        qz_gravidade = -20
        qy_vento = 10

        # A chave será uma tupla (qx, qy, qz) e o valor uma lista de índices de elements
        cargas_agrupadas = {}

        # Itera sobre todos os elements para identificar e atribuir cargas
        for indice_elemento, (no1_idx, no2_idx) in enumerate(conec):
            
            # Pega as coordenadas dos nós do elemento
            coord_no1 = coord[no1_idx]
            coord_no2 = coord[no2_idx]
            
            # Inicializa o vetor de carga para este elemento
            qx, qy, qz = 0.0, 0.0, 0.0
            
            # Carga de gravidade (qz) em todas as vigas
            if coord_no1[2] == coord_no2[2]:
                qz = qz_gravidade
                
            # Carga de vento (qy) em pilares e vigas no plano X-Z
            if coord_no1[1] == 0 and coord_no2[1] == 0:
                if coord_no1[1] == coord_no2[1]:
                    qy = qy_vento
                
            # Agrupa o elemento se ele tiver alguma carga aplicada
            carga_tupla = (qx, qy, qz)
            if carga_tupla != (0.0, 0.0, 0.0):
                # Se a carga não existe no dicionário, cria uma nova lista
                if carga_tupla not in cargas_agrupadas:
                    cargas_agrupadas[carga_tupla] = []
                # Adiciona o índice do elemento à lista correspondente
                cargas_agrupadas[carga_tupla].append(indice_elemento)

        # Formata o dicionário de cargas para o formato final desejado
        dload_final = {}
        for carga_tupla, indices in cargas_agrupadas.items():
            # A chave é uma tupla (qx, qy, qz)
            chave = tuple(indices)
            
            # O valor é a carga distribuída
            valor = [[carga_tupla[0], carga_tupla[1], carga_tupla[2]], 
                    [carga_tupla[0], carga_tupla[1], carga_tupla[2]]]
                    
            dload_final[chave] = valor
        
        print(dload_final)
        quit()

        # Atribuir as cargas
        estrutura.add_distributed_loads(dload_final)

    if caso == 4:
        # Matriz de coordenadas dos pontos (x, y, z)
        coord = np.array([[0, 0, 0],
                          [0, 0, 0.5],
                          [0, 0, 1.0],
                          [1, 0, 0],
                          [1, 0, 0.5],
                          [1, 0, 2]])

        # Matriz de conectividade
        conec = np.array([[0, 1],
                          [1, 2],
                          [3, 4],
                          [4, 5],
                          [1, 4],
                          [2, 5]])

        # Definir os índices dos apoios
        condicoes_contorno = {
            # 'YSYMM': range(len(coordinates)),
            'FIXOXZ': [0, 3],
            # 'ROLETE_X': [3]
        }

        # Definir o modelo estrutural (viga ou treliça)
        modelo = "viga"

        # Definir member_releases
        releases = {
            'FREE-HINGE': [0, 1, 2],
        }

        # Definir parâmetros constitutivos e geométricos
        secoes = {
            # range(len(connectivity)): {"geometry_type": "retangular", "E": 7.2e6, "nu": 0.1, "base": 0.1, "altura": 0.1},
            range(len(conec)): {"geometry_type": "retangular", "E": 1e6, "nu": 0.0, "base": 0.1, "altura": 0.1}
        }

        # Criar os dados do elemento estrutural
        estrutura = Structure(analise, modelo, coord, conec, secoes, n)
        estrutura.define_supports(condicoes_contorno)

        # Adicionar cargas
        # estrutura.NLOAD({
        #     3: [0, 0, -30, 0, 0, 0]
        # })
        estrutura.add_distributed_loads({
            5: [[0, 0, -1], [0, 0, -1]]
        })

    if caso == 5:
        # Matriz de coordenadas dos pontos (x, y, z)
        coord = np.array([
                # Base do pórtico (nível do chão)
                [0, 0, 0],
                [5, 0, 0],
                [10, 0, 0],
                [15, 0, 0],

                [0, 4, 0],
                [5, 4, 0],
                [10, 4, 0],
                [15, 4, 0],

                [0, 8, 0],
                [5, 8, 0],
                [10, 8, 0],
                [15, 8, 0],

                # Colunas verticais (elevação)
                [0, 0, 6],
                [5, 0, 7],
                [7.5, 0, 7.5],
                [10, 0, 7],
                [15, 0, 6],

                [0, 4, 6],
                [5, 4, 7],
                [7.5, 4, 7.5],
                [10, 4, 7],
                [15, 4, 6],

                [0, 8, 6],
                [5, 8, 7],
                [7.5, 8, 7.5],
                [10, 8, 7],
                [15, 8, 6]])

        # Matriz de conectividade
        conec = np.array([
                # Colunas verticais
                [0, 12],
                [1, 13],
                [2, 15],
                [3, 16],
                [4, 17],
                [5, 18],
                [6, 20],
                [7, 21],
                [8, 22],
                [9, 23],
                [10, 25],
                [11, 26],

                # Traves horizontais superiores
                [12, 17],
                [17, 22],
                [13, 18],
                [18, 23],
                [14, 19],
                [19, 24],
                [15, 20],
                [20, 25],
                [16, 21],
                [21, 26],

                # Traves diagonais superiores
                [12, 13],
                [13, 14],
                [14, 15],
                [15, 16],
                [17, 18],
                [18, 19],
                [19, 20],
                [20, 21],
                [22, 23],
                [23, 24],
                [24, 25],
                [25, 26]])

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
                              'ENGASTE': range(12)
                              }

        # Definir o modelo estrutural (viga ou treliça)
        modelo = "viga"

        # Definir parâmetros constitutivos e geométricos
        secoes = {
            range(len(conec)): {"geometry_type": "retangular", "E": 2.7e7, "nu": 0.2, "base": 0.2, "altura": 0.2},
        }

        # Criar os dados do elemento estrutural
        estrutura = Structure(analise, modelo, coord, conec, secoes, n)
        estrutura.define_supports(condicoes_contorno)

        # Adicionar cargas
        estrutura.add_distributed_loads({
            range(12, 34): [[0, 0, -25], [0, 0, -25]]
        })

    return estrutura

exemplo('linear', 3, 1)
