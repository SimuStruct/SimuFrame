# Built-in libraries
from typing import Dict
from abc import ABC, abstractmethod
from contextlib import contextmanager

# Third-party libraries
import pyvista as pv

# Local libraries
from SimuFrame.post_processing.visualization import plot_structure, plotar_reacoes
from SimuFrame.post_processing.forces_visualization import plotar_esforcos
from SimuFrame.post_processing.displacement_visualization import plotar_deslocamentos


class PlotCommand(ABC):
    """Comando abstrato para plotagem."""

    @abstractmethod
    def execute(self, plotter, state, **kwargs):
        """Executa o comando de plotagem."""
        pass


class PlotDisplacementCommand(PlotCommand):
    """Comando para plotar deslocamentos."""

    def __init__(self, componente: str):
        self.componente = componente

    def execute(self, plotter, state, **kwargs):
        """Plota deslocamentos."""
        if state.resultados.deslocamentos is None:
            return

        viz_options = state.get_visualization_options()
        viz_options.update(kwargs)
        viz_options['componente'] = self.componente

        # Delegar para a função especializada
        plotar_deslocamentos(
            state.estrutura,
            state.malhas.deformada,
            state.malhas.coords_deformadas,
            state.resultados.deslocamentos,
            plotter,
            **viz_options
        )


class PlotForceCommand(PlotCommand):
    """Comando para plotar esforços internos."""

    def __init__(self, force_type: str):
        self.force_type = force_type

    def execute(self, plotter, state, **kwargs):
        """Plota esforços internos."""
        if state.resultados.esforcos_int is None:
            return

        viz_options = state.get_visualization_options()
        viz_options.update(kwargs)
        viz_options['force'] = self.force_type

        # Delegar para a função especializada
        plotar_esforcos(
            state.estrutura,
            state.malhas.deformada,
            state.malhas.coords,
            state.resultados.esforcos_int,
            plotter,
            **viz_options
        )


class PlotReactionCommand(PlotCommand):
    """Comando para plotar reações de apoio."""

    def execute(self, plotter, state, reacoes_a_plotar=None, **kwargs):
        """Plota reações de apoio."""
        if state.resultados.reacoes_nos_apoios is None:
            return

        # Delegar para a função especializada
        plotar_reacoes(
            plotter,
            state.resultados.reacoes_nos_apoios,
            reacoes_a_plotar=reacoes_a_plotar or []
        )


class PlotManager:
    """Gerenciador central de plotagem."""
    def __init__(self, plotter, state_manager):
        self.plotter = plotter
        self.state = state_manager
        self.commands = self._create_commands()
        self.dark_mode = False

    @contextmanager
    def render_lock(self):
        """Bloqueia a renderização do VTK durante operações em lote."""
        # Acesso de baixo nível à janela de renderização do VTK
        render_window = self.plotter.ren_win

        # Desabilita atualizações (SwapBuffers off impede que o buffer incompleto vá para a tela)
        render_window.SetSwapBuffers(0)

        try:
            yield
        finally:
            # Reabilita e força um único render final
            render_window.SetSwapBuffers(1)
            self.plotter.render()

    @staticmethod
    def _create_commands() -> Dict[str, PlotCommand]:
        """Cria o dicionário de comandos de plotagem."""
        return {
            # Deslocamentos globais
            'u': PlotDisplacementCommand('u'),
            'ux': PlotDisplacementCommand('x'),
            'uy': PlotDisplacementCommand('y'),
            'uz': PlotDisplacementCommand('z'),
            'θx': PlotDisplacementCommand('θx'),
            'θy': PlotDisplacementCommand('θy'),
            'θz': PlotDisplacementCommand('θz'),

            # Esforços internos
            'fx': PlotForceCommand('Fx'),
            'fy': PlotForceCommand('Fy'),
            'fz': PlotForceCommand('Fz'),
            'mx': PlotForceCommand('Mx'),
            'my': PlotForceCommand('My'),
            'mz': PlotForceCommand('Mz'),

            # Reações de apoio
            'reacoes_apoio': PlotReactionCommand()
        }

    def clear_result_actors(self):
        """Remove apenas atores de resultados."""
        actors_to_remove = [
            # Diagramas e rótulos de esforços
            'diagrama_esforcos',
            'rotulos_esforcos',
            'legendas_esforcos',

            # Reações
            'reacoes_forcas',
            'reacoes_momentos',

            # Marcadores de deslocamentos
            'esfera_max_deslocamento',
            'rotulo_max_deslocamento'
        ]

        for name in actors_to_remove:
            try:
                self.plotter.remove_actor(name, render=False)
            except KeyError:
                pass

    def plot_result(self, result_key: str, **options):
        """Plota um resultado específico."""
        if result_key not in self.commands:
            print(f"[PlotManager] Comando '{result_key}' não encontrado.")
            return

        command = self.commands[result_key]

        with self.render_lock():
            # Limpar resultados anteriores
            self.clear_result_actors()

            # Executar novo plot
            command.execute(self.plotter, self.state, **options)

    def plot_base_structure(self):
        """Plota apenas a estrutura base (indeformada)."""
        if not self.state.has_structure():
            return

        # Limpar tudo
        self.plotter.clear_actors()

        # Adicionar eixos e orientação
        self.setup_initial_view()

        # Plotar apenas estrutura indeformada
        if self.state.malhas.indeformada:
            plot_structure(
                self.plotter,
                self.state.estrutura,
                self.state.malhas.indeformada,
                transparencia=0.2,
                plotar_secao=self.state.viz_options.plotar_secao,
                plotar_cargas=False,
                plotar_nos=False,
                name='estrutura_indeformada'
            )

    def update_section_visibility(self, visible: bool):
        """Atualiza a visibilidade das seções da estrutura indeformada."""
        try:
            actor = self.plotter.actors.get('estrutura_indeformada_secoes')
            if actor:
                actor.visibility = visible
                self.plotter.render()
        except (KeyError, AttributeError):
            pass

    def toggle_background(self):
        """Alterna entre fundo claro e escuro."""
        if self.dark_mode:
            self.plotter.set_background('darkgray', top='white')
            self.dark_mode = False
        else:
            self.plotter.set_background('midnightblue', top='black')
            self.dark_mode = True

    def take_screenshot(self, filepath: str):
        """Salva uma captura de tela."""
        self.plotter.screenshot(filepath)

    def setup_initial_view(self):
        """Configura a visualização inicial."""
        self.plotter.add_axes()
        self.plotter.add_orientation_widget(pv.AxesActor())

    def reset_camera(self):
        """Reseta a câmera para visualização padrão."""
        self.plotter.reset_camera()