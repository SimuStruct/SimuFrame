import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.text import Text


class InteractiveCursor:
    """
    Cursor interativo que segue a curva do gráfico mostrando valores nodes eixos.
    
    Features:
    - Interpola valores entre pontos da curva
    - Mostra ponto circular na curva interpolada
    - Exibe valores nodes eixos X e Y com linhas pontilhadas
    - Rótulos nodes eixos com fundo colorido
    """
    
    def __init__(self, ax, lines_data, x_label='X', y_label='Y', precision=4):
        """
        Args:
            ax: Axes do matplotlib
            lines_data: Lista de dicionários com 'x', 'y', 'label', 'color'
            x_label: Rótulo do eixo X
            y_label: Rótulo do eixo Y
            precision: Número de casas decimais para exibição
        """
        self.ax = ax
        self.lines_data = lines_data
        self.x_label = x_label
        self.y_label = y_label
        self.precision = precision
        
        # Elementos visuais do cursor
        self.cursor_point = None
        self.h_line = None
        self.v_line = None
        self.x_annotation = None
        self.y_annotation = None
        self.info_text = None
        
        # Estado
        self.active = False
        self.current_line_idx = 0
        
        # Conectar eventos
        self.cid_motion = ax.figure.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.cid_leave = ax.figure.canvas.mpl_connect('axes_leave_event', self.on_mouse_leave)
        
        self._create_cursor_elements()
    
    def _create_cursor_elements(self):
        """Cria os elements visuais do cursor (inicialmente invisíveis)."""
        
        # Ponto circular na curva
        self.cursor_point = Circle((0, 0), radius=0, 
                                   color='red', zorder=10, 
                                   visible=False, alpha=0.8)
        self.ax.add_patch(self.cursor_point)
        
        # Linhas pontilhadas até os eixos
        self.h_line = Line2D([], [], color='gray', linestyle='--', 
                           linewidth=1.5, alpha=0.7, visible=False, zorder=9)
        self.v_line = Line2D([], [], color='gray', linestyle='--', 
                           linewidth=1.5, alpha=0.7, visible=False, zorder=9)
        self.ax.add_line(self.h_line)
        self.ax.add_line(self.v_line)
        
        # Anotações nodes eixos
        self.x_annotation = Text(0, 0, '', ha='center', va='top',
                                bbox=dict(boxstyle='round,pad=0.4', 
                                        facecolor='white', 
                                        edgecolor='black', 
                                        linewidth=2,
                                        alpha=0.95),
                                fontsize=10, 
                                fontweight='bold',
                                visible=False, zorder=11)
        
        self.y_annotation = Text(0, 0, '', ha='right', va='center',
                                bbox=dict(boxstyle='round,pad=0.4', 
                                        facecolor='white', 
                                        edgecolor='black',
                                        linewidth=2,
                                        alpha=0.95),
                                fontsize=10,
                                fontweight='bold',
                                visible=False, zorder=11)
        
        self.ax.add_artist(self.x_annotation)
        self.ax.add_artist(self.y_annotation)
        
        # Texto informativo (nome da curva)
        self.info_text = Text(0, 0, '', ha='left', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.5', 
                                    facecolor='lightyellow', 
                                    edgecolor='black', 
                                    linewidth=2,
                                    alpha=0.9),
                            fontsize=10, 
                            fontweight='bold',
                            visible=False, zorder=12)
        self.ax.add_artist(self.info_text)
    
    def _find_nearest_point_interpolated(self, x_mouse, y_mouse):
        """
        Encontra o ponto mais próximo do mouse INTERPOLANDO entre pontos da curva.
        
        Returns:
            tuple: (line_idx, x_interp, y_interp, distance) ou None
        """
        if not self.lines_data:
            return None
        
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        
        if x_range == 0 or y_range == 0:
            return None
        
        min_distance = float('inf')
        best_result = None
        
        for line_idx, line_data in enumerate(self.lines_data):
            x_data = np.array(line_data['x'])
            y_data = np.array(line_data['y'])
            
            if len(x_data) < 2:
                continue
            
            # Para cada segmento da curva, encontrar o ponto mais próximo
            for i in range(len(x_data) - 1):
                x1, x2 = x_data[i], x_data[i + 1]
                y1, y2 = y_data[i], y_data[i + 1]
                
                # Encontrar projeção do mouse no segmento de reta
                x_proj, y_proj, t = self._project_point_on_segment(
                    x_mouse, y_mouse, x1, y1, x2, y2
                )
                
                # Calcular distância normalizada
                dx_norm = (x_proj - x_mouse) / x_range
                dy_norm = (y_proj - y_mouse) / y_range
                distance = np.sqrt(dx_norm**2 + dy_norm**2)
                
                if distance < min_distance:
                    min_distance = distance
                    best_result = (line_idx, x_proj, y_proj, distance)
        
        # Apenas mostrar cursor se estiver suficientemente próximo
        if min_distance > 0.08:  # Aumentado para 8% para melhor usabilidade
            return None
        
        return best_result
    
    def _project_point_on_segment(self, px, py, x1, y1, x2, y2):
        """
        Projeta um ponto (px, py) no segmento de reta entre (x1,y1) e (x2,y2).
        
        Returns:
            tuple: (x_proj, y_proj, t) onde t é o parâmetro [0,1] ao longo do segmento
        """
        # Vetor do segmento
        dx = x2 - x1
        dy = y2 - y1
        
        # Vetor do ponto 1 ao ponto mouse
        dpx = px - x1
        dpy = py - y1
        
        # Comprimento ao quadrado do segmento
        length_sq = dx*dx + dy*dy
        
        if length_sq < 1e-10:  # Segmento degenerado
            return x1, y1, 0.0
        
        # Parâmetro t da projeção (0 = início, 1 = fim)
        t = (dpx * dx + dpy * dy) / length_sq
        
        # Limitar t ao intervalo [0, 1] para ficar dentro do segmento
        t = max(0.0, min(1.0, t))
        
        # Calcular ponto projetado
        x_proj = x1 + t * dx
        y_proj = y1 + t * dy
        
        return x_proj, y_proj, t
    
    def _update_cursor(self, x, y, line_idx):
        """Atualiza a posição e aparência do cursor."""
        
        line_data = self.lines_data[line_idx]
        color = line_data.get('color', 'red')
        label = line_data.get('label', f'Curva {line_idx + 1}')
        
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # Atualizar ponto circular
        radius = min((xlim[1] - xlim[0]), (ylim[1] - ylim[0])) * 0.012
        self.cursor_point.set_center((x, y))
        self.cursor_point.set_radius(radius)
        self.cursor_point.set_color(color)
        self.cursor_point.set_edgecolor('white')
        self.cursor_point.set_linewidth(2)
        self.cursor_point.set_visible(True)
        
        # Atualizar linhas pontilhadas
        self.h_line.set_data([xlim[0], x], [y, y])
        self.v_line.set_data([x, x], [ylim[0], y])
        self.h_line.set_color(color)
        self.v_line.set_color(color)
        self.h_line.set_visible(True)
        self.v_line.set_visible(True)
        
        # Formatação dos valores
        format_str = f'{{:.{self.precision}g}}'
        x_text = format_str.format(x)
        y_text = format_str.format(y)
        
        # Calcular posições das anotações
        x_margin = (xlim[1] - xlim[0]) * 0.08
        y_margin = (ylim[1] - ylim[0]) * 0.05
        
        # Anotação no eixo X (abaixo do gráfico)
        y_pos_x = ylim[0] + y_margin
        self.x_annotation.set_position((x, y_pos_x))
        self.x_annotation.set_text(x_text)
        self.x_annotation.get_bbox_patch().set_facecolor(color)
        self.x_annotation.get_bbox_patch().set_alpha(0.9)
        self.x_annotation.set_color('white')
        self.x_annotation.set_visible(True)
        
        # Anotação no eixo Y (à esquerda do gráfico)
        x_pos_y = xlim[0] + x_margin
        self.y_annotation.set_position((x_pos_y, y))
        self.y_annotation.set_text(y_text)
        self.y_annotation.get_bbox_patch().set_facecolor(color)
        self.y_annotation.get_bbox_patch().set_alpha(0.9)
        self.y_annotation.set_color('white')
        self.y_annotation.set_visible(True)
        
        # Atualizar texto informativo (nome da curva)
        offset_x = (xlim[1] - xlim[0]) * 0.02
        offset_y = (ylim[1] - ylim[0]) * 0.02
        
        # Posicionamento inteligente do tooltip
        tooltip_x = x + offset_x
        tooltip_y = y + offset_y
        
        # Ajustar se próximo das bordas
        if tooltip_x > xlim[1] - (xlim[1] - xlim[0]) * 0.25:
            tooltip_x = x - offset_x
            self.info_text.set_ha('right')
        else:
            self.info_text.set_ha('left')
        
        if tooltip_y > ylim[1] - (ylim[1] - ylim[0]) * 0.15:
            tooltip_y = y - offset_y
            self.info_text.set_va('top')
        else:
            self.info_text.set_va('bottom')
        
        # Remover "SimuFrame - " do label se existir
        display_label = label.replace('SimuFrame - ', '')
        
        self.info_text.set_position((tooltip_x, tooltip_y))
        self.info_text.set_text(display_label)
        self.info_text.get_bbox_patch().set_edgecolor(color)
        self.info_text.get_bbox_patch().set_linewidth(2)
        self.info_text.set_visible(True)
    
    def _hide_cursor(self):
        """Esconde todos os elements do cursor."""
        self.cursor_point.set_visible(False)
        self.h_line.set_visible(False)
        self.v_line.set_visible(False)
        self.x_annotation.set_visible(False)
        self.y_annotation.set_visible(False)
        self.info_text.set_visible(False)
        self.active = False
    
    def on_mouse_move(self, event):
        """Callback para movimento do mouse."""
        if event.inaxes != self.ax:
            if self.active:
                self._hide_cursor()
                self.ax.figure.canvas.draw_idle()
            return
        
        # Encontrar ponto mais próximo (interpolado)
        result = self._find_nearest_point_interpolated(event.xdata, event.ydata)
        
        if result is None:
            if self.active:
                self._hide_cursor()
                self.ax.figure.canvas.draw_idle()
            return
        
        line_idx, x_interp, y_interp, distance = result
        
        # Atualizar cursor
        self._update_cursor(x_interp, y_interp, line_idx)
        self.active = True
        self.current_line_idx = line_idx
        
        # Redesenhar
        self.ax.figure.canvas.draw_idle()
    
    def on_mouse_leave(self, event):
        """Callback quando o mouse sai do eixo."""
        if self.active:
            self._hide_cursor()
            self.ax.figure.canvas.draw_idle()
    
    def disconnect(self):
        """Desconecta os eventos e remove elements visuais."""
        self.ax.figure.canvas.mpl_disconnect(self.cid_motion)
        self.ax.figure.canvas.mpl_disconnect(self.cid_leave)
        
        self.cursor_point.remove()
        self.h_line.remove()
        self.v_line.remove()
        self.x_annotation.remove()
        self.y_annotation.remove()
        self.info_text.remove()


def add_interactive_cursor_to_plot(ax, precision=4):
    """
    Adiciona cursor interativo a um axes do matplotlib existente.
    
    Args:
        ax: Axes do matplotlib
        precision: Número de casas decimais
    
    Returns:
        InteractiveCursor: Instância do cursor (guardar para desconectar depois)
    """
    # Extrair dados das linhas existentes no plot
    lines_data = []
    
    for line in ax.get_lines():
        # Pular linhas auxiliares (grid, pontilhadas fracas)
        if line.get_linestyle() in ['--', ':', '-.'] and line.get_alpha() and line.get_alpha() < 0.8:
            continue
        
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        
        if len(x_data) == 0:
            continue
        
        lines_data.append({
            'x': x_data,
            'y': y_data,
            'label': line.get_label() if line.get_label() and not line.get_label().startswith('_') else 'Curva',
            'color': line.get_color()
        })
    
    if not lines_data:
        return None
    
    # Criar e retornar cursor
    cursor = InteractiveCursor(
        ax, 
        lines_data,
        x_label=ax.get_xlabel(),
        y_label=ax.get_ylabel(),
        precision=precision
    )
    
    return cursor

# ============================================================================
# EXEMPLO DE USO STANDALONE
# ============================================================================

if __name__ == "__main__":
    # Criar dados de exemplo
    x = np.linspace(0, 10, 50)
    y1 = np.sin(x)
    y2 = np.cos(x)

    fig, ax = plt.subplots()
    ax.plot(x, y1, 'b-', label='sin(x)')
    ax.plot(x, y2, 'r-', label='cos(x)')
    ax.legend()
    
    # Criar figura e plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(x, y1, 'b-o', markersize=3, label='Curva 1: sin(x)', linewidth=2)
    ax.plot(x, y2, 'r-s', markersize=3, label='Curva 2: cos(x)', linewidth=2)
    # ax.plot(x, y3, 'g-^', markersize=3, label='Curva 3: 0.5*sin(2x)', linewidth=2)
    
    ax.set_xlabel('Eixo X (unidade)')
    ax.set_ylabel('Eixo Y (unidade)')
    ax.set_title('Exemplo de Cursor Interativo')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Adicionar cursor interativo
    cursor = add_interactive_cursor_to_plot(ax, precision=3)
    
    print("Passe o mouse sobre as curvas para ver o cursor interativo!")
    print("Pressione Ctrl+C para sair.")
    
    plt.show()