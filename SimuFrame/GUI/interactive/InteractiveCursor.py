# Third-party libraries
import numpy as np
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


class InteractiveCursor:
    """
    Interactive cursor that tracks the plot curve showing axis values.

    Features:
    - Interpolates values between curve points.
    - Shows circular point on interpolated curve.
    - Displays X and Y axis values with dashed lines.
    - Colored background for axis labels.
    """

    def __init__(self, ax, lines_data, x_label='X', y_label='Y', precision=4):
        """
        Args:
            ax: Matplotlib Axes object.
            lines_data: List of dictionaries with 'x', 'y', 'label', 'color'.
            x_label: X-axis label.
            y_label: Y-axis label.
            precision: Number of decimal places for display
        """
        self.ax = ax
        self.lines_data = lines_data
        self.x_label = x_label
        self.y_label = y_label
        self.precision = precision

        # Cursor visual elements
        self.cursor_point = None
        self.h_line = None
        self.v_line = None
        self.x_annotation = None
        self.y_annotation = None
        self.info_text = None

        # State
        self.active = False
        self.current_line_idx = 0

        # Connect events
        self.cid_motion = ax.figure.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.cid_leave = ax.figure.canvas.mpl_connect('axes_leave_event', self.on_mouse_leave)

        self._create_cursor_elements()

    def _create_cursor_elements(self):
        """Creates the visual elements of the cursor (initially invisible)."""

        # Circular point on the curve
        self.cursor_point = Circle((0, 0), radius=0,
                                   color='red', zorder=10,
                                   visible=False, alpha=0.8)
        self.ax.add_patch(self.cursor_point)

        # Dotted lines to axes
        self.h_line = Line2D([], [], color='gray', linestyle='--',
                           linewidth=1.5, alpha=0.7, visible=False, zorder=9)
        self.v_line = Line2D([], [], color='gray', linestyle='--',
                           linewidth=1.5, alpha=0.7, visible=False, zorder=9)
        self.ax.add_line(self.h_line)
        self.ax.add_line(self.v_line)

        # Axis annotations
        box_style = dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor='black', linewidth=2, alpha=0.95)

        self.x_annotation = Text(0, 0, '', ha='center', va='top',
                                 bbox=box_style,
                                 fontsize=10,
                                 fontweight='bold',
                                 visible=False, zorder=11)

        self.y_annotation = Text(0, 0, '', ha='right', va='center',
                                bbox=box_style,
                                fontsize=10,
                                fontweight='bold',
                                visible=False, zorder=11)

        self.ax.add_artist(self.x_annotation)
        self.ax.add_artist(self.y_annotation)

        # Info text (curve name)
        info_box_style = dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                              edgecolor='black', linewidth=2, alpha=0.9)

        self.info_text = Text(0, 0, '', ha='left', va='bottom',
                            bbox=info_box_style,
                            fontsize=10,
                            fontweight='bold',
                            visible=False, zorder=12)
        self.ax.add_artist(self.info_text)

    def _find_nearest_point_interpolated(self, x_mouse, y_mouse):
        """
        Finds the point nearest to the mouse by interpolating between curve points.

        Returns:
            tuple: (line_idx, x_interp, y_interp, distance) or None
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

            # Vectorized projection
            p1 = np.column_stack((x_data[:-1], y_data[:-1]))
            p2 = np.column_stack((x_data[1:], y_data[1:]))
            mouse_pt = np.array([x_mouse, y_mouse])

            # Segment vectors
            dp = p2 - p1

            # Vectors from p1 to mouse
            d_mouse = mouse_pt - p1

            # Project mouse onto segment lines
            lens_sq = np.sum(dp**2, axis=1)
            # Avoid division by zero
            lens_sq[lens_sq < 1e-10] = 1.0

            t = np.sum(d_mouse * dp, axis=1) / lens_sq
            t = np.clip(t, 0.0, 1.0)

            # Projected points
            projections = p1 + t[:, np.newaxis] * dp

            # Normalized distances
            d_norm = ((projections - mouse_pt) / [x_range, y_range])
            distances = np.sqrt(np.sum(d_norm**2, axis=1))

            # Find closest segment
            min_idx = np.argmin(distances)
            dist = distances[min_idx]

            if dist < min_distance:
                min_distance = dist
                best_result = (line_idx, projections[min_idx, 0], projections[min_idx, 1], dist)

        # Only show cursor if sufficiently close (tolerance 8%)
        if min_distance > 0.08:
            return None

        return best_result

    def _update_cursor(self, x, y, line_idx):
        """Updates the cursor position and appearance."""

        line_data = self.lines_data[line_idx]
        color = line_data.get('color', 'red')
        label = line_data.get('label', f'Curve {line_idx + 1}')

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # Update circular point
        radius = min((xlim[1] - xlim[0]), (ylim[1] - ylim[0])) * 0.012
        self.cursor_point.set_center((x, y))
        self.cursor_point.set_radius(radius)
        self.cursor_point.set_color(color)
        self.cursor_point.set_edgecolor('white')
        self.cursor_point.set_linewidth(2)
        self.cursor_point.set_visible(True)

        # Update dotted lines
        self.h_line.set_data([xlim[0], x], [y, y])
        self.v_line.set_data([x, x], [ylim[0], y])
        self.h_line.set_color(color)
        self.v_line.set_color(color)
        self.h_line.set_visible(True)
        self.v_line.set_visible(True)

        # Format values
        format_str = f'{{:.{self.precision}g}}'
        x_text = format_str.format(x)
        y_text = format_str.format(y)

        # Calcular posições das anotações
        x_margin = (xlim[1] - xlim[0]) * 0.08
        y_margin = (ylim[1] - ylim[0]) * 0.05

        # X-axis annotation
        y_pos_x = ylim[0] + y_margin
        self.x_annotation.set_position((x, y_pos_x))
        self.x_annotation.set_text(x_text)
        self.x_annotation.get_bbox_patch().set_facecolor(color)
        # self.x_annotation.get_bbox_patch().set_alpha(0.9)
        # self.x_annotation.set_color('white')
        self.x_annotation.set_visible(True)

        # Y-axis annotation
        x_pos_y = xlim[0] + x_margin
        self.y_annotation.set_position((x_pos_y, y))
        self.y_annotation.set_text(y_text)
        self.y_annotation.get_bbox_patch().set_facecolor(color)
        # self.y_annotation.get_bbox_patch().set_alpha(0.9)
        # self.y_annotation.set_color('white')
        self.y_annotation.set_visible(True)

        # Update info text
        offset_x = (xlim[1] - xlim[0]) * 0.02
        offset_y = (ylim[1] - ylim[0]) * 0.02
        tooltip_x = x + offset_x
        tooltip_y = y + offset_y

        # Smart positioning near edges
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

        # Remove "SimuFrame - " from label if it exists
        display_label = label.replace('SimuFrame - ', '')

        self.info_text.set_position((tooltip_x, tooltip_y))
        self.info_text.set_text(display_label)
        self.info_text.get_bbox_patch().set_edgecolor(color)
        self.info_text.set_visible(True)

    def _hide_cursor(self):
        """Hides all cursor elements."""
        self.cursor_point.set_visible(False)
        self.h_line.set_visible(False)
        self.v_line.set_visible(False)
        self.x_annotation.set_visible(False)
        self.y_annotation.set_visible(False)
        self.info_text.set_visible(False)
        self.active = False

    def on_mouse_move(self, event):
        """Mouse movement callback."""
        if event.inaxes != self.ax:
            if self.active:
                self._hide_cursor()
                self.ax.figure.canvas.draw_idle()
            return

        result = self._find_nearest_point_interpolated(event.xdata, event.ydata)

        if result is None:
            if self.active:
                self._hide_cursor()
                self.ax.figure.canvas.draw_idle()
            return

        line_idx, x_interp, y_interp, distance = result

        self._update_cursor(x_interp, y_interp, line_idx)
        self.active = True
        self.current_line_idx = line_idx
        self.ax.figure.canvas.draw_idle()

    def on_mouse_leave(self, event):
        """Callback when mouse leaves axes."""
        if self.active:
            self._hide_cursor()
            self.ax.figure.canvas.draw_idle()

    def disconnect(self):
        """Disconnects events and removes visual elements."""
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
    Adds an interactive cursor to an existing Matplotlib Axes.

    Args:
        ax: Matplotlib Axes object.
        precision: Number of decimal places.

    Returns:
        InteractiveCursor: Cursor instance (keep reference to avoid garbage collection).
    """
    lines_data = []

    for line in ax.get_lines():
        # Skip auxiliary lines (grid, faint lines)
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

    return InteractiveCursor(
            ax,
            lines_data,
            x_label=ax.get_xlabel(),
            y_label=ax.get_ylabel(),
            precision=precision
        )
