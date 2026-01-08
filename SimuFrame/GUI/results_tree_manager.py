# Built-in libraries
from typing import List, Dict, Any

# Third-party libraries
import shiboken6
from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtWidgets import QTreeWidgetItemIterator, QTreeWidgetItem


class ResultsTreeManager(QObject):
    """Manages selection logic for the results tree widget."""

    selection_changed = Signal(list)

    def __init__(self, tree_widget):
        super().__init__()
        self.tree = tree_widget
        self._processing = False

    def handle_item_changed(self, item: QTreeWidgetItem, column: int):
        """Handle changes to tree items with safety checks for C++ object validity."""
        # Check if Python object still points to valid C++ object
        try:
            if not shiboken6.isValid(item):
                return

            # Check if item still belongs to tree (deleted items return None)
            if item.treeWidget() is None:
                return

            # If C++ deleted parent before child during cascade clear, parent() may fail
            parent = item.parent()

        except RuntimeError:
            return

        # Ignore parent items or items without data_key
        if not item.parent() or item.data(0, Qt.ItemDataRole.UserRole) is None:
            return

        # Prevent recursion
        if self._processing:
            return

        self._processing = True

        try:
            if item.checkState(column) == Qt.CheckState.Checked:
                self.tree.blockSignals(True)
                try:
                    self._handle_item_checked(item, column)
                finally:
                    self.tree.blockSignals(False)

            # Emit signal with current selections
            selected_keys = self.get_selected_keys()
            self.selection_changed.emit(selected_keys)

        finally:
            self._processing = False

    def _handle_item_checked(self, checked_item: QTreeWidgetItem, column: int):
        """Handle logic when an item is checked."""
        is_reaction_parent = self._is_reaction_item(checked_item)

        # Iterate over all tree items
        iterator = QTreeWidgetItemIterator(self.tree)
        while iterator.value():
            current_item = iterator.value()

            # Uncheck all items except the checked one
            if current_item is not checked_item:
                is_current_reaction = self._is_reaction_item(current_item)

                # If not both reaction items, uncheck
                if not (is_current_reaction and is_reaction_parent):
                    if current_item.parent() and current_item.data(0, Qt.ItemDataRole.UserRole):
                        current_item.setCheckState(column, Qt.CheckState.Unchecked)

            iterator += 1

    @staticmethod
    def _is_reaction_item(item: QTreeWidgetItem) -> bool:
        """Check if an item belongs to the Support Reactions group."""
        parent = item.parent()
        return parent is not None and parent.text(0) == "Support Reactions"

    def get_selected_keys(self) -> List[Dict[str, Any]]:
        """Return list of data_keys from selected items."""
        selected = []
        iterator = QTreeWidgetItemIterator(
            self.tree,
            QTreeWidgetItemIterator.IteratorFlag.Checked
        )

        while iterator.value():
            item = iterator.value()
            parent = item.parent()

            if parent:
                data_key = item.data(0, Qt.ItemDataRole.UserRole)
                if data_key:
                    selected.append({
                        'key': data_key,
                        'is_reaction': self._is_reaction_item(item)
                    })
            iterator += 1

        return selected

    def clear_selection(self):
        """Uncheck all items in the tree."""
        self._processing = True
        try:
            iterator = QTreeWidgetItemIterator(self.tree)
            while iterator.value():
                item = iterator.value()
                if item.parent() and item.data(0, Qt.ItemDataRole.UserRole):
                    item.setCheckState(0, Qt.CheckState.Unchecked)
                iterator += 1
        finally:
            self._processing = False
