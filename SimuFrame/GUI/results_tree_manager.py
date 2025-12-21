# Third-party libraries
import shiboken6
from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtWidgets import QTreeWidgetItemIterator


class ResultsTreeManager(QObject):
    """Gerencia a lógica de seleção da árvore de resultados."""
    
    # Sinal emitido quando a seleção muda
    selection_changed = Signal(list)
    
    def __init__(self, tree_widget):
        super().__init__()
        self.tree = tree_widget
        self._processing = False

    def handle_item_changed(self, item, column):
        """Gerencia mudanças nodes itens da árvore."""
        # Verifica se o objeto Python ainda aponta para um objeto C++ válido
        try:
            if not shiboken6.isValid(item):
                return

            # Verifica se o item ainda pertence à árvore (se foi deletado, treeWidget é None)
            if item.treeWidget() is None:
                return

            # Se o C++ deletou o pai antes do filho durante um clear cascata, parent() pode falhar
            parent = item.parent()

        except RuntimeError:
            return

        # Ignorar pais ou itens sem data_key
        if not item.parent() or item.data(0, Qt.ItemDataRole.UserRole) is None:
            return
        
        # Evitar recursão
        if self._processing:
            return

        # Sinalizar que a UI está sendo atualizada
        self._processing = True

        try:
            if item.checkState(column) == Qt.CheckState.Checked:
                self.tree.blockSignals(True)

                try:
                    self._handle_item_checked(item, column)
                finally:
                    self.tree.blockSignals(False)
            
            # Emitir sinal com seleções atuais
            selected_keys = self.get_selected_keys()
            self.selection_changed.emit(selected_keys)
            
        finally:
            self._processing = False
    
    def _handle_item_checked(self, checked_item, column):
        """Trata a lógica quando um item é marcado."""
        is_reaction_parent = self._is_reaction_item(checked_item)

        # Iterar sobre todos os itens da árvore
        iterator = QTreeWidgetItemIterator(self.tree)
        while iterator.value():
            current_item = iterator.value()

            # Se não for o item marcado, desmarcar os demais
            if current_item is not checked_item:
                is_current_reaction = self._is_reaction_item(current_item)

                # Se não são ambos itens de reação, desmarcar
                if not (is_current_reaction and is_reaction_parent):
                    if current_item.parent() and current_item.data(0, Qt.ItemDataRole.UserRole):
                        current_item.setCheckState(column, Qt.CheckState.Unchecked)

            iterator += 1

    @staticmethod
    def _is_reaction_item(item):
        """Verifica se um item pertence ao grupo de Reações de Apoio."""
        parent = item.parent()
        return parent and parent.text(0) == "Reações de Apoio"
    
    def get_selected_keys(self):
        """Retorna lista de data_keys dos itens selecionados."""
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
        """Desmarca todos os itens."""
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
