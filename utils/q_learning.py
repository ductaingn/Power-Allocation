from typing import Any, Tuple, Union
import pickle
import numpy as np
from collections import defaultdict
import numpy as np
import attrs

@attrs.define
class Table:
    default_value: float = 0.0
    table: dict = attrs.field(init=False, factory=lambda: defaultdict(lambda: defaultdict(float)))

    def __attrs_post_init__(self):
        self.table = defaultdict(lambda: defaultdict(lambda: self.default_value))

    def get(self, state: Any, action: Tuple[int, ...]) -> float:
        return self.table[state][action]

    def update(self, state: Any, action: Tuple[int, ...], value: float):
        self.table[state][action] = value

    def all_state_actions(self):
        for state, actions in self.table.items():
            for action, value in actions.items():
                yield (state, action, value)

    def save(self, filename:str):
        """
        Save the able to a file. This method is intended to be overridden by child classes.
        """
        raise NotImplementedError("This method should be overridden by child classes.")

    def load(self, filename:str):
        """
        Load the table from a file. This method is intended to be overridden by child classes.
        """
        raise NotImplementedError("This method should be overridden by child classes.")

    def __add__(self, other: "Table") -> "Table":
        if not isinstance(other, Table):
            return NotImplemented
        result = Table(default_value=self.default_value)
        keys = set()
        for s in self.table:
            for a in self.table[s]:
                keys.add((s, a))
        for s in other.table:
            for a in other.table[s]:
                keys.add((s, a))
        for state, action in keys:
            result.update(state, action, self.get(state, action) + other.get(state, action))
        return result

    def __sub__(self, other: "Table") -> "Table":
        if not isinstance(other, Table):
            return NotImplemented
        result = Table(default_value=self.default_value)
        keys = set()
        for s in self.table:
            for a in self.table[s]:
                keys.add((s, a))
        for s in other.table:
            for a in other.table[s]:
                keys.add((s, a))
        for state, action in keys:
            result.update(state, action, self.get(state, action) - other.get(state, action))
        return result

    def __mul__(self, scalar: Union[int, float]) -> "Table":
        result = Table(default_value=self.default_value * scalar)
        for state, action, value in self.all_state_actions():
            result.update(state, action, value * scalar)
        return result
    
    def __truediv__(self, scalar: Union[int, float]) -> "Table":
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide Q-table by zero.")
        result = Table(default_value=self.default_value)
        for state, action, value in self.all_state_actions():
            result.update(state, action, value / scalar)
        return result

    def __rmul__(self, scalar: Union[int, float]) -> "Table":
        return self.__mul__(scalar)

    def __iadd__(self, other: "Table") -> "Table":
        if not isinstance(other, Table):
            return NotImplemented
        for state, action, value in other.all_state_actions():
            new_value = self.get(state, action) + value
            self.update(state, action, new_value)
        return self
    
    def __pow__(self, exponent: float) -> "Table":
        if not isinstance(exponent, (int, float)):
            raise TypeError("Exponent must be an int or float.")
        
        result = Table(default_value=self.default_value)
        for state, action, value in self.all_state_actions():
            result.update(state, action, value ** exponent)
        return result

    def copy(self) -> "Table":
        new_q = Table(default_value=self.default_value)
        for state, action, value in self.all_state_actions():
            new_q.update(state, action, value)
        return new_q

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Table):
            return False
        return dict(self.table) == dict(other.table)

    def __repr__(self) -> str:
        entries = list(self.all_state_actions())
        preview = entries[:5]
        repr_str = "\n".join(f"{s} | {a} → {q:.2f}" for s, a, q in preview)
        if len(entries) > 5:
            repr_str += f"\n... and {len(entries)-5} more entries"
        return repr_str or "Table(empty)"


@attrs.define
class QTable(Table):
    best_action_cache: dict = attrs.field(init=False, factory=dict)

    def update(self, state, action, value):
        super().update(state, action, value)

        # Update best action cache
        current_best = self.best_action_cache.get(state)
        if current_best is None or value > self.get(state, current_best):
            self.best_action_cache[state] = action

    def best_action(self, state: Any):
        return self.best_action_cache.get(state, None)

    def max_q_value(self, state: Any) -> float:
        best = self.best_action(state)
        if best is None:
            return self.default_value
        return self.get(state, best)
    
    def __add__(self, other: "QTable") -> "QTable":
        if not isinstance(other, QTable):
            return NotImplemented
        result = QTable(default_value=self.default_value)
        keys = set()
        for s in self.table:
            for a in self.table[s]:
                keys.add((s, a))
        for s in other.table:
            for a in other.table[s]:
                keys.add((s, a))
        for state, action in keys:
            result.update(state, action, self.get(state, action) + other.get(state, action))
        return result

    def __sub__(self, other: "QTable") -> "QTable":
        if not isinstance(other, QTable):
            return NotImplemented
        result = QTable(default_value=self.default_value)
        keys = set()
        for s in self.table:
            for a in self.table[s]:
                keys.add((s, a))
        for s in other.table:
            for a in other.table[s]:
                keys.add((s, a))
        for state, action in keys:
            result.update(state, action, self.get(state, action) - other.get(state, action))
        return result

    def __mul__(self, scalar: Union[int, float]) -> "QTable":
        result = QTable(default_value=self.default_value * scalar)
        for state, action, value in self.all_state_actions():
            result.update(state, action, value * scalar)
        return result
    
    def __truediv__(self, scalar: Union[int, float]) -> "QTable":
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide Q-table by zero.")
        result = QTable(default_value=self.default_value)
        for state, action, value in self.all_state_actions():
            result.update(state, action, value / scalar)
        return result

    def __rmul__(self, scalar: Union[int, float]) -> "QTable":
        return self.__mul__(scalar)

    def __iadd__(self, other: "QTable") -> "QTable":
        if not isinstance(other, QTable):
            return NotImplemented
        for state, action, value in other.all_state_actions():
            new_value = self.get(state, action) + value
            self.update(state, action, new_value)
        return self
    
    def __pow__(self, exponent: float) -> "QTable":
        if not isinstance(exponent, (int, float)):
            raise TypeError("Exponent must be an int or float.")
        
        result = QTable(default_value=self.default_value)
        for state, action, value in self.all_state_actions():
            result.update(state, action, value ** exponent)
        return result
    
    def copy(self):
        new_q = QTable(default_value=self.default_value)
        for state, action, value in self.all_state_actions():
            new_q.update(state, action, value)
        new_q.best_action_cache = self.best_action_cache.copy()
        return new_q
    
    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump({
                'table': dict(self.table),
                'best_action_cache': self.best_action_cache,
                'default_value': self.default_value,
            }, f)
    
    def load(self, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        self.default_value = data['default_value']
        self.table = defaultdict(lambda: defaultdict(lambda: self.default_value), data['table'])
        self.best_action_cache = data['best_action_cache']


@attrs.define
class VTable(Table):
    def update(self, state, action):
        if not (state in self.table):
            self.table[state][action] = 1 
        else:
            self.table[state][action] += 1
    
    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump({
                'table': dict(self.table),
                'default_value': self.default_value,
            }, f)
    
    def load(self, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        self.default_value = data['default_value']
        self.table = defaultdict(lambda: defaultdict(lambda: self.default_value), data['table'])


@attrs.define
class AlphaTable(Table):
    def update(self, state, action, value):
        return super().update(state, action, value)
    
    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump({
                'table': dict(self.table),
                'default_value': self.default_value,
            }, f)
    
    def load(self, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        self.default_value = data['default_value']
        self.table = defaultdict(lambda: defaultdict(lambda: self.default_value), data['table'])