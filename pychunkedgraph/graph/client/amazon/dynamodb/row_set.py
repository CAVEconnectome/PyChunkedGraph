class RowSet:
    def __init__(self, row_keys=None, row_ranges=None):
        if row_ranges is None:
            row_ranges = []
        if row_keys is None:
            row_keys = []
        self._row_keys = row_keys
        self._row_ranges = row_ranges
    
    @property
    def row_keys(self):
        return self._row_keys
    
    @row_keys.setter
    def row_keys(self, value):
        self._row_keys = value
    
    @property
    def row_ranges(self):
        return self._row_ranges
    
    @row_ranges.setter
    def row_ranges(self, value):
        self._row_ranges = value
