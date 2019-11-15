from abc import ABC, abstractmethod

class Client(ABC):

    def __init__(self, config):
        self._config = config
        
    
    def _check_and_create_table(self) -> None:
        """ Checks if table exists and creates new one if necessary """
        table_ids = [t.table_id for t in self.instance.list_tables()]

        if not self.table_id in table_ids:
            self.table.create()
            f = self.table.column_family(self.family_id)
            f.create()

            f_inc = self.table.column_family(
                self.incrementer_family_id, gc_rule=MaxVersionsGCRule(1)
            )
            f_inc.create()

            f_log = self.table.column_family(self.log_family_id)
            f_log.create()

            f_ce = self.table.column_family(
                self.cross_edge_family_id, gc_rule=MaxVersionsGCRule(1)
            )
            f_ce.create()

            self.logger.info(f"Table {self.table_id} created")
