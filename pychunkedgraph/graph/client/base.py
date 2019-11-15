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




    def check_and_write_table_parameters(
        self,
        column: column_keys._Column,
        value: Optional[Union[str, np.uint64]] = None,
        required: bool = True,
        is_new: bool = False,
    ) -> Union[str, np.uint64]:
        """ Checks if a parameter already exists in the table. If it already
        exists it returns the stored value, else it stores the given value.
        Storing the given values can be enforced with `is_new`. The function
        raises an exception if no value is passed and the parameter does not
        exist, yet.
        :param column: column_keys._Column
        :param value: Union[str, np.uint64]
        :param required: bool
        :param is_new: bool
        :return: Union[str, np.uint64]
            value
        """
        setting = self.read_byte_row(row_key=row_keys.GraphSettings, columns=column)
        if (not setting or is_new) and value is not None:
            row = self.mutate_row(row_keys.GraphSettings, {column: value})
            self.bulk_write([row])
        elif not setting and value is None:
            assert not required
            return None
        else:
            value = setting[0].value

        return value            
