def partial_row_data_to_column_dict(
    partial_row_data: bigtable.row_data.PartialRowData,
) -> Dict[column_keys._Column, bigtable.row_data.PartialRowData]:
    new_column_dict = {}
    for family_id, column_dict in partial_row_data._cells.items():
        for column_key, column_values in column_dict.items():
            column = column_keys.from_key(family_id, column_key)
            new_column_dict[column] = column_values
    return new_column_dict