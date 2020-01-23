"""
Functions for tracking root ID changes over time.
"""


# def get_latest_root_id(self, root_id: np.uint64) -> np.ndarray:
#     """Returns the latest root id associated with the provided root id
#     :param root_id: uint64
#     :return: list of uint64s
#     """
#     id_working_set = [root_id]
#     column = attributes.Hierarchy.NewParent
#     latest_root_ids = []
#     while len(id_working_set) > 0:
#         next_id = id_working_set[0]
#         del id_working_set[0]
#         node = self.client.read_node(next_id, properties=column)
#         # Check if a new root id was attached to this root id
#         if node:
#             id_working_set.extend(node[0].value)
#         else:
#             latest_root_ids.append(next_id)

#     return np.unique(latest_root_ids)


# def get_future_root_ids(
#     self,
#     root_id: basetypes.NODE_ID,
#     time_stamp: typing.Optional[datetime.datetime] = misc_utils.get_max_time(),
# ) -> np.ndarray:
#     """ Returns all future root ids emerging from this root
#     This search happens in a monotic fashion. At no point are past root
#     ids of future root ids taken into account.
#     :param root_id: np.uint64
#     :param time_stamp: None or datetime
#         restrict search to ids created before this time_stamp
#         None=search whole future
#     :return: array of uint64
#     """
#     time_stamp = misc_utils.get_valid_timestamp(time_stamp)
#     id_history = []
#     next_ids = [root_id]
#     while len(next_ids):
#         temp_next_ids = []
#         for next_id in next_ids:
#             row = self.client.read_node(
#                 next_id,
#                 properties=[
#                     attributes.Hierarchy.NewParent,
#                     attributes.Hierarchy.Child,
#                 ],
#             )
#             if attributes.Hierarchy.NewParent in row:
#                 ids = row[attributes.Hierarchy.NewParent][0].value
#                 row_time_stamp = row[attributes.Hierarchy.NewParent][0].timestamp
#             elif attributes.Hierarchy.Child in row:
#                 ids = None
#                 row_time_stamp = row[attributes.Hierarchy.Child][0].timestamp
#             else:
#                 raise exceptions.ChunkedGraphError(
#                     "Error retrieving future root ID of %s" % next_id
#                 )

#             if row_time_stamp < time_stamp:
#                 if ids is not None:
#                     temp_next_ids.extend(ids)
#                 if next_id != root_id:
#                     id_history.append(next_id)

#         next_ids = temp_next_ids
#     return np.unique(np.array(id_history, dtype=np.uint64))


# def get_past_root_ids(
#     self,
#     root_id: np.uint64,
#     time_stamp: typing.Optional[datetime.datetime] = misc_utils.get_min_time(),
# ) -> np.ndarray:
#     """ Returns all past root ids emerging from this root
#     This search happens in a monotic fashion. At no point are future root
#     ids of past root ids taken into account.
#     :param root_id: np.uint64
#     :param time_stamp: None or datetime
#         restrict search to ids created after this time_stamp
#         None=search whole future
#     :return: array of uint64
#     """
#     time_stamp = misc_utils.get_valid_timestamp(time_stamp)
#     id_history = []
#     next_ids = [root_id]
#     while len(next_ids):
#         temp_next_ids = []
#         for next_id in next_ids:
#             row = self.client.read_node(
#                 next_id,
#                 properties=[
#                     attributes.Hierarchy.FormerParent,
#                     attributes.Hierarchy.Child,
#                 ],
#             )
#             if attributes.Hierarchy.FormerParent in row:
#                 ids = row[attributes.Hierarchy.FormerParent][0].value
#                 row_time_stamp = row[attributes.Hierarchy.FormerParent][0].timestamp
#             elif attributes.Hierarchy.Child in row:
#                 ids = None
#                 row_time_stamp = row[attributes.Hierarchy.Child][0].timestamp
#             else:
#                 raise exceptions.ChunkedGraphError(
#                     "Error retrieving past root ID of %s" % next_id
#                 )

#             if row_time_stamp > time_stamp:
#                 if ids is not None:
#                     temp_next_ids.extend(ids)

#                 if next_id != root_id:
#                     id_history.append(next_id)

#         next_ids = temp_next_ids
#     return np.unique(np.array(id_history, dtype=np.uint64))

# def get_delta_roots

