from collections import deque
import data_file



class sets():
	def __init__(self):
		self.generated_veh = [deque([]) for _ in data_file.lanes]
		self.unspawned_veh = [deque([]) for _ in data_file.lanes]
		self.prov_veh = [deque([]) for _ in data_file.lanes]
		self.coord_veh = [deque([]) for _ in data_file.lanes]
		self.done_veh = [deque([]) for _ in data_file.lanes]

