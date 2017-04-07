from operator import itemgetter
import networkx as nx
import matplotlib.pyplot as plt
import copy
import dep_tree

class MaxSpanTree:
	def __init__(self, graph):
		# Prepare the graph for max spanning tree algorithm
		sorted_graph = sorted(graph, key=itemgetter(1))
		sorted_graph.reverse()
		# Construct the maximum spanning tree
		self.tree = None
		self.build_span_tree(sorted_graph, False)

	def build_span_tree(self, graph_list, show_tree):
		G = nx.Graph()
		root_set = False
		for edge in graph_list:
			if show_tree == True:
				print(edge)
			G.add_edge(edge[0][0]+1, edge[0][1]+1)
			if root_set == False:
				G.node[edge[0][0]+1]['root']=True
				root_set = True
			# Check if new node creates a cycle
			try:
				# Cycle found, remove the new node
				nx.find_cycle(G)
				G.remove_edge(edge[0][0]+1, edge[0][1]+1)
			except nx.exception.NetworkXNoCycle:
				continue
		if show_tree == True:
			print("Nodes: ", G.number_of_nodes())
			nx.draw_circular(G, with_labels=True)
			plt.show()
			print(G.nodes())
			
		self.tree = G
		
