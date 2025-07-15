"""
Graph query and analysis operations for financial knowledge graphs.
"""

import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class GraphQueryEngine:
    """Query engine for financial knowledge graphs."""
    
    def __init__(self, graph):
        self.graph = graph
        
    def find_nodes_by_label(self, label: str, fuzzy: bool = True) -> List:
        """Find nodes by label with optional fuzzy matching."""
        results = []
        label_lower = label.lower()
        
        for node in self.graph.nodes.values():
            if fuzzy:
                if label_lower in node.label.lower():
                    results.append(node)
            else:
                if node.label.lower() == label_lower:
                    results.append(node)
        
        return results
    
    def find_nodes_by_class(self, fibo_class: str) -> List:
        """Find nodes by FIBO class."""
        return self.graph.get_nodes_by_class(fibo_class)
    
    def find_path(self, start_node_id: str, end_node_id: str, max_depth: int = 5) -> Optional[List]:
        """Find shortest path between two nodes."""
        if start_node_id == end_node_id:
            return [start_node_id]
        
        if start_node_id not in self.graph.nodes or end_node_id not in self.graph.nodes:
            return None
        
        # BFS to find shortest path
        queue = deque([(start_node_id, [start_node_id])])
        visited = {start_node_id}
        
        while queue:
            current_node, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            # Get neighbors
            neighbors = self.graph.get_neighbors(current_node)
            
            for neighbor in neighbors:
                if neighbor.id == end_node_id:
                    return path + [neighbor.id]
                
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    queue.append((neighbor.id, path + [neighbor.id]))
        
        return None
    
    def find_related_entities(self, node_id: str, relationship_types: Optional[List[str]] = None, 
                            max_depth: int = 2) -> Dict[str, List]:
        """Find entities related to a node within max_depth hops."""
        if node_id not in self.graph.nodes:
            return {}
        
        related = defaultdict(list)
        visited = {node_id}
        queue = deque([(node_id, 0)])
        
        while queue:
            current_node, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            # Get edges for current node
            edges = self.graph.get_edges_for_node(current_node)
            
            for edge in edges:
                # Filter by relationship type if specified
                if relationship_types and edge.relationship not in relationship_types:
                    continue
                
                # Determine the neighbor node
                neighbor_id = edge.target if edge.source == current_node else edge.source
                
                if neighbor_id not in visited:
                    neighbor_node = self.graph.get_node(neighbor_id)
                    if neighbor_node:
                        related[edge.relationship].append({
                            'node': neighbor_node,
                            'edge': edge,
                            'depth': depth + 1
                        })
                        
                        visited.add(neighbor_id)
                        queue.append((neighbor_id, depth + 1))
        
        return dict(related)
    
    def query_by_pattern(self, pattern: Dict[str, Any]) -> List[Dict]:
        """Query graph using a pattern specification."""
        results = []
        
        # Simple pattern matching
        # Pattern format: {'node_class': 'Organization', 'relationship': 'owns', 'target_class': 'Product'}
        
        source_class = pattern.get('node_class')
        relationship = pattern.get('relationship')
        target_class = pattern.get('target_class')
        
        if source_class:
            source_nodes = self.find_nodes_by_class(source_class)
            
            for source_node in source_nodes:
                edges = self.graph.get_edges_for_node(source_node.id)
                
                for edge in edges:
                    if relationship and edge.relationship != relationship:
                        continue
                    
                    # Get target node
                    target_id = edge.target if edge.source == source_node.id else edge.source
                    target_node = self.graph.get_node(target_id)
                    
                    if target_node:
                        if target_class and target_class not in target_node.fibo_class:
                            continue
                        
                        results.append({
                            'source': source_node,
                            'edge': edge,
                            'target': target_node
                        })
        
        return results
    
    def get_subgraph(self, node_ids: List[str], include_relationships: bool = True) -> Dict:
        """Extract subgraph containing specified nodes."""
        subgraph_nodes = {}
        subgraph_edges = []
        
        # Get nodes
        for node_id in node_ids:
            if node_id in self.graph.nodes:
                subgraph_nodes[node_id] = self.graph.nodes[node_id]
        
        # Get edges if requested
        if include_relationships:
            for edge in self.graph.edges:
                if edge.source in subgraph_nodes and edge.target in subgraph_nodes:
                    subgraph_edges.append(edge)
        
        return {
            'nodes': subgraph_nodes,
            'edges': subgraph_edges
        }

class GraphAnalyzer:
    """Analyzer for financial knowledge graph metrics and insights."""
    
    def __init__(self, graph):
        self.graph = graph
        
    def get_node_statistics(self) -> Dict[str, Any]:
        """Get detailed node statistics."""
        stats = {
            'total_nodes': len(self.graph.nodes),
            'nodes_by_class': defaultdict(int),
            'nodes_by_document': defaultdict(int),
            'confidence_distribution': {
                'high': 0,  # > 0.8
                'medium': 0,  # 0.5 - 0.8
                'low': 0  # < 0.5
            }
        }
        
        for node in self.graph.nodes.values():
            stats['nodes_by_class'][node.fibo_class] += 1
            
            for doc_ref in node.document_refs:
                stats['nodes_by_document'][doc_ref] += 1
            
            confidence = node.properties.get('confidence', 0)
            if confidence > 0.8:
                stats['confidence_distribution']['high'] += 1
            elif confidence > 0.5:
                stats['confidence_distribution']['medium'] += 1
            else:
                stats['confidence_distribution']['low'] += 1
        
        return stats
    
    def get_edge_statistics(self) -> Dict[str, Any]:
        """Get detailed edge statistics."""
        stats = {
            'total_edges': len(self.graph.edges),
            'edges_by_relationship': defaultdict(int),
            'confidence_distribution': {
                'high': 0,
                'medium': 0,
                'low': 0
            },
            'average_confidence': 0
        }
        
        total_confidence = 0
        
        for edge in self.graph.edges:
            stats['edges_by_relationship'][edge.relationship] += 1
            
            confidence = edge.confidence
            total_confidence += confidence
            
            if confidence > 0.8:
                stats['confidence_distribution']['high'] += 1
            elif confidence > 0.5:
                stats['confidence_distribution']['medium'] += 1
            else:
                stats['confidence_distribution']['low'] += 1
        
        if self.graph.edges:
            stats['average_confidence'] = total_confidence / len(self.graph.edges)
        
        return stats
    
    def find_central_nodes(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find most central nodes by degree centrality."""
        node_degrees = defaultdict(int)
        
        # Calculate degree for each node
        for edge in self.graph.edges:
            node_degrees[edge.source] += 1
            node_degrees[edge.target] += 1
        
        # Sort by degree
        central_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
        
        return central_nodes[:top_k]
    
    def find_clusters(self, min_cluster_size: int = 3) -> List[List[str]]:
        """Find clusters of highly connected nodes."""
        # Simple clustering based on connected components
        visited = set()
        clusters = []
        
        def dfs(node_id, cluster):
            if node_id in visited:
                return
            
            visited.add(node_id)
            cluster.append(node_id)
            
            # Visit neighbors
            neighbors = self.graph.get_neighbors(node_id)
            for neighbor in neighbors:
                if neighbor.id not in visited:
                    dfs(neighbor.id, cluster)
        
        for node_id in self.graph.nodes:
            if node_id not in visited:
                cluster = []
                dfs(node_id, cluster)
                
                if len(cluster) >= min_cluster_size:
                    clusters.append(cluster)
        
        return clusters
    
    def analyze_relationship_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in relationships."""
        patterns = {
            'most_common_relationships': defaultdict(int),
            'class_relationship_matrix': defaultdict(lambda: defaultdict(int)),
            'hub_nodes': [],  # Nodes with many outgoing relationships
            'authority_nodes': []  # Nodes with many incoming relationships
        }
        
        outgoing_count = defaultdict(int)
        incoming_count = defaultdict(int)
        
        for edge in self.graph.edges:
            patterns['most_common_relationships'][edge.relationship] += 1
            
            source_node = self.graph.get_node(edge.source)
            target_node = self.graph.get_node(edge.target)
            
            if source_node and target_node:
                patterns['class_relationship_matrix'][source_node.fibo_class][target_node.fibo_class] += 1
            
            outgoing_count[edge.source] += 1
            incoming_count[edge.target] += 1
        
        # Find hub and authority nodes
        hub_threshold = max(1, len(self.graph.edges) // 20)  # Top 5% by outgoing
        authority_threshold = max(1, len(self.graph.edges) // 20)  # Top 5% by incoming
        
        for node_id, count in outgoing_count.items():
            if count >= hub_threshold:
                node = self.graph.get_node(node_id)
                if node:
                    patterns['hub_nodes'].append((node.label, count))
        
        for node_id, count in incoming_count.items():
            if count >= authority_threshold:
                node = self.graph.get_node(node_id)
                if node:
                    patterns['authority_nodes'].append((node.label, count))
        
        # Sort by count
        patterns['hub_nodes'].sort(key=lambda x: x[1], reverse=True)
        patterns['authority_nodes'].sort(key=lambda x: x[1], reverse=True)
        
        return patterns
    
    def get_graph_summary(self) -> Dict[str, Any]:
        """Get comprehensive graph summary."""
        node_stats = self.get_node_statistics()
        edge_stats = self.get_edge_statistics()
        central_nodes = self.find_central_nodes(5)
        clusters = self.find_clusters()
        patterns = self.analyze_relationship_patterns()
        
        return {
            'overview': {
                'total_nodes': node_stats['total_nodes'],
                'total_edges': edge_stats['total_edges'],
                'average_edge_confidence': edge_stats['average_confidence'],
                'number_of_clusters': len(clusters)
            },
            'node_statistics': node_stats,
            'edge_statistics': edge_stats,
            'top_central_nodes': central_nodes,
            'relationship_patterns': patterns,
            'largest_clusters': sorted(clusters, key=len, reverse=True)[:3]
        }