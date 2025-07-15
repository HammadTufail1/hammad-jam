"""
FIBO ontology parser for extracting financial entity definitions and relationships.
"""

import json
import logging
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from rdflib import Graph, Namespace, RDF, RDFS, OWL
import requests

logger = logging.getLogger(__name__)

@dataclass
class FiboClass:
    """Represents a FIBO class with its properties."""
    uri: str
    label: str
    definition: str
    parent_classes: List[str]
    properties: List[str]

@dataclass
class FiboProperty:
    """Represents a FIBO property."""
    uri: str
    label: str
    definition: str
    domain: List[str]
    range: List[str]

@dataclass
class FiboRelationship:
    """Represents a relationship between FIBO entities."""
    subject: str
    predicate: str
    object: str
    relationship_type: str

class FiboParser:
    """Parser for FIBO (Financial Industry Business Ontology) files."""
    
    def __init__(self):
        self.graph = Graph()
        self.classes: Dict[str, FiboClass] = {}
        self.properties: Dict[str, FiboProperty] = {}
        self.relationships: List[FiboRelationship] = []
        
        # Common FIBO namespaces
        self.fibo_ns = Namespace("https://spec.edmcouncil.org/fibo/ontology/")
        self.skos = Namespace("http://www.w3.org/2004/02/skos/core#")
        
    def load_ontology_file(self, file_path: str) -> bool:
        """Load FIBO ontology from RDF file."""
        try:
            self.graph.parse(file_path, format="xml")
            logger.info(f"Successfully loaded ontology from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return False
    
    def load_ontology_url(self, url: str) -> bool:
        """Load FIBO ontology from URL."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Save to temporary file and parse
            temp_file = "/tmp/fibo_temp.rdf"
            with open(temp_file, 'wb') as f:
                f.write(response.content)
            
            return self.load_ontology_file(temp_file)
        except Exception as e:
            logger.error(f"Failed to load ontology from {url}: {e}")
            return False
    
    def extract_classes(self) -> Dict[str, FiboClass]:
        """Extract all FIBO classes from the loaded ontology."""
        for subj in self.graph.subjects(RDF.type, OWL.Class):
            if str(subj).startswith(str(self.fibo_ns)):
                fibo_class = self._create_fibo_class(subj)
                if fibo_class:
                    self.classes[str(subj)] = fibo_class
        
        logger.info(f"Extracted {len(self.classes)} FIBO classes")
        return self.classes
    
    def extract_properties(self) -> Dict[str, FiboProperty]:
        """Extract all FIBO properties from the loaded ontology."""
        # Object properties
        for subj in self.graph.subjects(RDF.type, OWL.ObjectProperty):
            if str(subj).startswith(str(self.fibo_ns)):
                prop = self._create_fibo_property(subj)
                if prop:
                    self.properties[str(subj)] = prop
        
        # Data properties
        for subj in self.graph.subjects(RDF.type, OWL.DatatypeProperty):
            if str(subj).startswith(str(self.fibo_ns)):
                prop = self._create_fibo_property(subj)
                if prop:
                    self.properties[str(subj)] = prop
        
        logger.info(f"Extracted {len(self.properties)} FIBO properties")
        return self.properties
    
    def extract_relationships(self) -> List[FiboRelationship]:
        """Extract relationships between FIBO entities."""
        relationship_count = 0
        
        for subj, pred, obj in self.graph:
            if (str(subj).startswith(str(self.fibo_ns)) or 
                str(obj).startswith(str(self.fibo_ns))):
                
                rel_type = self._determine_relationship_type(pred)
                relationship = FiboRelationship(
                    subject=str(subj),
                    predicate=str(pred),
                    object=str(obj),
                    relationship_type=rel_type
                )
                self.relationships.append(relationship)
                relationship_count += 1
        
        logger.info(f"Extracted {relationship_count} FIBO relationships")
        return self.relationships
    
    def _create_fibo_class(self, uri) -> Optional[FiboClass]:
        """Create a FiboClass object from RDF data."""
        try:
            label = self._get_label(uri)
            definition = self._get_definition(uri)
            parent_classes = self._get_parent_classes(uri)
            properties = self._get_class_properties(uri)
            
            return FiboClass(
                uri=str(uri),
                label=label,
                definition=definition,
                parent_classes=parent_classes,
                properties=properties
            )
        except Exception as e:
            logger.warning(f"Failed to create FIBO class for {uri}: {e}")
            return None
    
    def _create_fibo_property(self, uri) -> Optional[FiboProperty]:
        """Create a FiboProperty object from RDF data."""
        try:
            label = self._get_label(uri)
            definition = self._get_definition(uri)
            domain = self._get_property_domain(uri)
            range_vals = self._get_property_range(uri)
            
            return FiboProperty(
                uri=str(uri),
                label=label,
                definition=definition,
                domain=domain,
                range=range_vals
            )
        except Exception as e:
            logger.warning(f"Failed to create FIBO property for {uri}: {e}")
            return None
    
    def _get_label(self, uri) -> str:
        """Get the label for a URI."""
        for label_pred in [RDFS.label, self.skos.prefLabel]:
            for label in self.graph.objects(uri, label_pred):
                return str(label)
        return str(uri).split('/')[-1]  # Fallback to URI fragment
    
    def _get_definition(self, uri) -> str:
        """Get the definition for a URI."""
        for def_pred in [self.skos.definition, RDFS.comment]:
            for definition in self.graph.objects(uri, def_pred):
                return str(definition)
        return ""
    
    def _get_parent_classes(self, uri) -> List[str]:
        """Get parent classes for a class URI."""
        parents = []
        for parent in self.graph.objects(uri, RDFS.subClassOf):
            parents.append(str(parent))
        return parents
    
    def _get_class_properties(self, uri) -> List[str]:
        """Get properties associated with a class."""
        properties = []
        for prop in self.graph.subjects(RDFS.domain, uri):
            properties.append(str(prop))
        return properties
    
    def _get_property_domain(self, uri) -> List[str]:
        """Get domain classes for a property."""
        domains = []
        for domain in self.graph.objects(uri, RDFS.domain):
            domains.append(str(domain))
        return domains
    
    def _get_property_range(self, uri) -> List[str]:
        """Get range classes for a property."""
        ranges = []
        for range_val in self.graph.objects(uri, RDFS.range):
            ranges.append(str(range_val))
        return ranges
    
    def _determine_relationship_type(self, predicate) -> str:
        """Determine the type of relationship based on predicate."""
        pred_str = str(predicate).lower()
        
        if 'subclass' in pred_str:
            return 'subclass'
        elif 'type' in pred_str:
            return 'type'
        elif 'domain' in pred_str:
            return 'domain'
        elif 'range' in pred_str:
            return 'range'
        elif 'equivalent' in pred_str:
            return 'equivalent'
        else:
            return 'related'
    
    def get_financial_entity_types(self) -> Dict[str, List[str]]:
        """Categorize FIBO classes into financial entity types."""
        entity_types = {
            'organizations': [],
            'financial_instruments': [],
            'agents': [],
            'products': [],
            'services': [],
            'markets': [],
            'contracts': [],
            'other': []
        }
        
        for uri, fibo_class in self.classes.items():
            label_lower = fibo_class.label.lower()
            
            if any(keyword in label_lower for keyword in ['organization', 'institution', 'company', 'corporation']):
                entity_types['organizations'].append(uri)
            elif any(keyword in label_lower for keyword in ['instrument', 'security', 'bond', 'equity']):
                entity_types['financial_instruments'].append(uri)
            elif any(keyword in label_lower for keyword in ['agent', 'person', 'individual', 'party']):
                entity_types['agents'].append(uri)
            elif any(keyword in label_lower for keyword in ['product', 'offering']):
                entity_types['products'].append(uri)
            elif any(keyword in label_lower for keyword in ['service']):
                entity_types['services'].append(uri)
            elif any(keyword in label_lower for keyword in ['market', 'exchange']):
                entity_types['markets'].append(uri)
            elif any(keyword in label_lower for keyword in ['contract', 'agreement']):
                entity_types['contracts'].append(uri)
            else:
                entity_types['other'].append(uri)
        
        return entity_types
    
    def save_parsed_data(self, output_file: str) -> bool:
        """Save parsed FIBO data to JSON file."""
        try:
            data = {
                'classes': {uri: {
                    'label': cls.label,
                    'definition': cls.definition,
                    'parent_classes': cls.parent_classes,
                    'properties': cls.properties
                } for uri, cls in self.classes.items()},
                'properties': {uri: {
                    'label': prop.label,
                    'definition': prop.definition,
                    'domain': prop.domain,
                    'range': prop.range
                } for uri, prop in self.properties.items()},
                'relationships': [{
                    'subject': rel.subject,
                    'predicate': rel.predicate,
                    'object': rel.object,
                    'relationship_type': rel.relationship_type
                } for rel in self.relationships],
                'entity_types': self.get_financial_entity_types()
            }
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved parsed FIBO data to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save parsed data: {e}")
            return False
    
    def load_parsed_data(self, input_file: str) -> bool:
        """Load parsed FIBO data from JSON file."""
        try:
            with open(input_file, 'r') as f:
                data = json.load(f)
            
            # Reconstruct classes
            self.classes = {}
            for uri, cls_data in data['classes'].items():
                self.classes[uri] = FiboClass(
                    uri=uri,
                    label=cls_data['label'],
                    definition=cls_data['definition'],
                    parent_classes=cls_data['parent_classes'],
                    properties=cls_data['properties']
                )
            
            # Reconstruct properties
            self.properties = {}
            for uri, prop_data in data['properties'].items():
                self.properties[uri] = FiboProperty(
                    uri=uri,
                    label=prop_data['label'],
                    definition=prop_data['definition'],
                    domain=prop_data['domain'],
                    range=prop_data['range']
                )
            
            # Reconstruct relationships
            self.relationships = []
            for rel_data in data['relationships']:
                self.relationships.append(FiboRelationship(
                    subject=rel_data['subject'],
                    predicate=rel_data['predicate'],
                    object=rel_data['object'],
                    relationship_type=rel_data['relationship_type']
                ))
            
            logger.info(f"Loaded parsed FIBO data from {input_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to load parsed data: {e}")
            return False