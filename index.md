# Leveraging Neo4j for Service Identification and Microservices Partitioning in Business Process Systems

# 1. Install Required Libraries


```python
!pip install neo4j torch pyvis pandas
```

    Requirement already satisfied: neo4j in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (5.25.0)
    Requirement already satisfied: torch in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (2.5.0)
    Requirement already satisfied: pyvis in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (0.3.2)
    Requirement already satisfied: pandas in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (2.2.3)
    Requirement already satisfied: pytz in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from neo4j) (2024.1)
    Requirement already satisfied: filelock in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from torch) (3.13.1)
    Requirement already satisfied: typing-extensions>=4.8.0 in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from torch) (4.11.0)
    Requirement already satisfied: setuptools in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from torch) (75.1.0)
    Requirement already satisfied: sympy==1.13.1 in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from torch) (1.13.1)
    Requirement already satisfied: networkx in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from torch) (3.2.1)
    Requirement already satisfied: jinja2 in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from torch) (3.1.4)
    Requirement already satisfied: fsspec in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from torch) (2024.9.0)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from sympy==1.13.1->torch) (1.3.0)
    Requirement already satisfied: ipython>=5.3.0 in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from pyvis) (8.27.0)
    Requirement already satisfied: jsonpickle>=1.4.1 in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from pyvis) (3.3.0)
    Requirement already satisfied: numpy>=1.26.0 in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from pandas) (2.0.1)
    Requirement already satisfied: python-dateutil>=2.8.2 in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from pandas) (2.9.0.post0)
    Requirement already satisfied: tzdata>=2022.7 in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from pandas) (2024.2)
    Requirement already satisfied: decorator in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from ipython>=5.3.0->pyvis) (5.1.1)
    Requirement already satisfied: jedi>=0.16 in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from ipython>=5.3.0->pyvis) (0.19.1)
    Requirement already satisfied: matplotlib-inline in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from ipython>=5.3.0->pyvis) (0.1.6)
    Requirement already satisfied: prompt-toolkit<3.1.0,>=3.0.41 in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from ipython>=5.3.0->pyvis) (3.0.43)
    Requirement already satisfied: pygments>=2.4.0 in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from ipython>=5.3.0->pyvis) (2.15.1)
    Requirement already satisfied: stack-data in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from ipython>=5.3.0->pyvis) (0.2.0)
    Requirement already satisfied: traitlets>=5.13.0 in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from ipython>=5.3.0->pyvis) (5.14.3)
    Requirement already satisfied: colorama in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from ipython>=5.3.0->pyvis) (0.4.6)
    Requirement already satisfied: MarkupSafe>=2.0 in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from jinja2->torch) (2.1.3)
    Requirement already satisfied: six>=1.5 in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)
    Requirement already satisfied: parso<0.9.0,>=0.8.3 in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from jedi>=0.16->ipython>=5.3.0->pyvis) (0.8.3)
    Requirement already satisfied: wcwidth in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from prompt-toolkit<3.1.0,>=3.0.41->ipython>=5.3.0->pyvis) (0.2.5)
    Requirement already satisfied: executing in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from stack-data->ipython>=5.3.0->pyvis) (0.8.3)
    Requirement already satisfied: asttokens in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from stack-data->ipython>=5.3.0->pyvis) (2.0.5)
    Requirement already satisfied: pure-eval in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from stack-data->ipython>=5.3.0->pyvis) (0.2.2)
    

# 2. Import Libraries

Import all necessary libraries for XML parsing, Neo4j interaction, GPU detection, and concurrent processing.


```python
# Import Libraries
import os
import re
import xml.etree.ElementTree as ET
import html
import uuid
import pandas as pd
from neo4j import GraphDatabase
from pyvis.network import Network
import torch
from concurrent.futures import ThreadPoolExecutor
```

# 3. Check CUDA Availability

Detect whether CUDA (GPU) is available on your system. This information will be printed at the beginning of the notebook.


```python
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

    CUDA available: True
    CUDA version: 11.8
    Using device: cuda
    


```python
# Function to check CUDA availability
def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available. GPU will be used if applicable.")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU.")

# Execute Cuda
check_cuda()
```

    CUDA is available. GPU will be used if applicable.
    Device Name: NVIDIA GeForce RTX 3060 Laptop GPU
    

# 4. Define Connection to Neo4j


```python
# Neo4j connection details
uri = "bolt://localhost:7687"
username = "neo4j"
password = "170202Kcf"

# Create a driver instance
driver = GraphDatabase.driver(uri, auth=(username, password))
```


```python
def test_connection():
    try:
        with driver.session(database="erpbpmn") as session:
            result = session.run("RETURN 1 AS test")
            for record in result:
                print(f"Connection successful, test query result: {record['test']}")
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")

test_connection()
```

    Connection successful, test query result: 1
    

# 5. Define Functions to Create Nodes and Relationships


```python
def get_node_color(node_type, level):
    color_map = {
        'Task': {
            0: '#FFD700',  # Gold
            1: '#FFFACD',  # LemonChiffon
            2: '#FAFAD2',  # LightGoldenrodYellow
            3: '#FFFFE0'   # LightYellow
        },
        'StartEvent': {
            0: '#90EE90',  # LightGreen
            1: '#98FB98',  # PaleGreen
            2: '#8FBC8F',  # DarkSeaGreen
            3: '#3CB371'   # MediumSeaGreen
        },
        'EndEvent': {
            0: '#FF6347',  # Tomato
            1: '#FF4500',  # OrangeRed
            2: '#FF0000',  # Red
            3: '#DC143C'   # Crimson
        }
    }
    default_color = '#D3D3D3'
    return color_map.get(node_type, {}).get(level, default_color)

def create_node(tx, label, properties):
    color = get_node_color(label, properties.get('level', 0))
    query = (
        f"CREATE (n:{label} {{id: $properties.id}}) "
        "SET n += $properties, n.color = $color "
        "RETURN n"
    )
    result = tx.run(query, properties=properties, color=color)
    return result.single()[0]

def create_relationship_with_id(tx, source_id, target_id, rel_type, properties):
    rel_color_map = {
        'SEQUENCE_FLOW': '#A9A9A9',
        'XOR_SPLIT': '#FF69B4',
        'XOR_JOIN': '#4169E1',
        'OR_SPLIT': '#FFD700',
        'OR_JOIN': '#00CED1',
        'DECOMPOSED_INTO': '#00FF00'
    }
    color = rel_color_map.get(rel_type, '#696969')
    query = (
        f"MATCH (a {{id: $source_id}}), (b {{id: $target_id}}) "
        f"CREATE (a)-[r:{rel_type} {{id: $properties.id}}]->(b) "
        "SET r += $properties, r.color = $color "
        "RETURN r"
    )
    result = tx.run(query, source_id=source_id, target_id=target_id, properties=properties, color=color)
    record = result.single()
    if record:
        return record[0]
    else:
        print(f"Warning: Could not create relationship {rel_type} between {source_id} and {target_id}.")
        return None
```

# 6. Parse BPMN XML Files and Load into Neo4j


```python
# Define functions to parse BPMN XML files
def clean_name(name):
    name = re.sub('<[^<]+?>', '', name)
    name = html.unescape(name)
    return name.strip()
```


```python
def parse_bpmn_xml(file_path, level, module, activity=None):
    tree = ET.parse(file_path)
    root = tree.getroot()

    elements = {
        'Task': [],
        'StartEvent': [],
        'EndEvent': []
    }
    flows = []
    gateways = {}

    mx_root = root.find('.//root')

    if mx_root is None:
        print("No mxGraphModel root found in the XML.")
        return elements, flows, gateways

    cells = mx_root.findall('mxCell')

    id_prefix = f"{module}_{activity}_{level}_" if activity else f"{module}_{level}_"

    for cell in cells:
        cell_id = cell.get('id')
        value = cell.get('value', '').strip()
        value = clean_name(value)
        style = cell.get('style', '')
        vertex = cell.get('vertex')
        edge = cell.get('edge')

        if vertex == '1':
            # It's a node
            if 'shape=mxgraph.bpmn.task' in style:
                # It's a Task
                elements['Task'].append({
                    'id': id_prefix + cell_id,
                    'name': value if value else 'Unnamed Task',
                    'level': level,
                    'module': module,
                    'activity': activity
                })
            elif 'shape=mxgraph.bpmn.event' in style:
                # It's an Event
                if 'fillColor=#60a917' in style:
                    # Start Event (green)
                    elements['StartEvent'].append({
                        'id': id_prefix + cell_id,
                        'name': value if value else 'Start',
                        'level': level,
                        'module': module,
                        'activity': activity
                    })
                elif 'fillColor=#e51400' in style:
                    # End Event (red)
                    elements['EndEvent'].append({
                        'id': id_prefix + cell_id,
                        'name': value if value else 'End',
                        'level': level,
                        'module': module,
                        'activity': activity
                    })
            elif 'shape=mxgraph.bpmn.gateway2' in style:
                # It's a Gateway
                if 'gwType=exclusive' in style:
                    gateway_kind = 'XOR'
                elif 'gwType=inclusive' in style:
                    gateway_kind = 'OR'
                else:
                    gateway_kind = 'UNKNOWN'

                gateways[cell_id] = {
                    'id': id_prefix + cell_id,
                    'gateway_kind': gateway_kind,
                    'level': level,
                    'module': module,
                    'activity': activity
                }
        elif edge == '1':
            # It's an edge
            source = cell.get('source')
            target = cell.get('target')
            if source and target:
                flows.append({
                    'id': id_prefix + cell_id,
                    'sourceRef': id_prefix + source,
                    'targetRef': id_prefix + target,
                    'name': value if value else 'Unnamed Flow',
                    'level': level,
                    'module': module,
                    'activity': activity
                })

    return elements, flows, gateways
```


```python
def process_gateways_and_flows(session, elements, flows, gateways):
    # Remove gateway nodes from elements
    # (we are not creating nodes for gateways)
    # Instead, we process flows to create direct relationships
    gateway_ids = gateways.keys()

    # Build maps for incoming and outgoing flows for each gateway
    incoming_flows = {}
    outgoing_flows = {}

    for flow in flows:
        source = flow['sourceRef']
        target = flow['targetRef']
        if target in gateway_ids:
            if target not in incoming_flows:
                incoming_flows[target] = []
            incoming_flows[target].append(flow)
        if source in gateway_ids:
            if source not in outgoing_flows:
                outgoing_flows[source] = []
            outgoing_flows[source].append(flow)

    # New flows to be created after processing gateways
    new_flows = []

    for gw_id, gw_info in gateways.items():
        gw_type = gw_info['gateway_kind']
        gw_incoming = incoming_flows.get(gw_id, [])
        gw_outgoing = outgoing_flows.get(gw_id, [])

        if len(gw_incoming) > 1 and len(gw_outgoing) == 1:
            # Join Gateway
            rel_type = f"{gw_type}_JOIN"
            target = gw_outgoing[0]['targetRef']
            for inc_flow in gw_incoming:
                source = inc_flow['sourceRef']
                new_flows.append({'source': source, 'target': target, 'type': rel_type, 'properties': inc_flow})
        elif len(gw_incoming) == 1 and len(gw_outgoing) > 1:
            # Split Gateway
            rel_type = f"{gw_type}_SPLIT"
            source = gw_incoming[0]['sourceRef']
            for out_flow in gw_outgoing:
                target = out_flow['targetRef']
                new_flows.append({'source': source, 'target': target, 'type': rel_type, 'properties': out_flow})
        else:
            # For other cases, create SEQUENCE_FLOW relationships
            print(f"Warning: Gateway {gw_id} has an unexpected number of incoming or outgoing flows.")
            # Connect each incoming flow to each outgoing flow
            for inc_flow in gw_incoming:
                source = inc_flow['sourceRef']
                for out_flow in gw_outgoing:
                    target = out_flow['targetRef']
                    new_flows.append({'source': source, 'target': target, 'type': 'SEQUENCE_FLOW', 'properties': out_flow})

    # Remove flows connected to gateways
    flows = [flow for flow in flows if flow['sourceRef'] not in gateway_ids and flow['targetRef'] not in gateway_ids]

    # Add new flows
    for flow in new_flows:
        rel_properties = flow['properties']
        rel_properties['name'] = flow['type']
        rel_type = flow['type']
        session.execute_write(
            create_relationship_with_id,
            flow['source'],
            flow['target'],
            rel_type,
            rel_properties
        )

    # Process remaining flows
    for flow in flows:
        rel_properties = {
            'id': flow['id'],
            'name': 'SEQUENCE_FLOW',
            'level': flow['level'],
            'module': flow['module'],
            'activity': flow['activity']
        }
        session.execute_write(
            create_relationship_with_id,
            flow['sourceRef'],
            flow['targetRef'],
            'SEQUENCE_FLOW',
            rel_properties
        )
```


```python
def find_parent_node(session, name, module, level):
    query = """
    MATCH (n {name: $name, module: $module, level: $parent_level})
    RETURN n.id AS id
    """
    result = session.run(query, name=name, module=module, parent_level=level - 1)
    record = result.single()
    if record:
        return record['id']
    else:
        return None

def create_decomposition_relationship(session, parent_id, child_id):
    rel_properties = {'id': f"{parent_id}_to_{child_id}"}
    session.execute_write(
        create_relationship_with_id,
        parent_id,
        child_id,
        'DECOMPOSED_INTO',
        rel_properties
    )
```


```python
def process_bpmn_file(session, filename, file_path, level, module, activity):
    print(f"\nProcessing {filename} at Level {level}...")
    elements, flows, gateways = parse_bpmn_xml(file_path, level, module, activity)

    total_elements = sum(len(v) for v in elements.values())
    print(f"Parsed {total_elements} elements and {len(flows)} flows from {filename}")

    # Identify unnamed nodes and rename them
    for element_type in ['Task', 'StartEvent', 'EndEvent']:
        for element in elements[element_type]:
            if not element['name'] or element['name'].startswith('Unnamed'):
                if element_type == 'StartEvent':
                    element['name'] = 'Start'
                elif element_type == 'EndEvent':
                    element['name'] = 'End'
                else:
                    element['name'] = f"{element_type}_{element['id']}"
                print(f"Renamed {element_type} {element['id']} to {element['name']}")

    # Create nodes (excluding Gateways)
    for element_type, element_list in elements.items():
        for element in element_list:
            session.execute_write(create_node, element_type, element)
            print(f"Created node: {element_type} with ID {element['id']} and name {element['name']}")

    # Process Gateways and Flows
    process_gateways_and_flows(session, elements, flows, gateways)

    # Create decomposition relationships between levels
    if level > 0:
        for element_type, element_list in elements.items():
            for element in element_list:
                parent_id = find_parent_node(session, element['name'], module, level)
                if parent_id:
                    create_decomposition_relationship(session, parent_id, element['id'])
                    print(f"Created DECOMPOSED_INTO relationship from {parent_id} to {element['id']}")
```

# 7. Main Execution


```python
def main():
    bpmn_dir = './assets'  # Directory containing BPMN XML files
    filenames = [f for f in os.listdir(bpmn_dir) if f.endswith('.xml')]

    with driver.session(database="erpbpmn") as session:
        for filename in filenames:
            file_path = os.path.join(bpmn_dir, filename)
            if filename == "BPMN Level 0.xml":
                level = 0
                module = "ERP"
                activity = None
            else:
                match = re.match(r'BPMN\s+(.+?)\s+Level\s+(\d+)(?:\s+-\s+(.+))?\.xml', filename)
                if match:
                    module = match.group(1).strip()
                    level = int(match.group(2))
                    activity = match.group(3).strip() if match.group(3) else None
                else:
                    print(f"Skipping file {filename} as it doesn't match the expected naming pattern.")
                    continue

            process_bpmn_file(session, filename, file_path, level, module, activity)
```


```python
def verify_data_import():
    with driver.session(database="erpbpmn") as session:
        result = session.run("MATCH (n) RETURN labels(n) AS Label, count(n) AS Count ORDER BY Count DESC")
        print("Node counts by label:")
        for record in result:
            print(f"{record['Label']}: {record['Count']}")

        result = session.run("MATCH ()-[r]->() RETURN type(r) AS RelationType, count(r) AS Count ORDER BY Count DESC")
        print("\nRelationship counts by type:")
        for record in result:
            print(f"{record['RelationType']}: {record['Count']}")

        result = session.run("MATCH (n) RETURN n.level AS Level, count(n) AS Count ORDER BY Level")
        print("\nNode counts by level:")
        for record in result:
            print(f"Level {record['Level']}: {record['Count']} nodes")
```

# 9. Execute Main Function


```python
def run():
    main()
    verify_data_import()

if __name__ == "__main__":
    run()
```

    
    Processing BPMN Account Payable Level 1.xml at Level 1...
    Parsed 3 elements and 2 flows from BPMN Account Payable Level 1.xml
    Created node: Task with ID Account Payable_1_5 and name Account Payable
    Created node: StartEvent with ID Account Payable_1_3 and name Start
    Created node: EndEvent with ID Account Payable_1_7 and name End
    
    Processing BPMN Account Payable Level 2.xml at Level 2...
    Parsed 23 elements and 39 flows from BPMN Account Payable Level 2.xml
    Created node: Task with ID Account Payable_2_24 and name Reviewing Purchase Return
    Created node: Task with ID Account Payable_2_25 and name Approving Purchase Return
    Created node: Task with ID Account Payable_2_26 and name Checking Purchase Invoice Detail
    Created node: Task with ID Account Payable_2_28 and name Updating Data Purchase Invoice
    Created node: Task with ID Account Payable_2_29 and name Finalizing Document
    Created node: Task with ID Account Payable_2_30 and name Reviewing Purchase Return
    Created node: Task with ID Account Payable_2_31 and name Checking Purchase Invoice Detail
    Created node: Task with ID Account Payable_2_32 and name Approving Purchase Return
    Created node: Task with ID Account Payable_2_33 and name Updating Data Purchase Invoice
    Created node: Task with ID Account Payable_2_43 and name Completing Purchase Invoice Form
    Created node: Task with ID Account Payable_2_46 and name Receiving Purchase Order
    Created node: Task with ID Account Payable_2_48 and name Receiving Purchase Order Down Payment
    Created node: Task with ID Account Payable_2_50 and name Checking Goods for Return or Acceptance
    Created node: Task with ID Account Payable_2_52 and name Checking Goods for Return or Acceptance
    Created node: Task with ID Account Payable_2_57 and name Processing Purchase Return
    Created node: Task with ID Account Payable_2_64 and name Validating Invoice Details
    Created node: Task with ID Account Payable_2_65 and name Completing Purchase DP Invoice Form
    Created node: Task with ID Account Payable_2_68 and name Processing Purchase DP Return
    Created node: Task with ID Account Payable_2_72 and name Creating the Purchase DP Invoice
    Created node: Task with ID Account Payable_2_76 and name Validating Invoice DP Details
    Created node: Task with ID Account Payable_2_78 and name Creating the Purchase Invoice
    Created node: StartEvent with ID Account Payable_2_80 and name Start
    Created node: EndEvent with ID Account Payable_2_22 and name End
    Warning: Gateway 12 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 15 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 16 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 18 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 42 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 45 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 55 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 60 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 67 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 73 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 81 has an unexpected number of incoming or outgoing flows.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_2_81 and Account Payable_2_12.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_2_81 and Account Payable_2_42.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_2_12 and Account Payable_2_15.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_2_12 and Account Payable_2_24.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_2_15 and Account Payable_2_16.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_2_15 and Account Payable_2_26.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_2_16 and Account Payable_2_30.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_2_16 and Account Payable_2_18.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_2_18 and Account Payable_2_31.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_2_42 and Account Payable_2_45.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_2_45 and Account Payable_2_48.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_2_42 and Account Payable_2_46.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_2_50 and Account Payable_2_67.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_2_55 and Account Payable_2_60.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_2_52 and Account Payable_2_55.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_2_55 and Account Payable_2_57.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_2_60 and Account Payable_2_78.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_2_67 and Account Payable_2_73.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_2_67 and Account Payable_2_68.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_2_73 and Account Payable_2_72.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_2_80 and Account Payable_2_81.
    Created DECOMPOSED_INTO relationship from Account Payable_1_3 to Account Payable_2_80
    Created DECOMPOSED_INTO relationship from Account Payable_1_7 to Account Payable_2_22
    
    Processing BPMN Account Payable Level 3 - Creating the Purchase DP Invoice.xml at Level 3...
    Parsed 8 elements and 17 flows from BPMN Account Payable Level 3 - Creating the Purchase DP Invoice.xml
    Created node: Task with ID Account Payable_Creating the Purchase DP Invoice_3_5 and name Receive Purchase Order Down Payment
    Created node: Task with ID Account Payable_Creating the Purchase DP Invoice_3_6 and name Check Goods for Acceptance
    Created node: Task with ID Account Payable_Creating the Purchase DP Invoice_3_14 and name Process Purchase DP Return
    Created node: Task with ID Account Payable_Creating the Purchase DP Invoice_3_16 and name Create Purchase DP Invoice
    Created node: Task with ID Account Payable_Creating the Purchase DP Invoice_3_29 and name Validate Invoice DP Details
    Created node: Task with ID Account Payable_Creating the Purchase DP Invoice_3_30 and name Complete Purchase DP Invoice Form
    Created node: StartEvent with ID Account Payable_Creating the Purchase DP Invoice_3_4 and name Start
    Created node: EndEvent with ID Account Payable_Creating the Purchase DP Invoice_3_35 and name End
    Warning: Gateway 11 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 17 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 36 has an unexpected number of incoming or outgoing flows.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Creating the Purchase DP Invoice_3_11 and Account Payable_Creating the Purchase DP Invoice_3_14.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Creating the Purchase DP Invoice_3_11 and Account Payable_Creating the Purchase DP Invoice_3_17.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Creating the Purchase DP Invoice_3_6 and Account Payable_Creating the Purchase DP Invoice_3_11.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Creating the Purchase DP Invoice_3_14 and Account Payable_Creating the Purchase DP Invoice_3_36.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Creating the Purchase DP Invoice_3_17 and Account Payable_Creating the Purchase DP Invoice_3_16.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Creating the Purchase DP Invoice_3_22 and Account Payable_Creating the Purchase DP Invoice_3_5.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Creating the Purchase DP Invoice_3_21 and Account Payable_Creating the Purchase DP Invoice_3_22.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Creating the Purchase DP Invoice_3_26 and Account Payable_Creating the Purchase DP Invoice_3_6.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Creating the Purchase DP Invoice_3_24 and Account Payable_Creating the Purchase DP Invoice_3_26.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Creating the Purchase DP Invoice_3_32 and Account Payable_Creating the Purchase DP Invoice_3_34.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Creating the Purchase DP Invoice_3_30 and Account Payable_Creating the Purchase DP Invoice_3_32.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Creating the Purchase DP Invoice_3_30 and Account Payable_Creating the Purchase DP Invoice_3_36.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Creating the Purchase DP Invoice_3_36 and Account Payable_Creating the Purchase DP Invoice_3_35.
    Created DECOMPOSED_INTO relationship from Account Payable_2_80 to Account Payable_Creating the Purchase DP Invoice_3_4
    Created DECOMPOSED_INTO relationship from Account Payable_2_22 to Account Payable_Creating the Purchase DP Invoice_3_35
    
    Processing BPMN Account Payable Level 3 - Creating the Purchase Invoice.xml at Level 3...
    Parsed 8 elements and 17 flows from BPMN Account Payable Level 3 - Creating the Purchase Invoice.xml
    Created node: Task with ID Account Payable_Creating the Purchase Invoice_3_5 and name Receive Purchase Order
    Created node: Task with ID Account Payable_Creating the Purchase Invoice_3_6 and name Check Goods for Acceptance
    Created node: Task with ID Account Payable_Creating the Purchase Invoice_3_14 and name Process Purchase Return
    Created node: Task with ID Account Payable_Creating the Purchase Invoice_3_16 and name Create Purchase Invoice
    Created node: Task with ID Account Payable_Creating the Purchase Invoice_3_29 and name Validate Invoice Details
    Created node: Task with ID Account Payable_Creating the Purchase Invoice_3_30 and name Complete Purchase Invoice Form
    Created node: StartEvent with ID Account Payable_Creating the Purchase Invoice_3_4 and name Start
    Created node: EndEvent with ID Account Payable_Creating the Purchase Invoice_3_35 and name End
    Warning: Gateway 11 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 17 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 36 has an unexpected number of incoming or outgoing flows.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Creating the Purchase Invoice_3_11 and Account Payable_Creating the Purchase Invoice_3_14.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Creating the Purchase Invoice_3_11 and Account Payable_Creating the Purchase Invoice_3_17.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Creating the Purchase Invoice_3_6 and Account Payable_Creating the Purchase Invoice_3_11.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Creating the Purchase Invoice_3_14 and Account Payable_Creating the Purchase Invoice_3_36.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Creating the Purchase Invoice_3_17 and Account Payable_Creating the Purchase Invoice_3_16.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Creating the Purchase Invoice_3_22 and Account Payable_Creating the Purchase Invoice_3_5.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Creating the Purchase Invoice_3_21 and Account Payable_Creating the Purchase Invoice_3_22.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Creating the Purchase Invoice_3_26 and Account Payable_Creating the Purchase Invoice_3_6.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Creating the Purchase Invoice_3_24 and Account Payable_Creating the Purchase Invoice_3_26.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Creating the Purchase Invoice_3_32 and Account Payable_Creating the Purchase Invoice_3_34.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Creating the Purchase Invoice_3_30 and Account Payable_Creating the Purchase Invoice_3_32.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Creating the Purchase Invoice_3_30 and Account Payable_Creating the Purchase Invoice_3_36.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Creating the Purchase Invoice_3_36 and Account Payable_Creating the Purchase Invoice_3_35.
    Created DECOMPOSED_INTO relationship from Account Payable_2_80 to Account Payable_Creating the Purchase Invoice_3_4
    Created DECOMPOSED_INTO relationship from Account Payable_2_22 to Account Payable_Creating the Purchase Invoice_3_35
    
    Processing BPMN Account Payable Level 3 - Finalizing Purchase DP Invoice Document.xml at Level 3...
    Parsed 5 elements and 8 flows from BPMN Account Payable Level 3 - Finalizing Purchase DP Invoice Document.xml
    Created node: Task with ID Account Payable_Finalizing Purchase DP Invoice Document_3_5 and name Retrieve DP Invoice Data
    Created node: Task with ID Account Payable_Finalizing Purchase DP Invoice Document_3_7 and name Verify Data Accuracy
    Created node: Task with ID Account Payable_Finalizing Purchase DP Invoice Document_3_17 and name Complete DP Invoice Document
    Created node: StartEvent with ID Account Payable_Finalizing Purchase DP Invoice Document_3_4 and name Start
    Created node: EndEvent with ID Account Payable_Finalizing Purchase DP Invoice Document_3_15 and name End
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Finalizing Purchase DP Invoice Document_3_10 and Account Payable_Finalizing Purchase DP Invoice Document_3_5.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Finalizing Purchase DP Invoice Document_3_9 and Account Payable_Finalizing Purchase DP Invoice Document_3_10.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Finalizing Purchase DP Invoice Document_3_13 and Account Payable_Finalizing Purchase DP Invoice Document_3_14.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Finalizing Purchase DP Invoice Document_3_17 and Account Payable_Finalizing Purchase DP Invoice Document_3_13.
    Created DECOMPOSED_INTO relationship from Account Payable_2_80 to Account Payable_Finalizing Purchase DP Invoice Document_3_4
    Created DECOMPOSED_INTO relationship from Account Payable_2_22 to Account Payable_Finalizing Purchase DP Invoice Document_3_15
    
    Processing BPMN Account Payable Level 3 - Finalizing Purchase Invoice Document.xml at Level 3...
    Parsed 5 elements and 8 flows from BPMN Account Payable Level 3 - Finalizing Purchase Invoice Document.xml
    Created node: Task with ID Account Payable_Finalizing Purchase Invoice Document_3_5 and name Retrieve  Invoice Data
    Created node: Task with ID Account Payable_Finalizing Purchase Invoice Document_3_7 and name Verify Data Accuracy
    Created node: Task with ID Account Payable_Finalizing Purchase Invoice Document_3_17 and name Complete  Invoice Document
    Created node: StartEvent with ID Account Payable_Finalizing Purchase Invoice Document_3_4 and name Start
    Created node: EndEvent with ID Account Payable_Finalizing Purchase Invoice Document_3_15 and name End
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Finalizing Purchase Invoice Document_3_10 and Account Payable_Finalizing Purchase Invoice Document_3_5.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Finalizing Purchase Invoice Document_3_9 and Account Payable_Finalizing Purchase Invoice Document_3_10.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Finalizing Purchase Invoice Document_3_13 and Account Payable_Finalizing Purchase Invoice Document_3_14.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Finalizing Purchase Invoice Document_3_17 and Account Payable_Finalizing Purchase Invoice Document_3_13.
    Created DECOMPOSED_INTO relationship from Account Payable_2_80 to Account Payable_Finalizing Purchase Invoice Document_3_4
    Created DECOMPOSED_INTO relationship from Account Payable_2_22 to Account Payable_Finalizing Purchase Invoice Document_3_15
    
    Processing BPMN Account Payable Level 3 - Reviewing Purchase DP Invoice.xml at Level 3...
    Parsed 6 elements and 13 flows from BPMN Account Payable Level 3 - Reviewing Purchase DP Invoice.xml
    Created node: Task with ID Account Payable_Reviewing Purchase DP Invoice_3_5 and name Receive Purchase DP Invoice
    Created node: Task with ID Account Payable_Reviewing Purchase DP Invoice_3_6 and name Return for Correction
    Created node: Task with ID Account Payable_Reviewing Purchase DP Invoice_3_16 and name Check Invoice Details
    Created node: Task with ID Account Payable_Reviewing Purchase DP Invoice_3_23 and name Approve Purchase DP Invoice
    Created node: StartEvent with ID Account Payable_Reviewing Purchase DP Invoice_3_4 and name Start
    Created node: EndEvent with ID Account Payable_Reviewing Purchase DP Invoice_3_14 and name End
    Warning: Gateway 19 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 20 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 28 has an unexpected number of incoming or outgoing flows.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Reviewing Purchase DP Invoice_3_9 and Account Payable_Reviewing Purchase DP Invoice_3_5.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Reviewing Purchase DP Invoice_3_8 and Account Payable_Reviewing Purchase DP Invoice_3_9.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Reviewing Purchase DP Invoice_3_12 and Account Payable_Reviewing Purchase DP Invoice_3_13.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Reviewing Purchase DP Invoice_3_16 and Account Payable_Reviewing Purchase DP Invoice_3_19.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Reviewing Purchase DP Invoice_3_19 and Account Payable_Reviewing Purchase DP Invoice_3_20.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Reviewing Purchase DP Invoice_3_19 and Account Payable_Reviewing Purchase DP Invoice_3_6.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Reviewing Purchase DP Invoice_3_20 and Account Payable_Reviewing Purchase DP Invoice_3_23.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Reviewing Purchase DP Invoice_3_23 and Account Payable_Reviewing Purchase DP Invoice_3_12.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Reviewing Purchase DP Invoice_3_28 and Account Payable_Reviewing Purchase DP Invoice_3_14.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Reviewing Purchase DP Invoice_3_6 and Account Payable_Reviewing Purchase DP Invoice_3_28.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Reviewing Purchase DP Invoice_3_23 and Account Payable_Reviewing Purchase DP Invoice_3_28.
    Created DECOMPOSED_INTO relationship from Account Payable_2_80 to Account Payable_Reviewing Purchase DP Invoice_3_4
    Created DECOMPOSED_INTO relationship from Account Payable_2_22 to Account Payable_Reviewing Purchase DP Invoice_3_14
    
    Processing BPMN Account Payable Level 3 - Reviewing Purchase Invoice.xml at Level 3...
    Parsed 6 elements and 13 flows from BPMN Account Payable Level 3 - Reviewing Purchase Invoice.xml
    Created node: Task with ID Account Payable_Reviewing Purchase Invoice_3_5 and name Receive Purchase Invoice
    Created node: Task with ID Account Payable_Reviewing Purchase Invoice_3_6 and name Return for Correction
    Created node: Task with ID Account Payable_Reviewing Purchase Invoice_3_16 and name Check Invoice Details
    Created node: Task with ID Account Payable_Reviewing Purchase Invoice_3_23 and name Approve Purchase Invoice
    Created node: StartEvent with ID Account Payable_Reviewing Purchase Invoice_3_4 and name Start
    Created node: EndEvent with ID Account Payable_Reviewing Purchase Invoice_3_14 and name End
    Warning: Gateway 19 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 20 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 28 has an unexpected number of incoming or outgoing flows.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Reviewing Purchase Invoice_3_9 and Account Payable_Reviewing Purchase Invoice_3_5.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Reviewing Purchase Invoice_3_8 and Account Payable_Reviewing Purchase Invoice_3_9.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Reviewing Purchase Invoice_3_12 and Account Payable_Reviewing Purchase Invoice_3_13.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Reviewing Purchase Invoice_3_16 and Account Payable_Reviewing Purchase Invoice_3_19.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Reviewing Purchase Invoice_3_19 and Account Payable_Reviewing Purchase Invoice_3_20.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Reviewing Purchase Invoice_3_19 and Account Payable_Reviewing Purchase Invoice_3_6.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Reviewing Purchase Invoice_3_20 and Account Payable_Reviewing Purchase Invoice_3_23.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Reviewing Purchase Invoice_3_23 and Account Payable_Reviewing Purchase Invoice_3_12.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Reviewing Purchase Invoice_3_28 and Account Payable_Reviewing Purchase Invoice_3_14.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Reviewing Purchase Invoice_3_6 and Account Payable_Reviewing Purchase Invoice_3_28.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Reviewing Purchase Invoice_3_23 and Account Payable_Reviewing Purchase Invoice_3_28.
    Created DECOMPOSED_INTO relationship from Account Payable_2_80 to Account Payable_Reviewing Purchase Invoice_3_4
    Created DECOMPOSED_INTO relationship from Account Payable_2_22 to Account Payable_Reviewing Purchase Invoice_3_14
    
    Processing BPMN Account Payable Level 3 - Updating Purchase DP Invoice Data.xml at Level 3...
    Parsed 4 elements and 7 flows from BPMN Account Payable Level 3 - Updating Purchase DP Invoice Data.xml
    Created node: Task with ID Account Payable_Updating Purchase DP Invoice Data_3_5 and name Review Purchase DP Invoice
    Created node: Task with ID Account Payable_Updating Purchase DP Invoice Data_3_7 and name Update DP Invoice Data
    Created node: StartEvent with ID Account Payable_Updating Purchase DP Invoice Data_3_4 and name Start
    Created node: EndEvent with ID Account Payable_Updating Purchase DP Invoice Data_3_15 and name End
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Updating Purchase DP Invoice Data_3_7 and Account Payable_Updating Purchase DP Invoice Data_3_13.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Updating Purchase DP Invoice Data_3_10 and Account Payable_Updating Purchase DP Invoice Data_3_5.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Updating Purchase DP Invoice Data_3_9 and Account Payable_Updating Purchase DP Invoice Data_3_10.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Updating Purchase DP Invoice Data_3_13 and Account Payable_Updating Purchase DP Invoice Data_3_14.
    Created DECOMPOSED_INTO relationship from Account Payable_2_80 to Account Payable_Updating Purchase DP Invoice Data_3_4
    Created DECOMPOSED_INTO relationship from Account Payable_2_22 to Account Payable_Updating Purchase DP Invoice Data_3_15
    
    Processing BPMN Account Payable Level 3 - Updating Purchase Invoice Data.xml at Level 3...
    Parsed 4 elements and 7 flows from BPMN Account Payable Level 3 - Updating Purchase Invoice Data.xml
    Created node: Task with ID Account Payable_Updating Purchase Invoice Data_3_5 and name Review Purchase Invoice
    Created node: Task with ID Account Payable_Updating Purchase Invoice Data_3_7 and name Update Invoice Data
    Created node: StartEvent with ID Account Payable_Updating Purchase Invoice Data_3_4 and name Start
    Created node: EndEvent with ID Account Payable_Updating Purchase Invoice Data_3_15 and name End
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Updating Purchase Invoice Data_3_7 and Account Payable_Updating Purchase Invoice Data_3_13.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Updating Purchase Invoice Data_3_10 and Account Payable_Updating Purchase Invoice Data_3_5.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Updating Purchase Invoice Data_3_9 and Account Payable_Updating Purchase Invoice Data_3_10.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Payable_Updating Purchase Invoice Data_3_13 and Account Payable_Updating Purchase Invoice Data_3_14.
    Created DECOMPOSED_INTO relationship from Account Payable_2_80 to Account Payable_Updating Purchase Invoice Data_3_4
    Created DECOMPOSED_INTO relationship from Account Payable_2_22 to Account Payable_Updating Purchase Invoice Data_3_15
    
    Processing BPMN Account Receivable Level 1.xml at Level 1...
    Parsed 3 elements and 2 flows from BPMN Account Receivable Level 1.xml
    Created node: Task with ID Account Receivable_1_5 and name Account Receivable
    Created node: StartEvent with ID Account Receivable_1_3 and name Start
    Created node: EndEvent with ID Account Receivable_1_7 and name End
    
    Processing BPMN Account Receivable Level 2.xml at Level 2...
    Parsed 23 elements and 39 flows from BPMN Account Receivable Level 2.xml
    Created node: Task with ID Account Receivable_2_24 and name Reviewing Sales Return
    Created node: Task with ID Account Receivable_2_25 and name Approving Sales Return
    Created node: Task with ID Account Receivable_2_26 and name Checking Sales Invoice Detail
    Created node: Task with ID Account Receivable_2_28 and name Updating Data Sales Invoice
    Created node: Task with ID Account Receivable_2_29 and name Finalizing Document
    Created node: Task with ID Account Receivable_2_30 and name Reviewing Sales Return
    Created node: Task with ID Account Receivable_2_31 and name Checking Sales Invoice Detail
    Created node: Task with ID Account Receivable_2_32 and name Approving Sales Return
    Created node: Task with ID Account Receivable_2_33 and name Updating Data Sales Invoice
    Created node: Task with ID Account Receivable_2_43 and name Completing Sales Invoice Form
    Created node: Task with ID Account Receivable_2_46 and name Receiving Sales Order
    Created node: Task with ID Account Receivable_2_48 and name Receiving Sales Order Down Payment
    Created node: Task with ID Account Receivable_2_50 and name Checking Goods for Return or Acceptance
    Created node: Task with ID Account Receivable_2_52 and name Checking Goods for Return or Acceptance
    Created node: Task with ID Account Receivable_2_57 and name Processing Sales Return
    Created node: Task with ID Account Receivable_2_64 and name Validating Invoice Details
    Created node: Task with ID Account Receivable_2_65 and name Completing Sales DP Invoice Form
    Created node: Task with ID Account Receivable_2_68 and name Processing Sales DP Return
    Created node: Task with ID Account Receivable_2_72 and name Creating the Sales DP Invoice
    Created node: Task with ID Account Receivable_2_76 and name Validating Invoice DP Details
    Created node: Task with ID Account Receivable_2_78 and name Creating the Sales Invoice
    Created node: StartEvent with ID Account Receivable_2_80 and name Start
    Created node: EndEvent with ID Account Receivable_2_22 and name End
    Warning: Gateway 12 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 15 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 16 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 18 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 42 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 45 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 55 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 60 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 67 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 73 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 81 has an unexpected number of incoming or outgoing flows.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_2_81 and Account Receivable_2_12.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_2_81 and Account Receivable_2_42.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_2_12 and Account Receivable_2_15.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_2_12 and Account Receivable_2_24.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_2_15 and Account Receivable_2_16.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_2_15 and Account Receivable_2_26.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_2_16 and Account Receivable_2_30.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_2_16 and Account Receivable_2_18.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_2_18 and Account Receivable_2_31.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_2_42 and Account Receivable_2_45.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_2_45 and Account Receivable_2_48.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_2_42 and Account Receivable_2_46.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_2_50 and Account Receivable_2_67.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_2_55 and Account Receivable_2_60.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_2_52 and Account Receivable_2_55.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_2_55 and Account Receivable_2_57.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_2_60 and Account Receivable_2_78.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_2_67 and Account Receivable_2_73.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_2_67 and Account Receivable_2_68.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_2_73 and Account Receivable_2_72.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_2_80 and Account Receivable_2_81.
    Created DECOMPOSED_INTO relationship from Account Receivable_1_3 to Account Receivable_2_80
    Created DECOMPOSED_INTO relationship from Account Receivable_1_7 to Account Receivable_2_22
    
    Processing BPMN Account Receivable Level 3 - Creating the Sales DP Invoice.xml at Level 3...
    Parsed 8 elements and 17 flows from BPMN Account Receivable Level 3 - Creating the Sales DP Invoice.xml
    Created node: Task with ID Account Receivable_Creating the Sales DP Invoice_3_5 and name Receive Sales Order Down Payment
    Created node: Task with ID Account Receivable_Creating the Sales DP Invoice_3_6 and name Check Goods for Acceptance
    Created node: Task with ID Account Receivable_Creating the Sales DP Invoice_3_14 and name Process Sales DP Return
    Created node: Task with ID Account Receivable_Creating the Sales DP Invoice_3_16 and name Create Sales DP Invoice
    Created node: Task with ID Account Receivable_Creating the Sales DP Invoice_3_29 and name Validate Invoice DP Details
    Created node: Task with ID Account Receivable_Creating the Sales DP Invoice_3_30 and name Complete Sales DP Invoice Form
    Created node: StartEvent with ID Account Receivable_Creating the Sales DP Invoice_3_4 and name Start
    Created node: EndEvent with ID Account Receivable_Creating the Sales DP Invoice_3_35 and name End
    Warning: Gateway 11 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 17 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 36 has an unexpected number of incoming or outgoing flows.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Creating the Sales DP Invoice_3_11 and Account Receivable_Creating the Sales DP Invoice_3_14.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Creating the Sales DP Invoice_3_11 and Account Receivable_Creating the Sales DP Invoice_3_17.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Creating the Sales DP Invoice_3_6 and Account Receivable_Creating the Sales DP Invoice_3_11.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Creating the Sales DP Invoice_3_14 and Account Receivable_Creating the Sales DP Invoice_3_36.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Creating the Sales DP Invoice_3_17 and Account Receivable_Creating the Sales DP Invoice_3_16.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Creating the Sales DP Invoice_3_22 and Account Receivable_Creating the Sales DP Invoice_3_5.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Creating the Sales DP Invoice_3_21 and Account Receivable_Creating the Sales DP Invoice_3_22.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Creating the Sales DP Invoice_3_26 and Account Receivable_Creating the Sales DP Invoice_3_6.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Creating the Sales DP Invoice_3_24 and Account Receivable_Creating the Sales DP Invoice_3_26.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Creating the Sales DP Invoice_3_32 and Account Receivable_Creating the Sales DP Invoice_3_34.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Creating the Sales DP Invoice_3_30 and Account Receivable_Creating the Sales DP Invoice_3_32.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Creating the Sales DP Invoice_3_30 and Account Receivable_Creating the Sales DP Invoice_3_36.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Creating the Sales DP Invoice_3_36 and Account Receivable_Creating the Sales DP Invoice_3_35.
    Created DECOMPOSED_INTO relationship from Account Receivable_2_80 to Account Receivable_Creating the Sales DP Invoice_3_4
    Created DECOMPOSED_INTO relationship from Account Receivable_2_22 to Account Receivable_Creating the Sales DP Invoice_3_35
    
    Processing BPMN Account Receivable Level 3 - Creating the Sales Invoice.xml at Level 3...
    Parsed 8 elements and 17 flows from BPMN Account Receivable Level 3 - Creating the Sales Invoice.xml
    Created node: Task with ID Account Receivable_Creating the Sales Invoice_3_5 and name Receive Sales Order
    Created node: Task with ID Account Receivable_Creating the Sales Invoice_3_6 and name Check Goods for Acceptance
    Created node: Task with ID Account Receivable_Creating the Sales Invoice_3_14 and name Process Sales Return
    Created node: Task with ID Account Receivable_Creating the Sales Invoice_3_16 and name Create Sales Invoice
    Created node: Task with ID Account Receivable_Creating the Sales Invoice_3_29 and name Validate Invoice Details
    Created node: Task with ID Account Receivable_Creating the Sales Invoice_3_30 and name Complete Sales Invoice Form
    Created node: StartEvent with ID Account Receivable_Creating the Sales Invoice_3_4 and name Start
    Created node: EndEvent with ID Account Receivable_Creating the Sales Invoice_3_35 and name End
    Warning: Gateway 11 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 17 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 36 has an unexpected number of incoming or outgoing flows.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Creating the Sales Invoice_3_11 and Account Receivable_Creating the Sales Invoice_3_14.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Creating the Sales Invoice_3_11 and Account Receivable_Creating the Sales Invoice_3_17.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Creating the Sales Invoice_3_6 and Account Receivable_Creating the Sales Invoice_3_11.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Creating the Sales Invoice_3_14 and Account Receivable_Creating the Sales Invoice_3_36.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Creating the Sales Invoice_3_17 and Account Receivable_Creating the Sales Invoice_3_16.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Creating the Sales Invoice_3_22 and Account Receivable_Creating the Sales Invoice_3_5.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Creating the Sales Invoice_3_21 and Account Receivable_Creating the Sales Invoice_3_22.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Creating the Sales Invoice_3_26 and Account Receivable_Creating the Sales Invoice_3_6.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Creating the Sales Invoice_3_24 and Account Receivable_Creating the Sales Invoice_3_26.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Creating the Sales Invoice_3_32 and Account Receivable_Creating the Sales Invoice_3_34.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Creating the Sales Invoice_3_30 and Account Receivable_Creating the Sales Invoice_3_32.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Creating the Sales Invoice_3_30 and Account Receivable_Creating the Sales Invoice_3_36.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Creating the Sales Invoice_3_36 and Account Receivable_Creating the Sales Invoice_3_35.
    Created DECOMPOSED_INTO relationship from Account Receivable_2_80 to Account Receivable_Creating the Sales Invoice_3_4
    Created DECOMPOSED_INTO relationship from Account Receivable_2_22 to Account Receivable_Creating the Sales Invoice_3_35
    
    Processing BPMN Account Receivable Level 3 - Finalizing Sales DP Invoice Document.xml at Level 3...
    Parsed 5 elements and 8 flows from BPMN Account Receivable Level 3 - Finalizing Sales DP Invoice Document.xml
    Created node: Task with ID Account Receivable_Finalizing Sales DP Invoice Document_3_5 and name Retrieve DP Invoice Data
    Created node: Task with ID Account Receivable_Finalizing Sales DP Invoice Document_3_7 and name Verify Data Accuracy
    Created node: Task with ID Account Receivable_Finalizing Sales DP Invoice Document_3_17 and name Complete DP Invoice Document
    Created node: StartEvent with ID Account Receivable_Finalizing Sales DP Invoice Document_3_4 and name Start
    Created node: EndEvent with ID Account Receivable_Finalizing Sales DP Invoice Document_3_15 and name End
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Finalizing Sales DP Invoice Document_3_10 and Account Receivable_Finalizing Sales DP Invoice Document_3_5.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Finalizing Sales DP Invoice Document_3_9 and Account Receivable_Finalizing Sales DP Invoice Document_3_10.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Finalizing Sales DP Invoice Document_3_13 and Account Receivable_Finalizing Sales DP Invoice Document_3_14.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Finalizing Sales DP Invoice Document_3_17 and Account Receivable_Finalizing Sales DP Invoice Document_3_13.
    Created DECOMPOSED_INTO relationship from Account Receivable_2_80 to Account Receivable_Finalizing Sales DP Invoice Document_3_4
    Created DECOMPOSED_INTO relationship from Account Receivable_2_22 to Account Receivable_Finalizing Sales DP Invoice Document_3_15
    
    Processing BPMN Account Receivable Level 3 - Finalizing Sales Invoice Document.xml at Level 3...
    Parsed 5 elements and 8 flows from BPMN Account Receivable Level 3 - Finalizing Sales Invoice Document.xml
    Created node: Task with ID Account Receivable_Finalizing Sales Invoice Document_3_5 and name Retrieve  Invoice Data
    Created node: Task with ID Account Receivable_Finalizing Sales Invoice Document_3_7 and name Verify Data Accuracy
    Created node: Task with ID Account Receivable_Finalizing Sales Invoice Document_3_17 and name Complete  Invoice Document
    Created node: StartEvent with ID Account Receivable_Finalizing Sales Invoice Document_3_4 and name Start
    Created node: EndEvent with ID Account Receivable_Finalizing Sales Invoice Document_3_15 and name End
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Finalizing Sales Invoice Document_3_10 and Account Receivable_Finalizing Sales Invoice Document_3_5.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Finalizing Sales Invoice Document_3_9 and Account Receivable_Finalizing Sales Invoice Document_3_10.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Finalizing Sales Invoice Document_3_13 and Account Receivable_Finalizing Sales Invoice Document_3_14.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Finalizing Sales Invoice Document_3_17 and Account Receivable_Finalizing Sales Invoice Document_3_13.
    Created DECOMPOSED_INTO relationship from Account Receivable_2_80 to Account Receivable_Finalizing Sales Invoice Document_3_4
    Created DECOMPOSED_INTO relationship from Account Receivable_2_22 to Account Receivable_Finalizing Sales Invoice Document_3_15
    
    Processing BPMN Account Receivable Level 3 - Reviewing Sales DP Invoice.xml at Level 3...
    Parsed 6 elements and 13 flows from BPMN Account Receivable Level 3 - Reviewing Sales DP Invoice.xml
    Created node: Task with ID Account Receivable_Reviewing Sales DP Invoice_3_5 and name Receive Sales DP Invoice
    Created node: Task with ID Account Receivable_Reviewing Sales DP Invoice_3_6 and name Return for Correction
    Created node: Task with ID Account Receivable_Reviewing Sales DP Invoice_3_16 and name Check Invoice Details
    Created node: Task with ID Account Receivable_Reviewing Sales DP Invoice_3_23 and name Approve Sales DP Invoice
    Created node: StartEvent with ID Account Receivable_Reviewing Sales DP Invoice_3_4 and name Start
    Created node: EndEvent with ID Account Receivable_Reviewing Sales DP Invoice_3_14 and name End
    Warning: Gateway 19 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 20 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 28 has an unexpected number of incoming or outgoing flows.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Reviewing Sales DP Invoice_3_9 and Account Receivable_Reviewing Sales DP Invoice_3_5.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Reviewing Sales DP Invoice_3_8 and Account Receivable_Reviewing Sales DP Invoice_3_9.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Reviewing Sales DP Invoice_3_12 and Account Receivable_Reviewing Sales DP Invoice_3_13.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Reviewing Sales DP Invoice_3_16 and Account Receivable_Reviewing Sales DP Invoice_3_19.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Reviewing Sales DP Invoice_3_19 and Account Receivable_Reviewing Sales DP Invoice_3_20.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Reviewing Sales DP Invoice_3_19 and Account Receivable_Reviewing Sales DP Invoice_3_6.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Reviewing Sales DP Invoice_3_20 and Account Receivable_Reviewing Sales DP Invoice_3_23.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Reviewing Sales DP Invoice_3_23 and Account Receivable_Reviewing Sales DP Invoice_3_12.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Reviewing Sales DP Invoice_3_28 and Account Receivable_Reviewing Sales DP Invoice_3_14.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Reviewing Sales DP Invoice_3_6 and Account Receivable_Reviewing Sales DP Invoice_3_28.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Reviewing Sales DP Invoice_3_23 and Account Receivable_Reviewing Sales DP Invoice_3_28.
    Created DECOMPOSED_INTO relationship from Account Receivable_2_80 to Account Receivable_Reviewing Sales DP Invoice_3_4
    Created DECOMPOSED_INTO relationship from Account Receivable_2_22 to Account Receivable_Reviewing Sales DP Invoice_3_14
    
    Processing BPMN Account Receivable Level 3 - Reviewing Sales Invoice.xml at Level 3...
    Parsed 6 elements and 13 flows from BPMN Account Receivable Level 3 - Reviewing Sales Invoice.xml
    Created node: Task with ID Account Receivable_Reviewing Sales Invoice_3_5 and name Receive Sales  Invoice
    Created node: Task with ID Account Receivable_Reviewing Sales Invoice_3_6 and name Return for Correction
    Created node: Task with ID Account Receivable_Reviewing Sales Invoice_3_16 and name Check Invoice Details
    Created node: Task with ID Account Receivable_Reviewing Sales Invoice_3_23 and name Approve Sales  Invoice
    Created node: StartEvent with ID Account Receivable_Reviewing Sales Invoice_3_4 and name Start
    Created node: EndEvent with ID Account Receivable_Reviewing Sales Invoice_3_14 and name End
    Warning: Gateway 19 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 20 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 28 has an unexpected number of incoming or outgoing flows.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Reviewing Sales Invoice_3_9 and Account Receivable_Reviewing Sales Invoice_3_5.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Reviewing Sales Invoice_3_8 and Account Receivable_Reviewing Sales Invoice_3_9.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Reviewing Sales Invoice_3_12 and Account Receivable_Reviewing Sales Invoice_3_13.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Reviewing Sales Invoice_3_16 and Account Receivable_Reviewing Sales Invoice_3_19.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Reviewing Sales Invoice_3_19 and Account Receivable_Reviewing Sales Invoice_3_20.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Reviewing Sales Invoice_3_19 and Account Receivable_Reviewing Sales Invoice_3_6.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Reviewing Sales Invoice_3_20 and Account Receivable_Reviewing Sales Invoice_3_23.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Reviewing Sales Invoice_3_23 and Account Receivable_Reviewing Sales Invoice_3_12.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Reviewing Sales Invoice_3_28 and Account Receivable_Reviewing Sales Invoice_3_14.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Reviewing Sales Invoice_3_6 and Account Receivable_Reviewing Sales Invoice_3_28.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Reviewing Sales Invoice_3_23 and Account Receivable_Reviewing Sales Invoice_3_28.
    Created DECOMPOSED_INTO relationship from Account Receivable_2_80 to Account Receivable_Reviewing Sales Invoice_3_4
    Created DECOMPOSED_INTO relationship from Account Receivable_2_22 to Account Receivable_Reviewing Sales Invoice_3_14
    
    Processing BPMN Account Receivable Level 3 - Updating Sales DP Invoice Data.xml at Level 3...
    Parsed 4 elements and 7 flows from BPMN Account Receivable Level 3 - Updating Sales DP Invoice Data.xml
    Created node: Task with ID Account Receivable_Updating Sales DP Invoice Data_3_5 and name Review Sales DP Invoice
    Created node: Task with ID Account Receivable_Updating Sales DP Invoice Data_3_7 and name Update DP Invoice Data
    Created node: StartEvent with ID Account Receivable_Updating Sales DP Invoice Data_3_4 and name Start
    Created node: EndEvent with ID Account Receivable_Updating Sales DP Invoice Data_3_15 and name End
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Updating Sales DP Invoice Data_3_7 and Account Receivable_Updating Sales DP Invoice Data_3_13.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Updating Sales DP Invoice Data_3_10 and Account Receivable_Updating Sales DP Invoice Data_3_5.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Updating Sales DP Invoice Data_3_9 and Account Receivable_Updating Sales DP Invoice Data_3_10.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Updating Sales DP Invoice Data_3_13 and Account Receivable_Updating Sales DP Invoice Data_3_14.
    Created DECOMPOSED_INTO relationship from Account Receivable_2_80 to Account Receivable_Updating Sales DP Invoice Data_3_4
    Created DECOMPOSED_INTO relationship from Account Receivable_2_22 to Account Receivable_Updating Sales DP Invoice Data_3_15
    
    Processing BPMN Account Receivable Level 3 - Updating Sales Invoice Data.xml at Level 3...
    Parsed 4 elements and 7 flows from BPMN Account Receivable Level 3 - Updating Sales Invoice Data.xml
    Created node: Task with ID Account Receivable_Updating Sales Invoice Data_3_5 and name Review Sales Invoice
    Created node: Task with ID Account Receivable_Updating Sales Invoice Data_3_7 and name Update Invoice Data
    Created node: StartEvent with ID Account Receivable_Updating Sales Invoice Data_3_4 and name Start
    Created node: EndEvent with ID Account Receivable_Updating Sales Invoice Data_3_15 and name End
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Updating Sales Invoice Data_3_7 and Account Receivable_Updating Sales Invoice Data_3_13.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Updating Sales Invoice Data_3_10 and Account Receivable_Updating Sales Invoice Data_3_5.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Updating Sales Invoice Data_3_9 and Account Receivable_Updating Sales Invoice Data_3_10.
    Warning: Could not create relationship SEQUENCE_FLOW between Account Receivable_Updating Sales Invoice Data_3_13 and Account Receivable_Updating Sales Invoice Data_3_14.
    Created DECOMPOSED_INTO relationship from Account Receivable_2_80 to Account Receivable_Updating Sales Invoice Data_3_4
    Created DECOMPOSED_INTO relationship from Account Receivable_2_22 to Account Receivable_Updating Sales Invoice Data_3_15
    
    Processing BPMN Asset Management Level 1.xml at Level 1...
    Parsed 3 elements and 2 flows from BPMN Asset Management Level 1.xml
    Created node: Task with ID Asset Management_1_4 and name Asset Management
    Created node: StartEvent with ID Asset Management_1_3 and name St
    Created node: EndEvent with ID Asset Management_1_6 and name End
    
    Processing BPMN Asset Management Level 2.xml at Level 2...
    Parsed 28 elements and 55 flows from BPMN Asset Management Level 2.xml
    Created node: Task with ID Asset Management_2_34 and name Approving AM Registration
    Created node: Task with ID Asset Management_2_35 and name Approving AM Transfer
    Created node: Task with ID Asset Management_2_36 and name Approving AM Maintenance
    Created node: Task with ID Asset Management_2_37 and name Approving AM Stock Take
    Created node: Task with ID Asset Management_2_38 and name Approving AM Revaluation
    Created node: Task with ID Asset Management_2_39 and name Approving AM Disposal
    Created node: Task with ID Asset Management_2_41 and name updating Approval Status
    Created node: Task with ID Asset Management_2_60 and name Selecting AM Transfer
    Created node: Task with ID Asset Management_2_64 and name Completing AM Transfer Form
    Created node: Task with ID Asset Management_2_69 and name Receiving AM Procument Registration
    Created node: Task with ID Asset Management_2_72 and name Receiving Request to Transfer Asset
    Created node: Task with ID Asset Management_2_73 and name Receiving Request to Maintenance Asset
    Created node: Task with ID Asset Management_2_74 and name Receiving Request to Stock Take Asset
    Created node: Task with ID Asset Management_2_75 and name Receiving Request to Revaluate Asset
    Created node: Task with ID Asset Management_2_76 and name Receiving Request to Dispose Asset
    Created node: Task with ID Asset Management_2_79 and name Looking up List Asset
    Created node: Task with ID Asset Management_2_81 and name Looking up List Asset
    Created node: Task with ID Asset Management_2_83 and name Looking up List Asset
    Created node: Task with ID Asset Management_2_85 and name Looking up List Asset
    Created node: Task with ID Asset Management_2_89 and name Looking up List Asset
    Created node: Task with ID Asset Management_2_91 and name Completing AM Revaluation Form
    Created node: Task with ID Asset Management_2_92 and name Completing AM Maintenance Form
    Created node: Task with ID Asset Management_2_93 and name Completing AM Disposal Form
    Created node: Task with ID Asset Management_2_95 and name Completing AM Registration Form
    Created node: Task with ID Asset Management_2_99 and name Adding New AM  Category
    Created node: Task with ID Asset Management_2_102 and name Adding New AM  Fiscal Type
    Created node: StartEvent with ID Asset Management_2_2 and name Start
    Created node: EndEvent with ID Asset Management_2_48 and name End
    Warning: Gateway 4 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 19 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 22 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 23 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 27 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 30 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 32 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 51 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 54 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 56 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 58 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 62 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 66 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 97 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 100 has an unexpected number of incoming or outgoing flows.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_2_4 and Asset Management_2_51.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_2_2 and Asset Management_2_4.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_2_4 and Asset Management_2_19.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_2_19 and Asset Management_2_22.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_2_19 and Asset Management_2_34.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_2_22 and Asset Management_2_23.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_2_22 and Asset Management_2_35.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_2_23 and Asset Management_2_36.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_2_27 and Asset Management_2_30.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_2_23 and Asset Management_2_27.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_2_27 and Asset Management_2_37.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_2_32 and Asset Management_2_39.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_2_30 and Asset Management_2_32.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_2_30 and Asset Management_2_38.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_2_54 and Asset Management_2_56.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_2_54 and Asset Management_2_74.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_2_56 and Asset Management_2_75.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_2_58 and Asset Management_2_76.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_2_56 and Asset Management_2_58.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_2_62 and Asset Management_2_72.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_2_51 and Asset Management_2_62.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_2_66 and Asset Management_2_73.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_2_62 and Asset Management_2_66.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_2_66 and Asset Management_2_54.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_2_51 and Asset Management_2_69.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_2_97 and Asset Management_2_99.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_2_58 and Asset Management_2_97.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_2_97 and Asset Management_2_100.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_2_100 and Asset Management_2_102.
    Created DECOMPOSED_INTO relationship from Asset Management_1_6 to Asset Management_2_48
    
    Processing BPMN Asset Management Level 3 - Asset Category Process.xml at Level 3...
    Parsed 10 elements and 25 flows from BPMN Asset Management Level 3 - Asset Category Process.xml
    Created node: Task with ID Asset Management_Asset Category Process_3_5 and name Open Asset Category Form
    Created node: Task with ID Asset Management_Asset Category Process_3_7 and name Fill Asset Category  Form
    Created node: Task with ID Asset Management_Asset Category Process_3_10 and name Save Asset Category Data
    Created node: Task with ID Asset Management_Asset Category Process_3_18 and name Open Detail Asset  Category Data
    Created node: Task with ID Asset Management_Asset Category Process_3_30 and name Open Update Asset Category Data
    Created node: Task with ID Asset Management_Asset Category Process_3_33 and name Edit CurrentAsset Category Data
    Created node: Task with ID Asset Management_Asset Category Process_3_37 and name ReviewAsset  Category Data
    Created node: Task with ID Asset Management_Asset Category Process_3_40 and name Open  Delete Asset Category Data
    Created node: StartEvent with ID Asset Management_Asset Category Process_3_4 and name Start
    Created node: EndEvent with ID Asset Management_Asset Category Process_3_11 and name End
    Warning: Gateway 15 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 20 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 23 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 45 has an unexpected number of incoming or outgoing flows.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Category Process_3_12 and Asset Management_Asset Category Process_3_40.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Category Process_3_10 and Asset Management_Asset Category Process_3_45.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Category Process_3_10 and Asset Management_Asset Category Process_3_13.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Category Process_3_4 and Asset Management_Asset Category Process_3_15.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Category Process_3_15 and Asset Management_Asset Category Process_3_5.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Category Process_3_15 and Asset Management_Asset Category Process_3_18.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Category Process_3_18 and Asset Management_Asset Category Process_3_20.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Category Process_3_23 and Asset Management_Asset Category Process_3_45.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Category Process_3_20 and Asset Management_Asset Category Process_3_23.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Category Process_3_37 and Asset Management_Asset Category Process_3_23.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Category Process_3_20 and Asset Management_Asset Category Process_3_37.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Category Process_3_12 and Asset Management_Asset Category Process_3_27.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Category Process_3_27 and Asset Management_Asset Category Process_3_37.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Category Process_3_15 and Asset Management_Asset Category Process_3_30.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Category Process_3_33 and Asset Management_Asset Category Process_3_45.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Category Process_3_35 and Asset Management_Asset Category Process_3_33.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Category Process_3_40 and Asset Management_Asset Category Process_3_45.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Category Process_3_15 and Asset Management_Asset Category Process_3_40.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Category Process_3_43 and Asset Management_Asset Category Process_3_12.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Category Process_3_40 and Asset Management_Asset Category Process_3_43.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Category Process_3_35 and Asset Management_Asset Category Process_3_12.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Category Process_3_13 and Asset Management_Asset Category Process_3_12.
    Created DECOMPOSED_INTO relationship from Asset Management_2_2 to Asset Management_Asset Category Process_3_4
    Created DECOMPOSED_INTO relationship from Asset Management_2_48 to Asset Management_Asset Category Process_3_11
    
    Processing BPMN Asset Management Level 3 - Asset Management Registration Process.xml at Level 3...
    Parsed 9 elements and 27 flows from BPMN Asset Management Level 3 - Asset Management Registration Process.xml
    Created node: Task with ID Asset Management_Asset Management Registration Process_3_5 and name Open Asset Additional  Form
    Created node: Task with ID Asset Management_Asset Management Registration Process_3_11 and name Fill Asset Registration Form
    Created node: Task with ID Asset Management_Asset Management Registration Process_3_19 and name Save Asset RegistrationData
    Created node: Task with ID Asset Management_Asset Management Registration Process_3_28 and name Open Detail Asset  Data
    Created node: Task with ID Asset Management_Asset Management Registration Process_3_41 and name Open Update Asset Data
    Created node: Task with ID Asset Management_Asset Management Registration Process_3_44 and name Edit CurrentAsset Data
    Created node: Task with ID Asset Management_Asset Management Registration Process_3_50 and name ReviewAsset  Data
    Created node: StartEvent with ID Asset Management_Asset Management Registration Process_3_4 and name Start
    Created node: EndEvent with ID Asset Management_Asset Management Registration Process_3_20 and name End
    Warning: Gateway 25 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 30 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 33 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 37 has an unexpected number of incoming or outgoing flows.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Management Registration Process_3_53 and Asset Management_Asset Management Registration Process_3_44.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Management Registration Process_3_7 and Asset Management_Asset Management Registration Process_3_44.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Management Registration Process_3_7 and Asset Management_Asset Management Registration Process_3_11.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Management Registration Process_3_14 and Asset Management_Asset Management Registration Process_3_11.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Management Registration Process_3_19 and Asset Management_Asset Management Registration Process_3_37.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Management Registration Process_3_23 and Asset Management_Asset Management Registration Process_3_21.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Management Registration Process_3_19 and Asset Management_Asset Management Registration Process_3_23.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Management Registration Process_3_4 and Asset Management_Asset Management Registration Process_3_25.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Management Registration Process_3_25 and Asset Management_Asset Management Registration Process_3_5.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Management Registration Process_3_25 and Asset Management_Asset Management Registration Process_3_28.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Management Registration Process_3_28 and Asset Management_Asset Management Registration Process_3_30.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Management Registration Process_3_33 and Asset Management_Asset Management Registration Process_3_37.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Management Registration Process_3_30 and Asset Management_Asset Management Registration Process_3_33.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Management Registration Process_3_50 and Asset Management_Asset Management Registration Process_3_33.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Management Registration Process_3_30 and Asset Management_Asset Management Registration Process_3_50.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Management Registration Process_3_21 and Asset Management_Asset Management Registration Process_3_38.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Management Registration Process_3_38 and Asset Management_Asset Management Registration Process_3_50.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Management Registration Process_3_25 and Asset Management_Asset Management Registration Process_3_41.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Management Registration Process_3_44 and Asset Management_Asset Management Registration Process_3_37.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Management Registration Process_3_47 and Asset Management_Asset Management Registration Process_3_21.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Management Registration Process_3_37 and Asset Management_Asset Management Registration Process_3_20.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Management Registration Process_3_14 and Asset Management_Asset Management Registration Process_3_44.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Management Registration Process_3_47 and Asset Management_Asset Management Registration Process_3_44.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Management Registration Process_3_53 and Asset Management_Asset Management Registration Process_3_11.
    Created DECOMPOSED_INTO relationship from Asset Management_2_2 to Asset Management_Asset Management Registration Process_3_4
    Created DECOMPOSED_INTO relationship from Asset Management_2_48 to Asset Management_Asset Management Registration Process_3_20
    
    Processing BPMN Asset Management Level 3 - Asset Transfer.xml at Level 3...
    Parsed 9 elements and 22 flows from BPMN Asset Management Level 3 - Asset Transfer.xml
    Created node: Task with ID Asset Management_Asset Transfer_3_4 and name Open Asset Transfer Form
    Created node: Task with ID Asset Management_Asset Transfer_3_6 and name Fill Asset Transfer Form
    Created node: Task with ID Asset Management_Asset Transfer_3_9 and name Save Asset Transfer Data
    Created node: Task with ID Asset Management_Asset Transfer_3_18 and name Open Detail Asset Transfer Data
    Created node: Task with ID Asset Management_Asset Transfer_3_31 and name Open Update Asset Transfer Data
    Created node: Task with ID Asset Management_Asset Transfer_3_34 and name Edit CurrentAsset TransferData
    Created node: Task with ID Asset Management_Asset Transfer_3_38 and name ReviewAsset Transfer  Data
    Created node: StartEvent with ID Asset Management_Asset Transfer_3_3 and name Start
    Created node: EndEvent with ID Asset Management_Asset Transfer_3_10 and name Edit
    Warning: Gateway 15 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 20 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 23 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 27 has an unexpected number of incoming or outgoing flows.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Transfer_3_9 and Asset Management_Asset Transfer_3_27.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Transfer_3_13 and Asset Management_Asset Transfer_3_11.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Transfer_3_9 and Asset Management_Asset Transfer_3_13.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Transfer_3_3 and Asset Management_Asset Transfer_3_15.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Transfer_3_15 and Asset Management_Asset Transfer_3_4.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Transfer_3_15 and Asset Management_Asset Transfer_3_18.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Transfer_3_18 and Asset Management_Asset Transfer_3_20.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Transfer_3_23 and Asset Management_Asset Transfer_3_27.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Transfer_3_20 and Asset Management_Asset Transfer_3_23.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Transfer_3_38 and Asset Management_Asset Transfer_3_23.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Transfer_3_20 and Asset Management_Asset Transfer_3_38.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Transfer_3_11 and Asset Management_Asset Transfer_3_28.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Transfer_3_28 and Asset Management_Asset Transfer_3_38.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Transfer_3_15 and Asset Management_Asset Transfer_3_31.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Transfer_3_34 and Asset Management_Asset Transfer_3_27.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Transfer_3_27 and Asset Management_Asset Transfer_3_10.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Transfer_3_36 and Asset Management_Asset Transfer_3_34.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Transfer_3_36 and Asset Management_Asset Transfer_3_11.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Asset Transfer_3_42 and Asset Management_Asset Transfer_3_6.
    Created DECOMPOSED_INTO relationship from Asset Management_2_2 to Asset Management_Asset Transfer_3_3
    
    Processing BPMN Asset Management Level 3 - Fiscal Type.xml at Level 3...
    Parsed 10 elements and 25 flows from BPMN Asset Management Level 3 - Fiscal Type.xml
    Created node: Task with ID Asset Management_Fiscal Type_3_4 and name Open Fiscal Type Form
    Created node: Task with ID Asset Management_Fiscal Type_3_6 and name Fill Fiscal Type  Form
    Created node: Task with ID Asset Management_Fiscal Type_3_9 and name Save Fiscal Type Data
    Created node: Task with ID Asset Management_Fiscal Type_3_18 and name Open Detail Fiscal Type Data
    Created node: Task with ID Asset Management_Fiscal Type_3_31 and name Open Update Fiscal Type Data
    Created node: Task with ID Asset Management_Fiscal Type_3_34 and name Edit CurrentFiscal TypeData
    Created node: Task with ID Asset Management_Fiscal Type_3_38 and name ReviewFiscal Type  Data
    Created node: Task with ID Asset Management_Fiscal Type_3_40 and name Open  Delete Fiscal Type Data
    Created node: StartEvent with ID Asset Management_Fiscal Type_3_3 and name Start
    Created node: EndEvent with ID Asset Management_Fiscal Type_3_10 and name End
    Warning: Gateway 15 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 20 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 23 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 27 has an unexpected number of incoming or outgoing flows.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Fiscal Type_3_9 and Asset Management_Fiscal Type_3_27.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Fiscal Type_3_13 and Asset Management_Fiscal Type_3_11.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Fiscal Type_3_9 and Asset Management_Fiscal Type_3_13.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Fiscal Type_3_3 and Asset Management_Fiscal Type_3_15.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Fiscal Type_3_15 and Asset Management_Fiscal Type_3_4.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Fiscal Type_3_15 and Asset Management_Fiscal Type_3_18.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Fiscal Type_3_18 and Asset Management_Fiscal Type_3_20.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Fiscal Type_3_23 and Asset Management_Fiscal Type_3_27.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Fiscal Type_3_20 and Asset Management_Fiscal Type_3_23.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Fiscal Type_3_38 and Asset Management_Fiscal Type_3_23.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Fiscal Type_3_20 and Asset Management_Fiscal Type_3_38.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Fiscal Type_3_11 and Asset Management_Fiscal Type_3_28.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Fiscal Type_3_28 and Asset Management_Fiscal Type_3_38.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Fiscal Type_3_15 and Asset Management_Fiscal Type_3_31.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Fiscal Type_3_34 and Asset Management_Fiscal Type_3_27.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Fiscal Type_3_27 and Asset Management_Fiscal Type_3_10.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Fiscal Type_3_36 and Asset Management_Fiscal Type_3_34.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Fiscal Type_3_15 and Asset Management_Fiscal Type_3_40.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Fiscal Type_3_43 and Asset Management_Fiscal Type_3_11.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Fiscal Type_3_40 and Asset Management_Fiscal Type_3_43.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Fiscal Type_3_40 and Asset Management_Fiscal Type_3_27.
    Warning: Could not create relationship SEQUENCE_FLOW between Asset Management_Fiscal Type_3_36 and Asset Management_Fiscal Type_3_11.
    Created DECOMPOSED_INTO relationship from Asset Management_2_2 to Asset Management_Fiscal Type_3_3
    Created DECOMPOSED_INTO relationship from Asset Management_2_48 to Asset Management_Fiscal Type_3_10
    
    Processing BPMN Cash Bank Level 1.xml at Level 1...
    Parsed 3 elements and 2 flows from BPMN Cash Bank Level 1.xml
    Created node: Task with ID Cash Bank_1_5 and name Cash Bank
    Created node: StartEvent with ID Cash Bank_1_3 and name Start
    Created node: EndEvent with ID Cash Bank_1_7 and name End
    
    Processing BPMN Cash Bank Level 2.xml at Level 2...
    Parsed 35 elements and 63 flows from BPMN Cash Bank Level 2.xml
    Created node: Task with ID Cash Bank_2_22 and name Updating Currency Records
    Created node: Task with ID Cash Bank_2_24 and name Recording to Journal
    Created node: Task with ID Cash Bank_2_32 and name Approving Cash Account
    Created node: Task with ID Cash Bank_2_34 and name Approving Cash Receipt
    Created node: Task with ID Cash Bank_2_35 and name Approving Cash Disbursement
    Created node: Task with ID Cash Bank_2_37 and name Updating Cash Records
    Created node: Task with ID Cash Bank_2_45 and name Approving Bank Account
    Created node: Task with ID Cash Bank_2_47 and name Approving Bank Receipt
    Created node: Task with ID Cash Bank_2_48 and name Approving Bank Disbursement
    Created node: Task with ID Cash Bank_2_49 and name Updating Bank Records
    Created node: Task with ID Cash Bank_2_56 and name Manage Bank Management
    Created node: Task with ID Cash Bank_2_63 and name Manage Cash Management
    Created node: Task with ID Cash Bank_2_71 and name Viewing Currency
    Created node: Task with ID Cash Bank_2_72 and name Managing Cash Account
    Created node: Task with ID Cash Bank_2_74 and name Validating Cash Receipt Detail
    Created node: Task with ID Cash Bank_2_75 and name Completing Cash Receipt Form
    Created node: Task with ID Cash Bank_2_77 and name Validating Cash Disbursement Detail
    Created node: Task with ID Cash Bank_2_78 and name Completing Cash Disbursement Form
    Created node: Task with ID Cash Bank_2_80 and name Managing Currency Exchange
    Created node: Task with ID Cash Bank_2_82 and name Completing Cash Account Form
    Created node: Task with ID Cash Bank_2_84 and name Viewing Cash Management
    Created node: Task with ID Cash Bank_2_91 and name Receiving Cash Receipt
    Created node: Task with ID Cash Bank_2_95 and name Disbursing Cash Payment
    Created node: Task with ID Cash Bank_2_98 and name Managing Bank Account
    Created node: Task with ID Cash Bank_2_100 and name Validating Bank Receipt Detail
    Created node: Task with ID Cash Bank_2_101 and name Completing Bank Receipt Form
    Created node: Task with ID Cash Bank_2_103 and name Validating Bank Disbursement Detail
    Created node: Task with ID Cash Bank_2_104 and name Completing Bank Disbursement Form
    Created node: Task with ID Cash Bank_2_106 and name Completing Bank Account Form
    Created node: Task with ID Cash Bank_2_108 and name Viewing Bank Management
    Created node: Task with ID Cash Bank_2_114 and name Receiving BankReceipt
    Created node: Task with ID Cash Bank_2_118 and name Disbursing Bank Payment
    Created node: Task with ID Cash Bank_2_129 and name Approving Currency Exchange
    Created node: StartEvent with ID Cash Bank_2_125 and name Start
    Created node: EndEvent with ID Cash Bank_2_19 and name End
    Warning: Gateway 13 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 16 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 18 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 23 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 27 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 30 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 40 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 43 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 51 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 53 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 58 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 60 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 69 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 70 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 86 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 88 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 90 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 109 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 111 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 113 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 122 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 126 has an unexpected number of incoming or outgoing flows.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_126 and Cash Bank_2_13.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_126 and Cash Bank_2_69.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_13 and Cash Bank_2_16.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_16 and Cash Bank_2_18.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_18 and Cash Bank_2_56.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_22 and Cash Bank_2_23.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_27 and Cash Bank_2_34.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_27 and Cash Bank_2_58.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_30 and Cash Bank_2_32.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_30 and Cash Bank_2_27.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_32 and Cash Bank_2_60.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_37 and Cash Bank_2_23.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_40 and Cash Bank_2_47.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_40 and Cash Bank_2_51.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_43 and Cash Bank_2_45.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_43 and Cash Bank_2_40.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_45 and Cash Bank_2_53.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_51 and Cash Bank_2_48.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_53 and Cash Bank_2_49.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_47 and Cash Bank_2_53.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_48 and Cash Bank_2_53.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_58 and Cash Bank_2_35.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_60 and Cash Bank_2_37.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_34 and Cash Bank_2_60.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_35 and Cash Bank_2_60.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_23 and Cash Bank_2_24.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_49 and Cash Bank_2_23.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_69 and Cash Bank_2_70.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_88 and Cash Bank_2_90.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_70 and Cash Bank_2_84.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_84 and Cash Bank_2_86.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_86 and Cash Bank_2_88.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_86 and Cash Bank_2_72.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_88 and Cash Bank_2_91.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_90 and Cash Bank_2_95.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_111 and Cash Bank_2_113.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_108 and Cash Bank_2_109.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_109 and Cash Bank_2_111.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_109 and Cash Bank_2_98.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_111 and Cash Bank_2_114.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_113 and Cash Bank_2_118.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_122 and Cash Bank_2_108.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_70 and Cash Bank_2_122.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_13 and Cash Bank_2_129.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_2_125 and Cash Bank_2_126.
    Created DECOMPOSED_INTO relationship from Cash Bank_1_3 to Cash Bank_2_125
    Created DECOMPOSED_INTO relationship from Cash Bank_1_7 to Cash Bank_2_19
    
    Processing BPMN Cash Bank Level 3 - Managing Currency Exchange.xml at Level 3...
    Parsed 6 elements and 9 flows from BPMN Cash Bank Level 3 - Managing Currency Exchange.xml
    Created node: Task with ID Cash Bank_Managing Currency Exchange_3_6 and name Viewing Currency
    Created node: Task with ID Cash Bank_Managing Currency Exchange_3_7 and name Managing Currency Exchange
    Created node: Task with ID Cash Bank_Managing Currency Exchange_3_10 and name Validating Currency Exchange
    Created node: Task with ID Cash Bank_Managing Currency Exchange_3_21 and name Completing Currency Exchange
    Created node: StartEvent with ID Cash Bank_Managing Currency Exchange_3_5 and name Start
    Created node: EndEvent with ID Cash Bank_Managing Currency Exchange_3_20 and name End
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Managing Currency Exchange_3_21 and Cash Bank_Managing Currency Exchange_3_18.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Managing Currency Exchange_3_14 and Cash Bank_Managing Currency Exchange_3_6.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Managing Currency Exchange_3_13 and Cash Bank_Managing Currency Exchange_3_14.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Managing Currency Exchange_3_18 and Cash Bank_Managing Currency Exchange_3_19.
    Created DECOMPOSED_INTO relationship from Cash Bank_2_71 to Cash Bank_Managing Currency Exchange_3_6
    Created DECOMPOSED_INTO relationship from Cash Bank_2_80 to Cash Bank_Managing Currency Exchange_3_7
    Created DECOMPOSED_INTO relationship from Cash Bank_2_125 to Cash Bank_Managing Currency Exchange_3_5
    Created DECOMPOSED_INTO relationship from Cash Bank_2_19 to Cash Bank_Managing Currency Exchange_3_20
    
    Processing BPMN Cash Bank Level 3 - Proses Disbursing Bank Payment.xml at Level 3...
    Parsed 6 elements and 9 flows from BPMN Cash Bank Level 3 - Proses Disbursing Bank Payment.xml
    Created node: Task with ID Cash Bank_Proses Disbursing Bank Payment_3_6 and name Disburse Bank Payment
    Created node: Task with ID Cash Bank_Proses Disbursing Bank Payment_3_7 and name Validate Bank Disbursement
    Created node: Task with ID Cash Bank_Proses Disbursing Bank Payment_3_10 and name Complete Bank Disbursement Form
    Created node: Task with ID Cash Bank_Proses Disbursing Bank Payment_3_21 and name Save Bank Disbursement Data
    Created node: StartEvent with ID Cash Bank_Proses Disbursing Bank Payment_3_5 and name Start
    Created node: EndEvent with ID Cash Bank_Proses Disbursing Bank Payment_3_20 and name End
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Disbursing Bank Payment_3_21 and Cash Bank_Proses Disbursing Bank Payment_3_18.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Disbursing Bank Payment_3_14 and Cash Bank_Proses Disbursing Bank Payment_3_6.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Disbursing Bank Payment_3_13 and Cash Bank_Proses Disbursing Bank Payment_3_14.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Disbursing Bank Payment_3_18 and Cash Bank_Proses Disbursing Bank Payment_3_19.
    Created DECOMPOSED_INTO relationship from Cash Bank_2_125 to Cash Bank_Proses Disbursing Bank Payment_3_5
    Created DECOMPOSED_INTO relationship from Cash Bank_2_19 to Cash Bank_Proses Disbursing Bank Payment_3_20
    
    Processing BPMN Cash Bank Level 3 - Proses Disbursing Cash Payment.xml at Level 3...
    Parsed 6 elements and 9 flows from BPMN Cash Bank Level 3 - Proses Disbursing Cash Payment.xml
    Created node: Task with ID Cash Bank_Proses Disbursing Cash Payment_3_6 and name Manage Cash Management
    Created node: Task with ID Cash Bank_Proses Disbursing Cash Payment_3_7 and name Validate Cash Management
    Created node: Task with ID Cash Bank_Proses Disbursing Cash Payment_3_10 and name Save Cash Management Data
    Created node: Task with ID Cash Bank_Proses Disbursing Cash Payment_3_21 and name Recording to Journal
    Created node: StartEvent with ID Cash Bank_Proses Disbursing Cash Payment_3_5 and name Start
    Created node: EndEvent with ID Cash Bank_Proses Disbursing Cash Payment_3_20 and name End
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Disbursing Cash Payment_3_21 and Cash Bank_Proses Disbursing Cash Payment_3_18.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Disbursing Cash Payment_3_14 and Cash Bank_Proses Disbursing Cash Payment_3_6.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Disbursing Cash Payment_3_13 and Cash Bank_Proses Disbursing Cash Payment_3_14.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Disbursing Cash Payment_3_18 and Cash Bank_Proses Disbursing Cash Payment_3_19.
    Created DECOMPOSED_INTO relationship from Cash Bank_2_63 to Cash Bank_Proses Disbursing Cash Payment_3_6
    Created DECOMPOSED_INTO relationship from Cash Bank_2_24 to Cash Bank_Proses Disbursing Cash Payment_3_21
    Created DECOMPOSED_INTO relationship from Cash Bank_2_125 to Cash Bank_Proses Disbursing Cash Payment_3_5
    Created DECOMPOSED_INTO relationship from Cash Bank_2_19 to Cash Bank_Proses Disbursing Cash Payment_3_20
    
    Processing BPMN Cash Bank Level 3 - Proses Managing Bank Account.xml at Level 3...
    Parsed 6 elements and 9 flows from BPMN Cash Bank Level 3 - Proses Managing Bank Account.xml
    Created node: Task with ID Cash Bank_Proses Managing Bank Account_3_6 and name Manage Bank Account
    Created node: Task with ID Cash Bank_Proses Managing Bank Account_3_7 and name Complete Bank Account Form
    Created node: Task with ID Cash Bank_Proses Managing Bank Account_3_10 and name Validate Bank Account
    Created node: Task with ID Cash Bank_Proses Managing Bank Account_3_21 and name Save Bank Account Data
    Created node: StartEvent with ID Cash Bank_Proses Managing Bank Account_3_5 and name Start
    Created node: EndEvent with ID Cash Bank_Proses Managing Bank Account_3_20 and name End
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Managing Bank Account_3_21 and Cash Bank_Proses Managing Bank Account_3_18.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Managing Bank Account_3_14 and Cash Bank_Proses Managing Bank Account_3_6.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Managing Bank Account_3_13 and Cash Bank_Proses Managing Bank Account_3_14.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Managing Bank Account_3_18 and Cash Bank_Proses Managing Bank Account_3_19.
    Created DECOMPOSED_INTO relationship from Cash Bank_2_125 to Cash Bank_Proses Managing Bank Account_3_5
    Created DECOMPOSED_INTO relationship from Cash Bank_2_19 to Cash Bank_Proses Managing Bank Account_3_20
    
    Processing BPMN Cash Bank Level 3 - Proses Managing Bank Management.xml at Level 3...
    Parsed 6 elements and 11 flows from BPMN Cash Bank Level 3 - Proses Managing Bank Management.xml
    Created node: Task with ID Cash Bank_Proses Managing Bank Management_3_6 and name Manage Bank Management
    Created node: Task with ID Cash Bank_Proses Managing Bank Management_3_7 and name Validate Bank Management
    Created node: Task with ID Cash Bank_Proses Managing Bank Management_3_11 and name Save Bank Management Data
    Created node: Task with ID Cash Bank_Proses Managing Bank Management_3_22 and name Recording to Journal
    Created node: StartEvent with ID Cash Bank_Proses Managing Bank Management_3_5 and name Start
    Created node: EndEvent with ID Cash Bank_Proses Managing Bank Management_3_21 and name End
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Managing Bank Management_3_22 and Cash Bank_Proses Managing Bank Management_3_19.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Managing Bank Management_3_11 and Cash Bank_Proses Managing Bank Management_3_26.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Managing Bank Management_3_15 and Cash Bank_Proses Managing Bank Management_3_6.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Managing Bank Management_3_14 and Cash Bank_Proses Managing Bank Management_3_15.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Managing Bank Management_3_19 and Cash Bank_Proses Managing Bank Management_3_20.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Managing Bank Management_3_26 and Cash Bank_Proses Managing Bank Management_3_27.
    Created DECOMPOSED_INTO relationship from Cash Bank_2_56 to Cash Bank_Proses Managing Bank Management_3_6
    Created DECOMPOSED_INTO relationship from Cash Bank_2_24 to Cash Bank_Proses Managing Bank Management_3_22
    Created DECOMPOSED_INTO relationship from Cash Bank_2_125 to Cash Bank_Proses Managing Bank Management_3_5
    Created DECOMPOSED_INTO relationship from Cash Bank_2_19 to Cash Bank_Proses Managing Bank Management_3_21
    
    Processing BPMN Cash Bank Level 3 - Proses Managing Cash Account.xml at Level 3...
    Parsed 6 elements and 9 flows from BPMN Cash Bank Level 3 - Proses Managing Cash Account.xml
    Created node: Task with ID Cash Bank_Proses Managing Cash Account_3_6 and name Manage Cash Account
    Created node: Task with ID Cash Bank_Proses Managing Cash Account_3_7 and name Complete Cash Account Form
    Created node: Task with ID Cash Bank_Proses Managing Cash Account_3_10 and name Validate Cash Account
    Created node: Task with ID Cash Bank_Proses Managing Cash Account_3_21 and name Save Cash Account Data
    Created node: StartEvent with ID Cash Bank_Proses Managing Cash Account_3_5 and name Start
    Created node: EndEvent with ID Cash Bank_Proses Managing Cash Account_3_20 and name End
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Managing Cash Account_3_21 and Cash Bank_Proses Managing Cash Account_3_18.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Managing Cash Account_3_14 and Cash Bank_Proses Managing Cash Account_3_6.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Managing Cash Account_3_13 and Cash Bank_Proses Managing Cash Account_3_14.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Managing Cash Account_3_18 and Cash Bank_Proses Managing Cash Account_3_19.
    Created DECOMPOSED_INTO relationship from Cash Bank_2_125 to Cash Bank_Proses Managing Cash Account_3_5
    Created DECOMPOSED_INTO relationship from Cash Bank_2_19 to Cash Bank_Proses Managing Cash Account_3_20
    
    Processing BPMN Cash Bank Level 3 - Proses Managing Cash Management.xml at Level 3...
    Parsed 6 elements and 11 flows from BPMN Cash Bank Level 3 - Proses Managing Cash Management.xml
    Created node: Task with ID Cash Bank_Proses Managing Cash Management_3_6 and name Manage Cash Management
    Created node: Task with ID Cash Bank_Proses Managing Cash Management_3_7 and name Validate Cash Management
    Created node: Task with ID Cash Bank_Proses Managing Cash Management_3_11 and name Save Cash Management Data
    Created node: Task with ID Cash Bank_Proses Managing Cash Management_3_22 and name Recording to Journal
    Created node: StartEvent with ID Cash Bank_Proses Managing Cash Management_3_5 and name Start
    Created node: EndEvent with ID Cash Bank_Proses Managing Cash Management_3_21 and name End
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Managing Cash Management_3_22 and Cash Bank_Proses Managing Cash Management_3_19.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Managing Cash Management_3_11 and Cash Bank_Proses Managing Cash Management_3_26.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Managing Cash Management_3_15 and Cash Bank_Proses Managing Cash Management_3_6.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Managing Cash Management_3_14 and Cash Bank_Proses Managing Cash Management_3_15.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Managing Cash Management_3_19 and Cash Bank_Proses Managing Cash Management_3_20.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Managing Cash Management_3_26 and Cash Bank_Proses Managing Cash Management_3_27.
    Created DECOMPOSED_INTO relationship from Cash Bank_2_63 to Cash Bank_Proses Managing Cash Management_3_6
    Created DECOMPOSED_INTO relationship from Cash Bank_2_24 to Cash Bank_Proses Managing Cash Management_3_22
    Created DECOMPOSED_INTO relationship from Cash Bank_2_125 to Cash Bank_Proses Managing Cash Management_3_5
    Created DECOMPOSED_INTO relationship from Cash Bank_2_19 to Cash Bank_Proses Managing Cash Management_3_21
    
    Processing BPMN Cash Bank Level 3 - Proses Receiving Bank Receipt.xml at Level 3...
    Parsed 6 elements and 9 flows from BPMN Cash Bank Level 3 - Proses Receiving Bank Receipt.xml
    Created node: Task with ID Cash Bank_Proses Receiving Bank Receipt_3_6 and name Receive BankReceipt
    Created node: Task with ID Cash Bank_Proses Receiving Bank Receipt_3_7 and name Validate BankReceipt Detail
    Created node: Task with ID Cash Bank_Proses Receiving Bank Receipt_3_10 and name Complete Bank Receipt Form
    Created node: Task with ID Cash Bank_Proses Receiving Bank Receipt_3_21 and name Save Bank Receipt Data
    Created node: StartEvent with ID Cash Bank_Proses Receiving Bank Receipt_3_5 and name Start
    Created node: EndEvent with ID Cash Bank_Proses Receiving Bank Receipt_3_20 and name End
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Receiving Bank Receipt_3_21 and Cash Bank_Proses Receiving Bank Receipt_3_18.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Receiving Bank Receipt_3_14 and Cash Bank_Proses Receiving Bank Receipt_3_6.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Receiving Bank Receipt_3_13 and Cash Bank_Proses Receiving Bank Receipt_3_14.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Receiving Bank Receipt_3_18 and Cash Bank_Proses Receiving Bank Receipt_3_19.
    Created DECOMPOSED_INTO relationship from Cash Bank_2_125 to Cash Bank_Proses Receiving Bank Receipt_3_5
    Created DECOMPOSED_INTO relationship from Cash Bank_2_19 to Cash Bank_Proses Receiving Bank Receipt_3_20
    
    Processing BPMN Cash Bank Level 3 - Proses Receiving Cash Receipt.xml at Level 3...
    Parsed 6 elements and 9 flows from BPMN Cash Bank Level 3 - Proses Receiving Cash Receipt.xml
    Created node: Task with ID Cash Bank_Proses Receiving Cash Receipt_3_6 and name Receive Cash Receipt
    Created node: Task with ID Cash Bank_Proses Receiving Cash Receipt_3_7 and name Validate Cash Receipt Detail
    Created node: Task with ID Cash Bank_Proses Receiving Cash Receipt_3_10 and name Complete Cash Receipt Form
    Created node: Task with ID Cash Bank_Proses Receiving Cash Receipt_3_21 and name Save Cash Receipt Data
    Created node: StartEvent with ID Cash Bank_Proses Receiving Cash Receipt_3_5 and name Start
    Created node: EndEvent with ID Cash Bank_Proses Receiving Cash Receipt_3_20 and name End
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Receiving Cash Receipt_3_21 and Cash Bank_Proses Receiving Cash Receipt_3_18.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Receiving Cash Receipt_3_14 and Cash Bank_Proses Receiving Cash Receipt_3_6.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Receiving Cash Receipt_3_13 and Cash Bank_Proses Receiving Cash Receipt_3_14.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Receiving Cash Receipt_3_18 and Cash Bank_Proses Receiving Cash Receipt_3_19.
    Created DECOMPOSED_INTO relationship from Cash Bank_2_125 to Cash Bank_Proses Receiving Cash Receipt_3_5
    Created DECOMPOSED_INTO relationship from Cash Bank_2_19 to Cash Bank_Proses Receiving Cash Receipt_3_20
    
    Processing BPMN Cash Bank Level 3 - Proses Updating Currency Records.xml at Level 3...
    Parsed 7 elements and 12 flows from BPMN Cash Bank Level 3 - Proses Updating Currency Records.xml
    Created node: Task with ID Cash Bank_Proses Updating Currency Records_3_6 and name Receive Currency Data
    Created node: Task with ID Cash Bank_Proses Updating Currency Records_3_8 and name Update Currency Records
    Created node: Task with ID Cash Bank_Proses Updating Currency Records_3_19 and name Validate Updated Records
    Created node: Task with ID Cash Bank_Proses Updating Currency Records_3_20 and name Save Updated Records
    Created node: Task with ID Cash Bank_Proses Updating Currency Records_3_26 and name Recording to Journal
    Created node: StartEvent with ID Cash Bank_Proses Updating Currency Records_3_5 and name Start
    Created node: EndEvent with ID Cash Bank_Proses Updating Currency Records_3_16 and name End
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Updating Currency Records_3_26 and Cash Bank_Proses Updating Currency Records_3_23.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Updating Currency Records_3_11 and Cash Bank_Proses Updating Currency Records_3_6.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Updating Currency Records_3_10 and Cash Bank_Proses Updating Currency Records_3_11.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Updating Currency Records_3_14 and Cash Bank_Proses Updating Currency Records_3_15.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Updating Currency Records_3_20 and Cash Bank_Proses Updating Currency Records_3_14.
    Warning: Could not create relationship SEQUENCE_FLOW between Cash Bank_Proses Updating Currency Records_3_23 and Cash Bank_Proses Updating Currency Records_3_24.
    Created DECOMPOSED_INTO relationship from Cash Bank_2_24 to Cash Bank_Proses Updating Currency Records_3_26
    Created DECOMPOSED_INTO relationship from Cash Bank_2_125 to Cash Bank_Proses Updating Currency Records_3_5
    Created DECOMPOSED_INTO relationship from Cash Bank_2_19 to Cash Bank_Proses Updating Currency Records_3_16
    
    Processing BPMN Level 0.xml at Level 0...
    Parsed 16 elements and 50 flows from BPMN Level 0.xml
    Created node: Task with ID ERP_0_22 and name HRM
    Created node: Task with ID ERP_0_23 and name User Management
    Created node: Task with ID ERP_0_28 and name Inventory
    Created node: Task with ID ERP_0_30 and name Sales
    Created node: Task with ID ERP_0_32 and name Purchasing
    Created node: Task with ID ERP_0_35 and name Scheduling
    Created node: Task with ID ERP_0_40 and name Manufacturing
    Created node: Task with ID ERP_0_43 and name Finance
    Created node: Task with ID ERP_0_44 and name Account Payable
    Created node: Task with ID ERP_0_47 and name Asset Management
    Created node: Task with ID ERP_0_54 and name Accounting
    Created node: Task with ID ERP_0_58 and name Cash bank
    Created node: Task with ID ERP_0_61 and name Account Receivable
    Created node: Task with ID ERP_0_62 and name Tax
    Created node: StartEvent with ID ERP_0_69 and name Start Event
    Created node: EndEvent with ID ERP_0_83 and name End Event
    Warning: Gateway 24 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 36 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 48 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 71 has an unexpected number of incoming or outgoing flows.
    Warning: Gateway 72 has an unexpected number of incoming or outgoing flows.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_71 and ERP_0_48.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_71 and ERP_0_36.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_71 and ERP_0_30.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_71 and ERP_0_32.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_58 and ERP_0_72.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_62 and ERP_0_72.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_47 and ERP_0_72.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_35 and ERP_0_72.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_23 and ERP_0_72.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_67 and ERP_0_32.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_24 and ERP_0_22.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_24 and ERP_0_23.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_36 and ERP_0_40.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_36 and ERP_0_35.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_48 and ERP_0_43.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_48 and ERP_0_44.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_48 and ERP_0_61.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_48 and ERP_0_62.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_48 and ERP_0_47.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_48 and ERP_0_54.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_48 and ERP_0_58.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_62 and ERP_0_67.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_22 and ERP_0_72.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_28 and ERP_0_72.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_30 and ERP_0_72.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_32 and ERP_0_72.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_40 and ERP_0_72.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_54 and ERP_0_72.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_43 and ERP_0_72.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_44 and ERP_0_72.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_44 and ERP_0_72.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_61 and ERP_0_72.
    Warning: Could not create relationship SEQUENCE_FLOW between ERP_0_72 and ERP_0_83.
    Node counts by label:
    ['Task']: 250
    ['StartEvent']: 39
    ['EndEvent']: 39
    
    Relationship counts by type:
    SEQUENCE_FLOW: 220
    DECOMPOSED_INTO: 75
    
    Node counts by level:
    Level 0: 16 nodes
    Level 1: 12 nodes
    Level 2: 109 nodes
    Level 3: 191 nodes
    

# 9. Visualize the Graph


```python
def get_neo4j_data():
    with driver.session(database="erpbpmn") as session:
        nodes_query = """
        MATCH (n) 
        RETURN n.id AS id, labels(n) AS labels, n.name AS name, n.color AS color, n.level AS level, n.module AS module, n.activity AS activity
        """
        nodes = session.run(nodes_query).data()
        nodes_df = pd.DataFrame(nodes)

        relationships_query = """
        MATCH (a)-[r]->(b) 
        RETURN r.id AS id, type(r) AS type, a.id AS source, b.id AS target, r.color AS color, r.level AS level, r.module AS module, r.activity AS activity
        """
        relationships = session.run(relationships_query).data()
        relationships_df = pd.DataFrame(relationships)

    return nodes_df, relationships_df

def visualize_neo4j_graph(nodes_df, relationships_df):
    net = Network(height='750px', width='100%', directed=True, notebook=True, cdn_resources='remote')

    for _, row in nodes_df.iterrows():
        label = row['name'] if pd.notnull(row['name']) else row['id']

        net.add_node(
            row['id'],
            label=label,
            title=f"ID: {row['id']}<br>Type: {row['labels']}<br>Name: {row['name']}<br>Level: {row['level']}<br>Module: {row['module']}<br>Activity: {row['activity']}",
            color=row['color'],
            level=row['level']
        )

    for _, row in relationships_df.iterrows():
        net.add_edge(
            row['source'],
            row['target'],
            title=f"ID: {row['id']}<br>Type: {row['type']}<br>Level: {row['level']}<br>Module: {row['module']}<br>Activity: {row['activity']}",
            color=row['color'],
            arrows='to'
        )

    net.force_atlas_2based()

    net.show('neo4j_graph.html')

    from IPython.display import IFrame
    return IFrame('neo4j_graph.html', width='100%', height='750px')
```


```python
nodes_df, relationships_df = get_neo4j_data()

print("Nodes:")
display(nodes_df.head())

print("\nRelationships:")
display(relationships_df.head())

graph_display = visualize_neo4j_graph(nodes_df, relationships_df)
graph_display
```

    Nodes:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>labels</th>
      <th>name</th>
      <th>color</th>
      <th>level</th>
      <th>module</th>
      <th>activity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Asset Management_2_72</td>
      <td>[Task]</td>
      <td>Receiving Request to Transfer Asset</td>
      <td>#FAFAD2</td>
      <td>2</td>
      <td>Asset Management</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Asset Management_2_73</td>
      <td>[Task]</td>
      <td>Receiving Request to Maintenance Asset</td>
      <td>#FAFAD2</td>
      <td>2</td>
      <td>Asset Management</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Asset Management_2_74</td>
      <td>[Task]</td>
      <td>Receiving Request to Stock Take Asset</td>
      <td>#FAFAD2</td>
      <td>2</td>
      <td>Asset Management</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Asset Management_2_75</td>
      <td>[Task]</td>
      <td>Receiving Request to Revaluate Asset</td>
      <td>#FAFAD2</td>
      <td>2</td>
      <td>Asset Management</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Asset Management_2_76</td>
      <td>[Task]</td>
      <td>Receiving Request to Dispose Asset</td>
      <td>#FAFAD2</td>
      <td>2</td>
      <td>Asset Management</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>


    
    Relationships:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>type</th>
      <th>source</th>
      <th>target</th>
      <th>color</th>
      <th>level</th>
      <th>module</th>
      <th>activity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Asset Management_2_71</td>
      <td>SEQUENCE_FLOW</td>
      <td>Asset Management_2_72</td>
      <td>Asset Management_2_79</td>
      <td>#A9A9A9</td>
      <td>2.0</td>
      <td>Asset Management</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Asset Management_2_86</td>
      <td>SEQUENCE_FLOW</td>
      <td>Asset Management_2_73</td>
      <td>Asset Management_2_81</td>
      <td>#A9A9A9</td>
      <td>2.0</td>
      <td>Asset Management</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Asset Management_2_87</td>
      <td>SEQUENCE_FLOW</td>
      <td>Asset Management_2_74</td>
      <td>Asset Management_2_85</td>
      <td>#A9A9A9</td>
      <td>2.0</td>
      <td>Asset Management</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Asset Management_2_88</td>
      <td>SEQUENCE_FLOW</td>
      <td>Asset Management_2_75</td>
      <td>Asset Management_2_83</td>
      <td>#A9A9A9</td>
      <td>2.0</td>
      <td>Asset Management</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Asset Management_2_90</td>
      <td>SEQUENCE_FLOW</td>
      <td>Asset Management_2_76</td>
      <td>Asset Management_2_89</td>
      <td>#A9A9A9</td>
      <td>2.0</td>
      <td>Asset Management</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>


    neo4j_graph.html
    





<iframe
    width="100%"
    height="750px"
    src="neo4j_graph.html"
    frameborder="0"
    allowfullscreen

></iframe>



