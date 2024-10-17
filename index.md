# Leveraging Neo4j for Service Identification and Microservices Partitioning in Business Process Systems

# 1. Install Required Libraries


```python
!pip install neo4j torch
```

    Requirement already satisfied: neo4j in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (5.25.0)
    Requirement already satisfied: torch in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (2.5.0)
    Requirement already satisfied: pytz in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from neo4j) (2024.1)
    Requirement already satisfied: filelock in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from torch) (3.13.1)
    Requirement already satisfied: typing-extensions>=4.8.0 in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from torch) (4.11.0)
    Requirement already satisfied: setuptools in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from torch) (75.1.0)
    Requirement already satisfied: sympy==1.13.1 in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from torch) (1.13.1)
    Requirement already satisfied: networkx in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from torch) (3.2.1)
    Requirement already satisfied: jinja2 in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from torch) (3.1.4)
    Requirement already satisfied: fsspec in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from torch) (2024.9.0)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from sympy==1.13.1->torch) (1.3.0)
    Requirement already satisfied: MarkupSafe>=2.0 in c:\users\asus\.conda\envs\cuda_env\lib\site-packages (from jinja2->torch) (2.1.3)
    

# 2. Import Libraries

Import all necessary libraries for XML parsing, Neo4j interaction, GPU detection, and concurrent processing.


```python
# Import Libraries
import xml.etree.ElementTree as ET
from neo4j import GraphDatabase
import os
import re
import html
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

# Execute CUDA check
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
# Define functions to create nodes and relationships
def create_node(tx, label, properties):
    color_map = {
        'Task': '#ADD8E6',
        'StartEvent': '#90EE90',
        'EndEvent': '#FFB6C1',
        'Element': '#FFA07A'
    }
    color = color_map.get(label, '#D3D3D3')

    query = (
        f"MERGE (n:{label} {{id: $properties.id}}) "
        "SET n += $properties, n.color = $color "
        "RETURN n"
    )
    result = tx.run(query, properties=properties, color=color)
    return result.single()[0]

def create_relationship_with_id(tx, label1, id1, label2, id2, rel_type, properties):
    rel_color_map = {
        'SEQUENCE_FLOW': '#A9A9A9',
        'SPLIT': '#FF69B4',
        'JOIN': '#4169E1'
    }
    color = rel_color_map.get(rel_type, '#696969')

    query = (
        f"MATCH (a:{label1} {{id: $id1}}), (b:{label2} {{id: $id2}}) "
        f"MERGE (a)-[r:{rel_type} {{id: $properties.id}}]->(b) "
        "SET r += $properties, r.color = $color "
        "RETURN r"
    )
    result = tx.run(query, id1=id1, id2=id2, properties=properties, color=color)
    record = result.single()
    if record:
        return record[0]
    else:
        print(f"Warning: Could not create relationship {rel_type} between {label1}({id1}) and {label2}({id2}). One of the nodes may not exist.")
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
# Function to parse draw.io BPMN XML files
def parse_drawio_bpmn_xml(file_path, level, module, activity=None):
    tree = ET.parse(file_path)
    root = tree.getroot()

    tasks = []
    events = []
    sequence_flows = []
    
    for cell in root.findall('.//mxCell'):
        style = cell.get('style', '').lower()
        cell_id = cell.get('id')
        value = clean_name(cell.get('value', ''))
        if 'shape=mxgraph.bpmn.task' in style:
            tasks.append({'id': f"{level}_{cell_id}", 'name': value, 'level': level, 'module': module, 'activity': activity})
        elif 'shape=mxgraph.bpmn.event' in style:
            if 'outline=end' in style:
                events.append({'id': f"{level}_{cell_id}", 'name': 'End', 'type': 'EndEvent', 'level': level, 'module': module, 'activity': activity})
            else:
                events.append({'id': f"{level}_{cell_id}", 'name': 'Start', 'type': 'StartEvent', 'level': level, 'module': module, 'activity': activity})
        elif cell.get('edge') == '1':
            sequence_flows.append({
                'id': f"{level}_{cell_id}",
                'sourceRef': f"{level}_{cell.get('source')}",
                'targetRef': f"{level}_{cell.get('target')}",
                'name': value,
                'level': level,
                'module': module,
                'activity': activity
            })

    return tasks, events, sequence_flows, root
```


```python
# Helper function to determine the label of an element by its ID
def get_element_label_by_id(root, element_id):
    try:
        level, cell_id = element_id.split('_', 1)
    except ValueError:
        cell_id = element_id
    for cell in root.findall('.//mxCell'):
        if cell.get('id') == cell_id:
            style = cell.get('style', '').lower()
            if 'shape=mxgraph.bpmn.task' in style:
                return 'Task'
            elif 'shape=mxgraph.bpmn.event' in style:
                if 'outline=end' in style:
                    return 'EndEvent'
                else:
                    return 'StartEvent'
            else:
                return 'Element'
    return 'Element'
```


```python
# Main function to process all BPMN files with parallel processing
def process_bpmn_file(session, filename, file_path, level, module, activity):
    print(f"\nProcessing {filename} at Level {level}...")
    tasks, events, sequence_flows, root = parse_drawio_bpmn_xml(file_path, level, module, activity)

    print(f"Parsed {len(tasks)} tasks, {len(events)} events, {len(sequence_flows)} sequence flows from {filename}")

    for task in tasks:
        session.execute_write(create_node, 'Task', task)
    for event in events:
        session.execute_write(create_node, event['type'], event)

    # Identify split and join patterns
    source_targets = {}
    target_sources = {}
    for flow in sequence_flows:
        source = flow['sourceRef']
        target = flow['targetRef']
        source_targets.setdefault(source, set()).add(target)
        target_sources.setdefault(target, set()).add(source)

    for flow in sequence_flows:
        source_id = flow['sourceRef']
        target_id = flow['targetRef']
        rel_properties = {'id': flow['id'], 'name': flow.get('name'), 'level': level, 'module': module, 'activity': activity}
        source_label = get_element_label_by_id(root, source_id)
        target_label = get_element_label_by_id(root, target_id)
        
        # Determine relationship type
        if len(source_targets.get(source_id, [])) > 1:
            rel_type = 'SPLIT'
        elif len(target_sources.get(target_id, [])) > 1:
            rel_type = 'JOIN'
        else:
            rel_type = 'SEQUENCE_FLOW'

        rel_created = session.execute_write(
            create_relationship_with_id,
            source_label, source_id, target_label, target_id,
            rel_type, rel_properties
        )
        if rel_created:
            print(f"Created {rel_type} relationship from {source_label}({source_id}) to {target_label}({target_id})")
        else:
            print(f"Failed to create {rel_type} relationship from {source_id} to {target_id}")
```

# 7. Main Execution


```python
# Run the processing function with parallelization
def main():
    test_connection()
    bpmn_dir = './assets'
    filenames = [f for f in os.listdir(bpmn_dir) if f.endswith('.xml')]

    with driver.session(database="erpbpmn") as session:
        for filename in filenames:
            file_path = os.path.join(bpmn_dir, filename)
            if filename == "BPMN Level 0.xml":
                level = 0
                module = "ERP"
                activity = None
            else:
                match = re.match(r'BPMN\s+(.+)\s+Level\s+(\d)(?:\s+-\s+(.+))?\.xml', filename)
                if match:
                    module = match.group(1)
                    level = int(match.group(2))
                    activity = match.group(3) if match.group(3) else None
                else:
                    print(f"Skipping file {filename} as it doesn't match the expected naming pattern.")
                    continue
            
            process_bpmn_file(session, filename, file_path, level, module, activity)

```

# 8. Verification query


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
if __name__ == "__main__":
    main()
    verify_data_import()
    driver.close()
```

    Connection successful, test query result: 1
    
    Processing BPMN Account Payable Level 1.xml at Level 1...
    Parsed 1 tasks, 2 events, 2 sequence flows from BPMN Account Payable Level 1.xml
    Created SEQUENCE_FLOW relationship from Task(1_6) to EndEvent(1_8)
    Created SEQUENCE_FLOW relationship from StartEvent(1_4) to Task(1_6)
    
    Processing BPMN Account Payable Level 2.xml at Level 2...
    Parsed 21 tasks, 2 events, 38 sequence flows from BPMN Account Payable Level 2.xml
    Warning: Could not create relationship SPLIT between Element(2_80) and Element(2_13). One of the nodes may not exist.
    Failed to create SPLIT relationship from 2_80 to 2_13
    Warning: Could not create relationship SPLIT between Element(2_80) and Element(2_41). One of the nodes may not exist.
    Failed to create SPLIT relationship from 2_80 to 2_41
    Created SEQUENCE_FLOW relationship from Task(2_64) to Task(2_32)
    Created SEQUENCE_FLOW relationship from Task(2_67) to Task(2_31)
    Created JOIN relationship from Task(2_56) to Task(2_25)
    Created JOIN relationship from Task(2_42) to Task(2_27)
    Warning: Could not create relationship SPLIT between Element(2_13) and Element(2_16). One of the nodes may not exist.
    Failed to create SPLIT relationship from 2_13 to 2_16
    Warning: Could not create relationship SPLIT between Element(2_13) and Task(2_25). One of the nodes may not exist.
    Failed to create SPLIT relationship from 2_13 to 2_25
    Warning: Could not create relationship SPLIT between Element(2_16) and Element(2_17). One of the nodes may not exist.
    Failed to create SPLIT relationship from 2_16 to 2_17
    Warning: Could not create relationship SPLIT between Element(2_16) and Task(2_27). One of the nodes may not exist.
    Failed to create SPLIT relationship from 2_16 to 2_27
    Warning: Could not create relationship SPLIT between Element(2_17) and Element(2_None). One of the nodes may not exist.
    Failed to create SPLIT relationship from 2_17 to 2_None
    Warning: Could not create relationship SPLIT between Element(2_17) and Element(2_19). One of the nodes may not exist.
    Failed to create SPLIT relationship from 2_17 to 2_19
    Warning: Could not create relationship JOIN between Element(2_19) and Element(2_None). One of the nodes may not exist.
    Failed to create JOIN relationship from 2_19 to 2_None
    Warning: Could not create relationship SEQUENCE_FLOW between Element(2_None) and Task(2_29). One of the nodes may not exist.
    Failed to create SEQUENCE_FLOW relationship from 2_None to 2_29
    Created SEQUENCE_FLOW relationship from Task(2_25) to Task(2_26)
    Created JOIN relationship from Task(2_29) to Task(2_30)
    Created JOIN relationship from Task(2_26) to Task(2_30)
    Created JOIN relationship from Task(2_33) to Task(2_30)
    Created JOIN relationship from Task(2_34) to Task(2_30)
    Created SEQUENCE_FLOW relationship from Task(2_30) to EndEvent(2_23)
    Warning: Could not create relationship SPLIT between Element(2_41) and Element(2_44). One of the nodes may not exist.
    Failed to create SPLIT relationship from 2_41 to 2_44
    Warning: Could not create relationship SEQUENCE_FLOW between Element(2_44) and Task(2_47). One of the nodes may not exist.
    Failed to create SEQUENCE_FLOW relationship from 2_44 to 2_47
    Warning: Could not create relationship SPLIT between Element(2_41) and Task(2_45). One of the nodes may not exist.
    Failed to create SPLIT relationship from 2_41 to 2_45
    Warning: Could not create relationship SEQUENCE_FLOW between Task(2_49) and Element(2_66). One of the nodes may not exist.
    Failed to create SEQUENCE_FLOW relationship from 2_49 to 2_66
    Created SEQUENCE_FLOW relationship from Task(2_47) to Task(2_49)
    Created SEQUENCE_FLOW relationship from Task(2_45) to Task(2_51)
    Warning: Could not create relationship SPLIT between Element(2_54) and Element(2_59). One of the nodes may not exist.
    Failed to create SPLIT relationship from 2_54 to 2_59
    Warning: Could not create relationship SEQUENCE_FLOW between Task(2_51) and Element(2_54). One of the nodes may not exist.
    Failed to create SEQUENCE_FLOW relationship from 2_51 to 2_54
    Warning: Could not create relationship SPLIT between Element(2_54) and Task(2_56). One of the nodes may not exist.
    Failed to create SPLIT relationship from 2_54 to 2_56
    Warning: Could not create relationship SEQUENCE_FLOW between Element(2_59) and Task(2_77). One of the nodes may not exist.
    Failed to create SEQUENCE_FLOW relationship from 2_59 to 2_77
    Created SEQUENCE_FLOW relationship from Task(2_63) to Task(2_42)
    Warning: Could not create relationship SPLIT between Element(2_66) and Element(2_72). One of the nodes may not exist.
    Failed to create SPLIT relationship from 2_66 to 2_72
    Warning: Could not create relationship SPLIT between Element(2_66) and Task(2_67). One of the nodes may not exist.
    Failed to create SPLIT relationship from 2_66 to 2_67
    Created SEQUENCE_FLOW relationship from Task(2_71) to Task(2_75)
    Warning: Could not create relationship SEQUENCE_FLOW between Element(2_72) and Task(2_71). One of the nodes may not exist.
    Failed to create SEQUENCE_FLOW relationship from 2_72 to 2_71
    Created SEQUENCE_FLOW relationship from Task(2_77) to Task(2_63)
    Created SEQUENCE_FLOW relationship from Task(2_75) to Task(2_64)
    Warning: Could not create relationship SEQUENCE_FLOW between StartEvent(2_79) and Element(2_80). One of the nodes may not exist.
    Failed to create SEQUENCE_FLOW relationship from 2_79 to 2_80
    
    Processing BPMN Account Payable Level 3 - Creating the Purchase DP Invoice.xml at Level 3...
    Parsed 6 tasks, 2 events, 17 sequence flows from BPMN Account Payable Level 3 - Creating the Purchase DP Invoice.xml
    Created JOIN relationship from StartEvent(3_4) to Task(3_5)
    Created JOIN relationship from Task(3_5) to Task(3_6)
    Warning: Could not create relationship SPLIT between Element(3_11) and Task(3_14). One of the nodes may not exist.
    Failed to create SPLIT relationship from 3_11 to 3_14
    Warning: Could not create relationship SPLIT between Element(3_11) and Element(3_17). One of the nodes may not exist.
    Failed to create SPLIT relationship from 3_11 to 3_17
    Warning: Could not create relationship SEQUENCE_FLOW between Task(3_6) and Element(3_11). One of the nodes may not exist.
    Failed to create SEQUENCE_FLOW relationship from 3_6 to 3_11
    Warning: Could not create relationship JOIN between Task(3_14) and Element(3_36). One of the nodes may not exist.
    Failed to create JOIN relationship from 3_14 to 3_36
    Created SEQUENCE_FLOW relationship from Task(3_16) to Task(3_29)
    Warning: Could not create relationship SEQUENCE_FLOW between Element(3_17) and Task(3_16). One of the nodes may not exist.
    Failed to create SEQUENCE_FLOW relationship from 3_17 to 3_16
    Warning: Could not create relationship JOIN between Element(3_22) and Task(3_5). One of the nodes may not exist.
    Failed to create JOIN relationship from 3_22 to 3_5
    Warning: Could not create relationship SEQUENCE_FLOW between Element(3_21) and Element(3_22). One of the nodes may not exist.
    Failed to create SEQUENCE_FLOW relationship from 3_21 to 3_22
    Warning: Could not create relationship JOIN between Element(3_26) and Task(3_6). One of the nodes may not exist.
    Failed to create JOIN relationship from 3_26 to 3_6
    Warning: Could not create relationship SEQUENCE_FLOW between Element(3_24) and Element(3_26). One of the nodes may not exist.
    Failed to create SEQUENCE_FLOW relationship from 3_24 to 3_26
    Created SEQUENCE_FLOW relationship from Task(3_29) to Task(3_30)
    Warning: Could not create relationship SEQUENCE_FLOW between Element(3_32) and Element(3_34). One of the nodes may not exist.
    Failed to create SEQUENCE_FLOW relationship from 3_32 to 3_34
    Warning: Could not create relationship SPLIT between Task(3_30) and Element(3_32). One of the nodes may not exist.
    Failed to create SPLIT relationship from 3_30 to 3_32
    Warning: Could not create relationship SPLIT between Task(3_30) and Element(3_36). One of the nodes may not exist.
    Failed to create SPLIT relationship from 3_30 to 3_36
    Warning: Could not create relationship SEQUENCE_FLOW between Element(3_36) and EndEvent(3_35). One of the nodes may not exist.
    Failed to create SEQUENCE_FLOW relationship from 3_36 to 3_35
    
    Processing BPMN Account Payable Level 3 - Finalizing Purchase DP Invoice Document.xml at Level 3...
    Parsed 3 tasks, 2 events, 8 sequence flows from BPMN Account Payable Level 3 - Finalizing Purchase DP Invoice Document.xml
    Created JOIN relationship from StartEvent(3_4) to Task(3_5)
    Created SEQUENCE_FLOW relationship from Task(3_7) to Task(3_17)
    Warning: Could not create relationship JOIN between Element(3_10) and Task(3_5). One of the nodes may not exist.
    Failed to create JOIN relationship from 3_10 to 3_5
    Warning: Could not create relationship SEQUENCE_FLOW between Element(3_9) and Element(3_10). One of the nodes may not exist.
    Failed to create SEQUENCE_FLOW relationship from 3_9 to 3_10
    Warning: Could not create relationship SEQUENCE_FLOW between Element(3_13) and Element(3_14). One of the nodes may not exist.
    Failed to create SEQUENCE_FLOW relationship from 3_13 to 3_14
    Created SEQUENCE_FLOW relationship from Task(3_5) to Task(3_7)
    Created SPLIT relationship from Task(3_17) to EndEvent(3_15)
    Warning: Could not create relationship SPLIT between Task(3_17) and Element(3_13). One of the nodes may not exist.
    Failed to create SPLIT relationship from 3_17 to 3_13
    
    Processing BPMN Account Payable Level 3 - Reviewing Purchase DP Invoice.xml at Level 3...
    Parsed 4 tasks, 2 events, 13 sequence flows from BPMN Account Payable Level 3 - Reviewing Purchase DP Invoice.xml
    Created JOIN relationship from StartEvent(3_4) to Task(3_5)
    Warning: Could not create relationship JOIN between Element(3_9) and Task(3_5). One of the nodes may not exist.
    Failed to create JOIN relationship from 3_9 to 3_5
    Warning: Could not create relationship SEQUENCE_FLOW between Element(3_8) and Element(3_9). One of the nodes may not exist.
    Failed to create SEQUENCE_FLOW relationship from 3_8 to 3_9
    Warning: Could not create relationship SEQUENCE_FLOW between Element(3_12) and Element(3_13). One of the nodes may not exist.
    Failed to create SEQUENCE_FLOW relationship from 3_12 to 3_13
    Warning: Could not create relationship SEQUENCE_FLOW between Task(3_16) and Element(3_19). One of the nodes may not exist.
    Failed to create SEQUENCE_FLOW relationship from 3_16 to 3_19
    Created SEQUENCE_FLOW relationship from Task(3_5) to Task(3_16)
    Warning: Could not create relationship SPLIT between Element(3_19) and Element(3_20). One of the nodes may not exist.
    Failed to create SPLIT relationship from 3_19 to 3_20
    Warning: Could not create relationship SPLIT between Element(3_19) and Task(3_6). One of the nodes may not exist.
    Failed to create SPLIT relationship from 3_19 to 3_6
    Warning: Could not create relationship SEQUENCE_FLOW between Element(3_20) and Task(3_23). One of the nodes may not exist.
    Failed to create SEQUENCE_FLOW relationship from 3_20 to 3_23
    Warning: Could not create relationship SPLIT between Task(3_23) and Element(3_12). One of the nodes may not exist.
    Failed to create SPLIT relationship from 3_23 to 3_12
    Warning: Could not create relationship SEQUENCE_FLOW between Element(3_28) and EndEvent(3_14). One of the nodes may not exist.
    Failed to create SEQUENCE_FLOW relationship from 3_28 to 3_14
    Warning: Could not create relationship JOIN between Task(3_6) and Element(3_28). One of the nodes may not exist.
    Failed to create JOIN relationship from 3_6 to 3_28
    Warning: Could not create relationship SPLIT between Task(3_23) and Element(3_28). One of the nodes may not exist.
    Failed to create SPLIT relationship from 3_23 to 3_28
    
    Processing BPMN Account Payable Level 3 - Updating Purchase DP Invoice Data.xml at Level 3...
    Parsed 2 tasks, 2 events, 7 sequence flows from BPMN Account Payable Level 3 - Updating Purchase DP Invoice Data.xml
    Created JOIN relationship from StartEvent(3_4) to Task(3_5)
    Warning: Could not create relationship SPLIT between Task(3_7) and Element(3_13). One of the nodes may not exist.
    Failed to create SPLIT relationship from 3_7 to 3_13
    Warning: Could not create relationship JOIN between Element(3_10) and Task(3_5). One of the nodes may not exist.
    Failed to create JOIN relationship from 3_10 to 3_5
    Warning: Could not create relationship SEQUENCE_FLOW between Element(3_9) and Element(3_10). One of the nodes may not exist.
    Failed to create SEQUENCE_FLOW relationship from 3_9 to 3_10
    Warning: Could not create relationship SEQUENCE_FLOW between Element(3_13) and Element(3_14). One of the nodes may not exist.
    Failed to create SEQUENCE_FLOW relationship from 3_13 to 3_14
    Created SEQUENCE_FLOW relationship from Task(3_5) to Task(3_7)
    Created SPLIT relationship from Task(3_7) to EndEvent(3_15)
    
    Processing BPMN Level 0.xml at Level 0...
    Parsed 14 tasks, 2 events, 53 sequence flows from BPMN Level 0.xml
    Warning: Could not create relationship SPLIT between Element(0_70) and Element(0_49). One of the nodes may not exist.
    Failed to create SPLIT relationship from 0_70 to 0_49
    Warning: Could not create relationship SPLIT between Element(0_70) and Element(0_37). One of the nodes may not exist.
    Failed to create SPLIT relationship from 0_70 to 0_37
    Warning: Could not create relationship SPLIT between Element(0_70) and Task(0_31). One of the nodes may not exist.
    Failed to create SPLIT relationship from 0_70 to 0_31
    Warning: Could not create relationship SPLIT between Element(0_70) and Task(0_33). One of the nodes may not exist.
    Failed to create SPLIT relationship from 0_70 to 0_33
    Warning: Could not create relationship SPLIT between Element(0_None) and Task(0_29). One of the nodes may not exist.
    Failed to create SPLIT relationship from 0_None to 0_29
    Warning: Could not create relationship SPLIT between Element(0_None) and Element(0_25). One of the nodes may not exist.
    Failed to create SPLIT relationship from 0_None to 0_25
    Warning: Could not create relationship SPLIT between Task(0_59) and Element(0_71). One of the nodes may not exist.
    Failed to create SPLIT relationship from 0_59 to 0_71
    Warning: Could not create relationship SPLIT between Task(0_63) and Element(0_71). One of the nodes may not exist.
    Failed to create SPLIT relationship from 0_63 to 0_71
    Warning: Could not create relationship SPLIT between Task(0_48) and Element(0_71). One of the nodes may not exist.
    Failed to create SPLIT relationship from 0_48 to 0_71
    Warning: Could not create relationship JOIN between Task(0_36) and Element(0_71). One of the nodes may not exist.
    Failed to create JOIN relationship from 0_36 to 0_71
    Warning: Could not create relationship JOIN between Task(0_24) and Element(0_71). One of the nodes may not exist.
    Failed to create JOIN relationship from 0_24 to 0_71
    Warning: Could not create relationship SPLIT between Element(0_None) and Element(0_71). One of the nodes may not exist.
    Failed to create SPLIT relationship from 0_None to 0_71
    Created SPLIT relationship from Task(0_23) to Task(0_55)
    Created SPLIT relationship from Task(0_63) to Task(0_31)
    Created SPLIT relationship from Task(0_31) to Task(0_62)
    Created SPLIT relationship from Task(0_23) to Task(0_44)
    Created SPLIT relationship from Task(0_63) to Task(0_33)
    Created SPLIT relationship from Task(0_23) to Task(0_41)
    Warning: Could not create relationship SPLIT between Element(0_25) and Task(0_23). One of the nodes may not exist.
    Failed to create SPLIT relationship from 0_25 to 0_23
    Warning: Could not create relationship SPLIT between Element(0_25) and Task(0_24). One of the nodes may not exist.
    Failed to create SPLIT relationship from 0_25 to 0_24
    Created SPLIT relationship from Task(0_48) to Task(0_41)
    Warning: Could not create relationship SPLIT between Element(0_37) and Task(0_41). One of the nodes may not exist.
    Failed to create SPLIT relationship from 0_37 to 0_41
    Warning: Could not create relationship SPLIT between Element(0_37) and Task(0_36). One of the nodes may not exist.
    Failed to create SPLIT relationship from 0_37 to 0_36
    Created SPLIT relationship from Task(0_41) to Task(0_36)
    Created SPLIT relationship from Task(0_45) to Task(0_59)
    Created SPLIT relationship from Task(0_62) to Task(0_59)
    Created SPLIT relationship from Task(0_48) to Task(0_44)
    Warning: Could not create relationship SPLIT between Element(0_49) and Task(0_44). One of the nodes may not exist.
    Failed to create SPLIT relationship from 0_49 to 0_44
    Warning: Could not create relationship SPLIT between Element(0_49) and Task(0_45). One of the nodes may not exist.
    Failed to create SPLIT relationship from 0_49 to 0_45
    Warning: Could not create relationship SPLIT between Element(0_49) and Task(0_62). One of the nodes may not exist.
    Failed to create SPLIT relationship from 0_49 to 0_62
    Warning: Could not create relationship SPLIT between Element(0_49) and Task(0_63). One of the nodes may not exist.
    Failed to create SPLIT relationship from 0_49 to 0_63
    Warning: Could not create relationship SPLIT between Element(0_49) and Task(0_48). One of the nodes may not exist.
    Failed to create SPLIT relationship from 0_49 to 0_48
    Warning: Could not create relationship SPLIT between Element(0_49) and Task(0_55). One of the nodes may not exist.
    Failed to create SPLIT relationship from 0_49 to 0_55
    Created SPLIT relationship from Task(0_48) to Task(0_55)
    Created SPLIT relationship from Task(0_55) to Task(0_44)
    Warning: Could not create relationship SPLIT between Element(0_49) and Task(0_59). One of the nodes may not exist.
    Failed to create SPLIT relationship from 0_49 to 0_59
    Created SPLIT relationship from Task(0_59) to Task(0_55)
    Created SPLIT relationship from Task(0_31) to Task(0_29)
    Created SPLIT relationship from Task(0_33) to Task(0_29)
    Created SPLIT relationship from Task(0_33) to Task(0_29)
    Created SPLIT relationship from Task(0_41) to Task(0_55)
    Warning: Could not create relationship SEQUENCE_FLOW between StartEvent(0_68) and Element(0_None). One of the nodes may not exist.
    Failed to create SEQUENCE_FLOW relationship from 0_68 to 0_None
    Warning: Could not create relationship SPLIT between Task(0_23) and Element(0_71). One of the nodes may not exist.
    Failed to create SPLIT relationship from 0_23 to 0_71
    Warning: Could not create relationship JOIN between Task(0_29) and Element(0_71). One of the nodes may not exist.
    Failed to create JOIN relationship from 0_29 to 0_71
    Warning: Could not create relationship SPLIT between Task(0_31) and Element(0_71). One of the nodes may not exist.
    Failed to create SPLIT relationship from 0_31 to 0_71
    Warning: Could not create relationship SPLIT between Task(0_33) and Element(0_71). One of the nodes may not exist.
    Failed to create SPLIT relationship from 0_33 to 0_71
    Warning: Could not create relationship SPLIT between Task(0_41) and Element(0_71). One of the nodes may not exist.
    Failed to create SPLIT relationship from 0_41 to 0_71
    Warning: Could not create relationship SPLIT between Task(0_55) and Element(0_71). One of the nodes may not exist.
    Failed to create SPLIT relationship from 0_55 to 0_71
    Warning: Could not create relationship JOIN between Task(0_44) and Element(0_71). One of the nodes may not exist.
    Failed to create JOIN relationship from 0_44 to 0_71
    Warning: Could not create relationship SPLIT between Task(0_45) and Element(0_71). One of the nodes may not exist.
    Failed to create SPLIT relationship from 0_45 to 0_71
    Warning: Could not create relationship SPLIT between Task(0_45) and Element(0_71). One of the nodes may not exist.
    Failed to create SPLIT relationship from 0_45 to 0_71
    Warning: Could not create relationship SPLIT between Task(0_62) and Element(0_71). One of the nodes may not exist.
    Failed to create SPLIT relationship from 0_62 to 0_71
    Warning: Could not create relationship SEQUENCE_FLOW between Element(0_71) and EndEvent(0_82). One of the nodes may not exist.
    Failed to create SEQUENCE_FLOW relationship from 0_71 to 0_82
    Node counts by label:
    ['Task']: 45
    ['EndEvent']: 6
    ['StartEvent']: 4
    
    Relationship counts by type:
    SPLIT: 20
    SEQUENCE_FLOW: 17
    JOIN: 8
    
    Node counts by level:
    Level 0: 16 nodes
    Level 1: 3 nodes
    Level 2: 23 nodes
    Level 3: 13 nodes
    
