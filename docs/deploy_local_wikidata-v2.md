# Deploying a Local Offline Wikidata Server with QEndpoint

This guide provides a comprehensive walkthrough for deploying a completely offline, zero-latency clone of Wikidata using [QEndpoint](https://github.com/the-qa-company/qEndpoint).

By default, QEndpoint is designed to automatically download the necessary dataset and search indices upon first boot. This document covers the default automatic deployment, as well as a manual fallback method for environments with strict firewalls or timeout limits.

---

## Method 1: The Default Deployment (Automatic Download)

If your deployment environment has unrestricted internet access, QEndpoint will automatically fetch the latest Wikidata HDT file and compute its index.

### Option A: Using Docker (Standard)

Run the following command to start the server. It will create a local volume, download the dataset, and expose the SPARQL endpoint on port 9090.

```bash
docker run -d \
  -p 9090:9090 \
  -e HDT_BASE=wikidata_all \
  -e MEM_SIZE=64G \
  -e SERVER_PORT=9090 \
  -v $(pwd)/qendpoint_data:/app/qendpoint_data \
  qanswer/qendpoint:latest
```

> **Note:** The official upstream image is `qacompany/qendpoint-wikidata` and defaults to port `1234`. Adjust `SERVER_PORT`, port mapping, and image name if you use that image instead.

### Option B: Using Apptainer / Singularity (HPC Environments)

For high-performance computing clusters where Docker is unavailable, use Apptainer/Singularity.

```bash
# 1. Build the sandbox container (if not already created)
singularity build --sandbox qendpoint_data docker://qanswer/qendpoint:latest

# 2. Set environment variables
export APPTAINERENV_HDT_BASE=wikidata_all
export APPTAINERENV_MEM_SIZE=64G
export APPTAINERENV_SERVER_PORT=9090

# 3. Run the endpoint
singularity run --writable --pwd /app qendpoint_data
```

The initial boot takes significant time while it downloads over 200GB of data. Subsequent boots are much faster because the volume already contains the HDT file and index.

---

## Method 2: The Fallback (Manual Deployment)

If the automated download fails, times out, or you are on an isolated compute node behind a strict firewall, download the data and generate the search index manually.

### Step 1: Prepare the Directory

QEndpoint expects the data in a specific folder structure inside your volume or sandbox.

```bash
mkdir -p qendpoint_data/app/qendpoint/hdt-store/
cd qendpoint_data/app/qendpoint/hdt-store/
```

### Step 2: Download the Wikidata HDT File

On a machine or login node with internet access, download the pre-compressed dataset from the [RDF HDT Datasets page](https://www.rdfhdt.org/datasets/).

```bash
# Download the dataset (use tmux or screen to prevent disconnects)
wget https://www.rdfhdt.org/datasets/wikidata-2024-08-16.hdt

# Rename it so QEndpoint recognizes it
mv wikidata-2024-08-16.hdt index_dev.hdt
```

Pick the latest Wikidata HDT snapshot listed on the datasets page if a newer file is available.

### Step 3: Manually Generate the Search Index

To query the file instantly, QEndpoint needs a `.index.v1-1` file. Generate it with the official `hdt-java` tools.

```bash
# Clone and build HDT-Java
git clone https://github.com/rdfhdt/hdt-java.git
cd hdt-java
mvn clean install

# Extract the standalone distribution
cd hdt-java-package/target/
tar -xzf hdt-java-package-*-distribution.tar.gz
cd hdt-java-package-*/bin/

# Allocate maximum RAM and force index generation
export _JAVA_OPTIONS="-Xmx64G"
./hdtSearch.sh /path/to/qendpoint_data/app/qendpoint/hdt-store/index_dev.hdt
```

When the interactive prompt (`>>`) appears, type `exit`. You should then have `index_dev.hdt.index.v1-1` in the same directory as the HDT file.

### Step 4: Start the Server

Once both `index_dev.hdt` and `index_dev.hdt.index.v1-1` are in the `hdt-store` folder, run the Docker or Apptainer commands from **Method 1**. The container detects the files and skips the download phase.

---

## Querying the Endpoint in Python

The machine-to-machine API is at `/api/endpoint/sparql`. Because of Spring Security defaults, use **POST** requests with form-url-encoded payloads for SPARQL queries to avoid `405 Method Not Allowed` errors.

```python
import json

import requests


def query_local_wikidata(sparql_query, port=9090):
    url = f"http://localhost:{port}/api/endpoint/sparql"

    headers = {
        "Accept": "application/sparql-results+json",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    payload = {"query": sparql_query}

    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        data = response.json()
        return data["results"]["bindings"]
    except requests.exceptions.RequestException as e:
        print(f"Error querying QEndpoint: {e}")
        return None


if __name__ == "__main__":
    # Example: founding date and coordinates of the University of Oregon
    query = """
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>

    SELECT ?founded ?location WHERE {
      wd:Q766145 wdt:P571 ?founded ;
                 wdt:P625 ?location .
    }
    """

    results = query_local_wikidata(query)

    if results:
        print(json.dumps(results, indent=2))
        for row in results:
            print(f"Founded: {row['founded']['value']}")
            print(f"Location: {row['location']['value']}")
```

---

## Operational Notes

| Setting | Recommendation |
|---------|----------------|
| `MEM_SIZE` | At least `10G` for `wikidata_all` per upstream docs; use `64G` if you have headroom for faster indexing and query performance |
| Persistent volume | Mount `qendpoint_data` so downloads and indexes survive container restarts |
| First boot | Expect a long download and index build; monitor disk space (200GB+) |
| Firewall fallback | Use Method 2 on a login node, then copy `qendpoint_data/` to the compute node |

---

## Using with `langgraph_coe`

Point `wikidata.sparql_endpoint` in `langgraph_coe/config.yaml` at your tunnel or node URL (for example `http://127.0.0.1:30162/api/endpoint/sparql`). Entity search and Wikipedia still use the public APIs; only k-hop SPARQL traversals hit QEndpoint.

Agent wiring, hop budgets, pruning, and Redis `wd:*` cache: **[setup_wikidata_tools.md](setup_wikidata_tools.md)**.
