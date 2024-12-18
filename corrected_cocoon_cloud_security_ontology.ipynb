{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "b69d1bac",
   "metadata": {},
   "source": [
    "# **Simplified CoCoOn Cloud Security Ontology**\n",
    "This notebook demonstrates a minimal version of the CoCoOn Cloud Security Ontology for cloud environments using RDFLib in Python. It models relationships between assets, vulnerabilities, threats, and controls."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c743f11c",
   "metadata": {},
   "source": [
    "## **Step 1: Install Required Libraries**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "619d6b8f",
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install rdflib pandas networkx matplotlib"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8814cf4c",
   "metadata": {},
   "source": [
    "## **Step 2: Import Libraries**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "eb906139",
   "metadata": {},
   "outputs": [],
   "source": [
    "from rdflib import Graph, Namespace, RDF, OWL, URIRef\n",
    "import pandas as pd\n",
    "import networkx as nx\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9098d304",
   "metadata": {},
   "source": [
    "## **Step 3: Initialize Ontology and Namespace**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d4b1e5c2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize the graph and define namespace\n",
    "g = Graph()\n",
    "COCOON = Namespace(\"http://example.org/cocoon#\")\n",
    "g.bind(\"cocoon\", COCOON)\n",
    "\n",
    "# Define Classes\n",
    "g.add((COCOON.Asset, RDF.type, OWL.Class))\n",
    "g.add((COCOON.Threat, RDF.type, OWL.Class))\n",
    "g.add((COCOON.Vulnerability, RDF.type, OWL.Class))\n",
    "g.add((COCOON.Control, RDF.type, OWL.Class))\n",
    "\n",
    "# Define Relationships\n",
    "g.add((COCOON.hasVulnerability, RDF.type, OWL.ObjectProperty))\n",
    "g.add((COCOON.exploitedBy, RDF.type, OWL.ObjectProperty))\n",
    "g.add((COCOON.mitigatedBy, RDF.type, OWL.ObjectProperty))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "76577733",
   "metadata": {},
   "source": [
    "## **Step 4: Add Instances and Relationships**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "db7094ee",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Add Instances\n",
    "asset1 = URIRef(COCOON.CloudServer)\n",
    "vuln1 = URIRef(COCOON.DataExposure)\n",
    "threat1 = URIRef(COCOON.MalwareAttack)\n",
    "control1 = URIRef(COCOON.Encryption)\n",
    "\n",
    "# Define relationships\n",
    "g.add((asset1, RDF.type, COCOON.Asset))\n",
    "g.add((asset1, COCOON.hasVulnerability, vuln1))\n",
    "g.add((vuln1, RDF.type, COCOON.Vulnerability))\n",
    "g.add((vuln1, COCOON.exploitedBy, threat1))\n",
    "g.add((threat1, RDF.type, COCOON.Threat))\n",
    "g.add((vuln1, COCOON.mitigatedBy, control1))\n",
    "g.add((control1, RDF.type, COCOON.Control))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cad6761d",
   "metadata": {},
   "source": [
    "## **Step 5: Query the Ontology**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "972a0e09",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Query for vulnerabilities and the threats exploiting them\n",
    "results = []\n",
    "for s, p, o in g.triples((None, COCOON.exploitedBy, None)):\n",
    "    results.append((s.split(\"#\")[-1], o.split(\"#\")[-1]))\n",
    "\n",
    "# Convert to pandas DataFrame\n",
    "df = pd.DataFrame(results, columns=[\"Vulnerability\", \"Exploited By\"])\n",
    "print(\"--- Query Results ---\")\n",
    "print(df)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bd861c63",
   "metadata": {},
   "source": [
    "## **Step 6: Visualize Results as a Graph**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "aa25dd27",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize relationships with NetworkX\n",
    "G = nx.DiGraph()\n",
    "for _, row in df.iterrows():\n",
    "    G.add_edge(row['Vulnerability'], row['Exploited By'], label='exploitedBy')\n",
    "\n",
    "# Draw graph\n",
    "plt.figure(figsize=(8, 6))\n",
    "pos = nx.spring_layout(G)\n",
    "nx.draw(G, pos, with_labels=True, node_size=2000, font_size=12, font_weight=\"bold\")\n",
    "nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['label'] for u, v, d in G.edges(data=True)})\n",
    "plt.title(\"CoCoOn Cloud Security Ontology Graph\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f9db1113",
   "metadata": {},
   "source": [
    "## **Step 7: Save Ontology to File**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3960cded",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save the ontology in Turtle format\n",
    "output_path = \"/content/cocoon_cloud_security_ontology.ttl\"\n",
    "g.serialize(destination=output_path, format=\"turtle\")\n",
    "print(f\"Ontology saved to: {output_path}\")"
   ]
  }
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}
