{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "8d318e7a",
   "metadata": {},
   "source": [
    "# **Simplified Takeshi Takahashi Cybersecurity Ontology**\n",
    "This notebook demonstrates a minimal version of Takeshi Takahashi et al.'s cybersecurity ontology in Python with table and graph outputs."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "45714fc1",
   "metadata": {},
   "source": [
    "## **Step 1: Install Required Libraries**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d681004a",
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install rdflib pandas networkx matplotlib"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e2ec4d90",
   "metadata": {},
   "source": [
    "## **Step 2: Import Libraries**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9ad70f7a",
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
   "id": "4575f1ad",
   "metadata": {},
   "source": [
    "## **Step 3: Define Ontology and Instances**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7b88b52f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize Ontology and Namespace\n",
    "g = Graph()\n",
    "CYBERSEC = Namespace(\"http://example.org/cybersecurity#\")\n",
    "g.bind(\"cybersec\", CYBERSEC)\n",
    "\n",
    "# Define Classes and Property\n",
    "g.add((CYBERSEC.Asset, RDF.type, OWL.Class))\n",
    "g.add((CYBERSEC.Threat, RDF.type, OWL.Class))\n",
    "g.add((CYBERSEC.Vulnerability, RDF.type, OWL.Class))\n",
    "g.add((CYBERSEC.exploitedBy, RDF.type, OWL.ObjectProperty))\n",
    "\n",
    "# Add Instances\n",
    "asset = URIRef(CYBERSEC.Server)\n",
    "vulnerability = URIRef(CYBERSEC.SQLInjection)\n",
    "threat = URIRef(CYBERSEC.HackerAttack)\n",
    "\n",
    "# Add Relationships\n",
    "g.add((asset, RDF.type, CYBERSEC.Asset))\n",
    "g.add((vulnerability, RDF.type, CYBERSEC.Vulnerability))\n",
    "g.add((threat, RDF.type, CYBERSEC.Threat))\n",
    "g.add((vulnerability, CYBERSEC.exploitedBy, threat))\n",
    "g.add((asset, CYBERSEC.exploitedBy, vulnerability))\n",
    "\n",
    "print(\"Ontology initialized with instances and relationships.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b8fc0717",
   "metadata": {},
   "source": [
    "## **Step 4: Query the Ontology**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5a63db13",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Query Results\n",
    "results = []\n",
    "for s, p, o in g.triples((None, CYBERSEC.exploitedBy, None)):\n",
    "    results.append((s.split(\"#\")[-1], o.split(\"#\")[-1]))\n",
    "\n",
    "# Convert results to pandas DataFrame\n",
    "df = pd.DataFrame(results, columns=[\"Subject\", \"Exploited By\"])\n",
    "print(\"Query Results:\")\n",
    "print(df)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fb96601e",
   "metadata": {},
   "source": [
    "## **Step 5: Visualize Ontology Relationships as a Graph**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3513ec2c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize the relationships using NetworkX\n",
    "G = nx.DiGraph()\n",
    "for _, row in df.iterrows():\n",
    "    G.add_edge(row[\"Subject\"], row[\"Exploited By\"], label=\"exploitedBy\")\n",
    "\n",
    "# Draw the graph\n",
    "plt.figure(figsize=(8, 6))\n",
    "pos = nx.spring_layout(G)\n",
    "nx.draw(G, pos, with_labels=True, node_size=2000, font_size=12, font_weight=\"bold\")\n",
    "nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['label'] for u, v, d in G.edges(data=True)})\n",
    "plt.title(\"Simplified Cybersecurity Ontology Visualization\")\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}
