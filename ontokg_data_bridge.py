# ontokg_data_bridge.py
"""
Cau noi giua dataset TransMTL va OntoKG (Neo4j).
Mat xich con thieu de tich hop: tu article_id -> truy van Neo4j -> kg_batch.

Cach dung trong train_v2.py / testing_v2.py:

    from ontokg_data_bridge import OntoKGBridge

    bridge = OntoKGBridge(
        uri="bolt://localhost:7687", user="neo4j", password="password",
        d_model=300, enabled=use_ontokg,
    )

    # Trong vong lap batch (dataset da tra them article_ids):
    kg_batch = bridge.build_kg_batch(article_ids)   # None neu enabled=False
    out = model(inp=src, tar=tgt, labels=labels, task="both",
                training=True, kg_batch=kg_batch)

    # Cuoi chuong trinh:
    bridge.close()
"""
from typing import List, Optional, Dict, Any

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class OntoKGBridge:
    """Truy van subgraph tu Neo4j cho ca batch, tra ve kg_batch cho TransMTL."""

    def __init__(self, uri, user, password, d_model=300,
                 database="neo4j", enabled=True):
        self.enabled = enabled
        self.d_model = d_model
        self.retriever = None
        if not enabled:
            return
        # Import o day de khi tat OntoKG khong can neo4j
        from OntoKG.module9_neo4j_retrieval import Neo4jRetriever
        # LUU Y: entity embedding trong Neo4j la 768 chieu (tu Module 7),
        # GraphEncoder se chieu 768 -> d_model. Nen retriever giu dim=768.
        self.retriever = Neo4jRetriever(
            uri=uri, user=user, password=password,
            database=database, embedding_dim=768,
        )

    def build_kg_batch(self, article_ids: List[str]) -> Optional[List[Optional[Dict[str, Any]]]]:
        """
        article_ids: list article_id cua ca batch.
        Tra ve list subgraph (moi sample 1 phan tu) hoac None neu OntoKG tat.
        """
        if not self.enabled or self.retriever is None:
            return None

        kg_batch = []
        for aid in article_ids:
            try:
                sg = self.retriever.get_article_subgraph(aid)
                if sg["uris"]:
                    kg_batch.append(self.retriever.subgraph_to_torch(sg))
                else:
                    kg_batch.append(None)
            except Exception:
                kg_batch.append(None)
        return kg_batch

    def close(self):
        if self.retriever is not None:
            self.retriever.close()