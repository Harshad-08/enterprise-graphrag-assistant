"""Documented Cypher used by :class:`app.graph.neo4j_client.Neo4jGraphStore`.

Relationship types and labels cannot be query parameters in Cypher, so the few
statements that need them are formatted with values from fixed whitelists
(``ENTITY_LABELS`` / ``TRAVERSABLE_RELATIONSHIPS``) — never with user input.
"""

# --- Construction ----------------------------------------------------------------

UPSERT_DOCUMENT = """
MERGE (d:Document {document_id: $document_id})
SET d += $props
"""

UPSERT_CHUNKS = """
MATCH (d:Document {document_id: $document_id})
UNWIND $chunks AS c
MERGE (ch:Chunk {chunk_id: c.chunk_id})
SET ch.document_id = c.document_id, ch.text = c.text, ch.chunk_index = c.chunk_index,
    ch.page = c.page, ch.section = c.section, ch.content_hash = c.content_hash,
    ch.embedding = c.embedding
MERGE (d)-[:CONTAINS]->(ch)
"""

# {label} is one of ENTITY_LABELS.
UPSERT_ENTITIES = """
UNWIND $entities AS e
MERGE (n:Entity {{entity_id: e.entity_id}})
SET n:{label}, n.name = e.name,
    n.aliases = reduce(acc = coalesce(n.aliases, []), a IN e.aliases |
                       CASE WHEN a IN acc THEN acc ELSE acc + a END)
"""

UPSERT_MENTIONS = """
UNWIND $mentions AS m
MATCH (ch:Chunk {chunk_id: m.chunk_id})
MATCH (e:Entity {entity_id: m.entity_id})
MERGE (ch)-[:MENTIONS]->(e)
"""

UPSERT_DEFINES = """
MATCH (d:Document {document_id: $document_id})
MATCH (e:Entity {entity_id: $entity_id})
MERGE (d)-[:DEFINES]->(e)
"""

# {rel_type} is one of TRAVERSABLE_RELATIONSHIPS. Provenance lists record every
# document/chunk that asserts the relationship so deletion can retract it.
UPSERT_RELATIONS = """
UNWIND $relations AS r
MATCH (a:Entity {{entity_id: r.source_id}})
MATCH (b:Entity {{entity_id: r.target_id}})
MERGE (a)-[x:{rel_type}]->(b)
SET x.source_documents = CASE WHEN r.document_id IN coalesce(x.source_documents, [])
                              THEN x.source_documents
                              ELSE coalesce(x.source_documents, []) + r.document_id END,
    x.source_chunks = CASE WHEN r.chunk_id IS NULL OR r.chunk_id IN coalesce(x.source_chunks, [])
                           THEN coalesce(x.source_chunks, [])
                           ELSE coalesce(x.source_chunks, []) + r.chunk_id END,
    x.evidence = coalesce(x.evidence, r.evidence)
"""

UPSERT_CLAIMS = """
MATCH (d:Document {document_id: $document_id})
UNWIND $claims AS c
MERGE (cl:Claim {claim_id: c.claim_id})
SET cl += c.props
MERGE (d)-[:CONTAINS]->(cl)
WITH cl, c
MATCH (ch:Chunk {chunk_id: c.props.chunk_id})
MERGE (cl)-[:EXTRACTED_FROM]->(ch)
WITH cl, c
OPTIONAL MATCH (e:Entity {entity_id: c.props.subject_id})
FOREACH (_ IN CASE WHEN e IS NULL THEN [] ELSE [1] END | MERGE (cl)-[:REFERS_TO]->(e))
"""

CLEAR_SUPERSESSIONS = "MATCH (:Document)-[s:SUPERSEDES]->(:Document) DELETE s"

SET_SUPERSESSIONS = """
UNWIND $edges AS edge
MATCH (a:Document {document_id: edge[0]})
MATCH (b:Document {document_id: edge[1]})
MERGE (a)-[:SUPERSEDES]->(b)
"""

# {rel_type} is one of CLAIM_RELATIONSHIPS.
UPSERT_CLAIM_RELATIONS = """
UNWIND $relations AS r
MATCH (a:Claim {{claim_id: r.claim_a_id}})
MATCH (b:Claim {{claim_id: r.claim_b_id}})
MERGE (a)-[x:{rel_type}]->(b)
SET x += r
"""

# --- Deletion --------------------------------------------------------------------

DELETE_DOCUMENT_NODES = """
MATCH (d:Document {document_id: $document_id})
OPTIONAL MATCH (d)-[:CONTAINS]->(x)
DETACH DELETE x, d
RETURN count(d) AS deleted
"""

RETRACT_RELATION_PROVENANCE = """
MATCH (:Entity)-[r]->(:Entity)
WHERE $document_id IN coalesce(r.source_documents, [])
SET r.source_documents = [x IN r.source_documents WHERE x <> $document_id],
    r.source_chunks = [x IN coalesce(r.source_chunks, []) WHERE NOT x STARTS WITH $chunk_prefix]
WITH r WHERE size(r.source_documents) = 0
DELETE r
"""

DELETE_ORPHAN_ENTITIES = "MATCH (e:Entity) WHERE NOT (e)--() DELETE e"

# --- Retrieval -------------------------------------------------------------------

# Neo4j returns cosine scores normalised to [0, 1] as (1 + cos) / 2.
VECTOR_SEARCH = """
CALL db.index.vector.queryNodes($index_name, $k, $embedding) YIELD node, score
MATCH (d:Document)-[:CONTAINS]->(node)
WHERE ($department IS NULL OR d.department = $department)
  AND ($doc_type IS NULL OR d.doc_type = $doc_type)
  AND ($statuses IS NULL OR d.status IN $statuses)
RETURN node {.chunk_id, .document_id, .text, .chunk_index, .page, .section, .content_hash} AS chunk, score
ORDER BY score DESC
"""

# Graph recovery / multi-hop traversal. Paths may *end* at a Department but not
# pass through one (departments own many artifacts and would connect unrelated
# policies). {rel_types} and {max_hops} come from whitelisted configuration.
TRAVERSE = """
MATCH (seed:Entity) WHERE seed.entity_id IN $seed_ids
MATCH path = (seed)-[:{rel_types}*1..{max_hops}]-(other:Entity)
WHERE other <> seed
  AND none(n IN nodes(path)[1..-1] WHERE n:Department)
  AND all(i IN range(0, size(nodes(path)) - 2) WHERE NOT nodes(path)[i] IN nodes(path)[i + 1..])
WITH path ORDER BY length(path) ASC LIMIT $limit
WITH path, nodes(path) AS ns, relationships(path) AS rs
OPTIONAL MATCH (doc:Document)-[:DEFINES]->(defined) WHERE defined IN ns
WITH path, ns, rs, collect(DISTINCT doc.document_id) AS defining_docs
RETURN [n IN ns | n.name] AS names,
       [n IN ns | n.entity_id] AS ids,
       [i IN range(0, size(rs) - 1) |
          CASE WHEN startNode(rs[i]) = ns[i] THEN type(rs[i]) ELSE '<' + type(rs[i]) END] AS rels,
       reduce(acc = [], r IN rs | acc + coalesce(r.source_documents, [])) + defining_docs AS docs
ORDER BY size(rs) ASC
"""

CHUNKS_MENTIONING = """
MATCH (ch:Chunk)-[:MENTIONS]->(e:Entity) WHERE e.entity_id IN $entity_ids
RETURN DISTINCT ch {.chunk_id, .document_id, .text, .chunk_index, .page, .section, .content_hash} AS chunk
ORDER BY chunk.chunk_id LIMIT $limit
"""

RELATION_CHUNKS = """
MATCH (a:Entity)-[r]-(b:Entity) WHERE a.entity_id IN $entity_ids
UNWIND coalesce(r.source_chunks, []) AS chunk_id
MATCH (ch:Chunk {chunk_id: chunk_id})
RETURN DISTINCT ch {.chunk_id, .document_id, .text, .chunk_index, .page, .section, .content_hash} AS chunk
ORDER BY chunk.chunk_id LIMIT $limit
"""

CHUNKS_FOR_DOCUMENTS = """
MATCH (d:Document)-[:CONTAINS]->(ch:Chunk) WHERE d.document_id IN $document_ids
RETURN ch {.chunk_id, .document_id, .text, .chunk_index, .page, .section, .content_hash} AS chunk
ORDER BY ch.chunk_id LIMIT $limit
"""
