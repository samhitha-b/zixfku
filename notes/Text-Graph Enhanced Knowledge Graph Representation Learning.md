## Review of Representation Learning

- The problem: representation of KG embeddings, no specific models

### Translation-based and Semantic Matching models
- TransE (represented relations as translations from head to tail entity), TransH (hyperplane representation), TransD (dot product among vectors), RotatE (rotation), TransG (gaussian random variable considering uncertainties of entities), TransR (represent embeddings in specific space).
- Semantic Matching: ConvE (convolutional layer scoring function), GCN
- Problem: considered only KG triplets, suffer from _structure sparsity_.

### Text-enhanced KG Representation
- Various approaches: average word embeddings, BERT encoded entities, concise descriptions of entity names, GANs to encode entities based merely on noisy data, entity descriptions and relationships
- Problem: 
	- Only local short description consecutive word relationships captured, not global co-occurences
	- Use CNN and LSTM-based, good for semantics, not for long-range correlations

# Proposed Solution
Stages:
1. Triplet embedding
2. Auxiliary text encoding (creating text-graph then applying GCN for neighbouring semantic information)
3. KG representation fusion (GCN encodings integrated with triplet encodings)

## Triplet Embedding
- Triplet can be thought of as a (subject, object, predicate) or a (head, tail, relationship) relationship in the context of a sentence. 
- For eg: $\text{Delhi is located in India.} \to \text{(Delhi, India, locat)}$
- used TransE
- check semantic relationships by how much closer head entity plus relationship is to the tail entity
- $$f (h, r, t) = -||h + r − t||^2$$
- entities are closer in semantic meaning when this distance is less

## Auxiliary Text encoding
- Text graph = entity-word graph
- entity (named entity, important special noun, subject/object) vs word (NLP unit, there can be many words in an entity)
- Text Graph construction:
	- $G = \{V, E\}$, where $V = \text{nodes}$ given by $E \text{(entities)} + W \text{(words)}$; $E = \text{edges}$
	- For every entity $e$, we get first $k$ words with highest TF-IDF

## KG Representation Fusion
- We calculate a convolution to integrate the entity embeddings of structural triplets ($e_s$) and those of auxiliary texts ($e_d$) using the gating vector ($g_e$) through the relation: 
	$$e = g_e \odot e_s + (1 − g_e) \odot e_d$$


## End-to-end Model Training
- The loss function $L$ is calculated as the margin between collection of correct samples ($S$) and the collection of incorrect triplets ($S'$) by generating negative samples using a sampling strategy "bern".
	$$L = \sum_{(h,r,t)\in S} \sum_{(h',r,t')\in S'} max(\gamma	 + f (h, r, t) − f (h', r, t'), 0)$$
- $f$ is a scoring function defined for a particular triplet also defined in the paper. 
- Optimizers like Adam are also used.