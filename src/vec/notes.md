
Sources:
- https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/li-dialogue-2019.pdf
- spacy:
    - Pretrain
    - Taken from: https://arxiv.org/ftp/arxiv/papers/1910/1910.11241.pdf :
        In order to improve the performance of transfer learning models
        further, we employed a newly released spaCy package feature, that
        of pre-training. Pre-training allows us to initialize the neural
        network layers of spaCy’s CNN layers with a custom vector layer.
        This custom vector can be trained by utilizing a domain specific
        text corpus using the spaCy library pre-training command [12].
        The pre-training API spaCy has implemented a deep learning
        implementation for obtaining dynamic word embeddings using a
        Language Modelling with Approximate Outputs (LMAO)
        described in spaCy Language model pretraining [25].
        We leveraged spaCy pre-training API and trained our custom
        dynamic embedding model over our domain specific text corpus. 


Git:
- https://github.com/julianser/hed-dlg-truncated
