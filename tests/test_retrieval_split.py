"""Q28: collections routes and indexing live in manager/retrieval.py.

The absence checks are what make a stale test patch loud: once app.py no
longer has these names, patch("manager.app.EmbeddingsClient") raises
AttributeError instead of silently patching nothing and letting a test build
a real embeddings client.
"""
import manager.app as app_module


def test_app_no_longer_holds_retrieval_names():
    for name in ("EmbeddingsClient", "VectorDB", "index_collection",
                 "load_collections_config", "get_children",
                 "_collection_indexer", "_prune_orphaned_collections"):
        assert not hasattr(app_module, name), name


def test_retrieval_module_provides_the_router_and_hooks():
    from manager import retrieval
    assert callable(retrieval.make_router)
    assert callable(retrieval.start_retrieval)
    assert callable(retrieval.stop_retrieval)


def test_collection_routes_are_unchanged(collection_config):
    from manager.app import create_app
    app = create_app(collection_config)
    routes = [(r.path, sorted(r.methods)) for r in app.routes
              if r.path.startswith("/collections")]
    assert routes == [
        ("/collections", ["GET"]),
        ("/collections/{collection_id}/search", ["POST"]),
        ("/collections/{collection_id}/docs/{doc_id:path}", ["GET"]),
        ("/collections/{collection_id}/reindex", ["POST"]),
        ("/collections/{collection_id}/reindex/status", ["GET"]),
        ("/collections/{collection_id}/reindex/{job_id}", ["GET"]),
    ]
