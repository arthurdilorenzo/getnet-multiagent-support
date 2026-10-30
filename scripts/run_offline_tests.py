"""Run selected pytest contracts with network denied and dotenv disabled."""
import os
from pathlib import Path
import secrets
import shutil
import socket
import sys
import tempfile
from pydantic_settings import BaseSettings

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.dont_write_bytecode = True
original_init = BaseSettings.__init__
def init(self, *args, **kwargs):
    kwargs['_env_file'] = None
    return original_init(self, *args, **kwargs)
BaseSettings.__init__ = init
def denied(*args, **kwargs):
    raise RuntimeError('OFFLINE_NETWORK_DENIED')
socket.socket.connect = denied
socket.socket.connect_ex = denied
socket.create_connection = denied
def main():
    with tempfile.TemporaryDirectory(prefix='getnet_offline_') as folder:
        scratch = Path(folder)
        for source in (root/'data').iterdir():
            if source.suffix in {'.md', '.json'} and source.is_file():
                shutil.copyfile(source, scratch/source.name)
        os.environ.update(DATA_DIR=folder, SQLITE_DB_PATH=str(scratch/'state.db'),
            VECTOR_INDEX_PATH=str(scratch/'vector_index.json'), GEMINI_API_KEY='',
            GETNET_LIVE_EMBEDDINGS='0', APP_ENV='development',
            AUTH_API_KEY=secrets.token_urlsafe(48), JWT_SECRET=secrets.token_urlsafe(48),
            TELEGRAM_BOT_TOKEN='', TELEGRAM_CHAT_ID='', HITL_WEBHOOK_URL='')
        for name in ('vector_index.json', 'embeddings_cache.json'):
            shutil.copyfile(root/'data'/name, scratch/name)
        from app.rag.embeddings import SemanticEmbedder
        SemanticEmbedder.CACHE_PATH = scratch/'embeddings_cache.json'
        os.chdir(root)
        import pytest
        raise SystemExit(pytest.main(sys.argv[1:]))

if __name__ == '__main__':
    main()
