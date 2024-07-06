import logging

import httpx
from httpx import AsyncClient

logger = logging.getLogger(__name__)


class OptimizedAsyncClient(AsyncClient):
    """Wrapper for httpx.AsyncClient to provide an interface for keeping SSL handshakes warm"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, http2=True)
        self.warmed_up_hosts = set()

    async def warmup_if_needed(self, url, headers):
        host = httpx.URL(url).host
        if host not in self.warmed_up_hosts:

            logger.debug(f'Warming up host: {host}')
            # todo this might fail, so maybe need to check that before assuming it's warm
            self.warmed_up_hosts.add(host)
            return await self.head(url, headers=headers)
        else:
            logger.debug(f'Skipping warmup for host: {host}')
