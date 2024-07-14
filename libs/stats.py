import os

from pymongo import MongoClient

from libs.models import RepoCrawlStats

mongo_connection_string = f'mongodb://{os.environ["MONGO_HOST"]}:{os.environ["MONGO_PORT"]}/stats'


class NoStatsFound(Exception):
    """Raised when no stats are found in the statistics database. This can happen when we have
    vector data, but missing history of a run"""


class CrawlStats:
    """
    A class for ORM bindings for crawl statistics and operations.
    """

    def __init__(self, connection_string=mongo_connection_string):
        self.client = MongoClient(connection_string)
        self.db = self.client.get_default_database()
        self.collection = self.db['stats']

    def update_finished_crawl(self, stats, collection_name):
        """Save the finished crawl stats to the database.

        Args:
            stats (dict): A dictionary containing the statistics of the finished crawl.
            collection_name (str): The name of the collection in the database where the statistics
            should be logged.

        Returns:
            ObjectId: The generated unique identifier of the inserted document.
        """
        collection = self.db[collection_name]
        result = collection.insert_one(stats)
        return result.inserted_id

    def get_repo_stats(self, repo):
        """Get the repo crawl stats.

        Args:
            repo (str): The name of the repository.

        Returns:
            If repo stats exist, return a RepoMetadata.
        """
        repo_crawl_stats = self.collection.find_one(
            {'_id': repo},
        )

        if repo_crawl_stats:
            return RepoCrawlStats.model_validate(repo_crawl_stats)
        else:
            raise NoStatsFound()

    def update_crawl_stats(self, repo_id, metadata):
        """Update the stats for this repository.

        Repo name.
        Repo picture.
        The repo's description (if it has one).
        Last time it was updated by the crawler.
        Time the repo was added to our list of repos.
        A unique id for that repo.

        Args:
            repo_id (str): The name or identifier of the repository.
            metadata (RepoCrawlStats): the metadata info for this repo/branch combo.

        """
        metadata = metadata.model_dump()

        # todo: might wanna refactor this somehow. not keen on littering with random assignments
        metadata['repo_id'] = repo_id

        self.collection.update_one(
            {
                '_id': repo_id
            },
            {
                '$set': metadata
            },
            upsert=True
        )

    def get_repos(self):
        """Get the list of repositories, together will their crawl stats"""
        return [RepoCrawlStats.model_validate(doc) for doc in self.collection.find()]
